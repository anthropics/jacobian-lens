#!/usr/bin/env bash
# Runs ON the vast.ai instance. Prepares the environment and starts the J-lens dashboard.
# Idempotent: safe to re-run. Re-running skips finished stages and restarts the server.
set -euo pipefail

LOG_DIR=/var/log/portal
APP_LOG="$LOG_DIR/jlens-app.log"
SETUP_LOG="$LOG_DIR/jlens-setup.log"
# Own log file: the onstart script that calls this one already tees its stdout to
# $SETUP_LOG, and teeing to the same path from here writes every line twice.
BOOTSTRAP_LOG="$LOG_DIR/jlens-bootstrap.log"
REAPER_LOG="$LOG_DIR/jlens-reaper.log"
PID_FILE=/var/run/jlens.pid
REAPER_PID_FILE=/var/run/jlens-reaper.pid
STAMP_DIR=/workspace/.jlens
DEPS_STAMP="$STAMP_DIR/deps.ok"
# Sentinels the laptop-side scripts can read over SSH: a bootstrap failure is
# otherwise invisible while the instance keeps billing.
READY_FILE="$STAMP_DIR/READY"
FAILED_FILE="$STAMP_DIR/FAILED"

mkdir -p "$LOG_DIR" "$STAMP_DIR"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1
rm -f "$READY_FILE" "$FAILED_FILE"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"

# Exported under the names dashboard.py reads, so the pre-download and the server
# cannot end up pointed at different repos.
export JLENS_MODEL_ID="${JLENS_MODEL_ID:-google/gemma-4-12B-it}"
export JLENS_LENS_REPO="${JLENS_LENS_REPO:-lamm-mit/gemma4-jacobian-lenses}"
export JLENS_LENS_FILE="${JLENS_LENS_FILE:-gemma4-12b-it/paper/seed0.pt}"
# The revision seed0.pt was fitted against, read from its own .pt.meta.json
# (model.model_revision). Pinned so an upstream re-upload of google/gemma-4-12B-it
# cannot silently put the lens and the served model out of step - this branch's
# whole claim is that there is no lens/model drift to caveat.
export JLENS_MODEL_REVISION="${JLENS_MODEL_REVISION:-707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"

export JLENS_PRECISION="${JLENS_PRECISION:-bf16}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_XET_HIGH_PERFORMANCE=1   # hf_transfer is gone in huggingface_hub 1.x; this replaces it
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

T0=$(date +%s)
T_STAGE=$T0
stage() {
  local now; now=$(date +%s)
  printf '\n========== %s | stage %ds | total %ds ==========\n' "$1" "$((now - T_STAGE))" "$((now - T0))"
  T_STAGE=$now
}
die() { printf '%s\n' "$1" > "$FAILED_FILE"; printf '\nFATAL: %s\n' "$1" >&2; exit 1; }

echo "[$(date -Is)] bootstrap begin | repo=$REPO_ROOT precision=$JLENS_PRECISION HF_HOME=$HF_HOME"

[ -f "$DEPLOY_DIR/dashboard.py" ] || die "dashboard.py missing from $DEPLOY_DIR - the pushed repo is incomplete or stale."

# The image ships python in a conda env that is not on PATH for non-interactive shells.
if [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
command -v python >/dev/null || die "no python on PATH after venv activation"
echo "python: $(python -V) at $(command -v python)"

# ---------------------------------------------------------------- GPU preflight
# Fail before downloading 24GB if the card cannot hold the model.
# 11,959,730,224 params. The lens file is 703MiB and roughly doubles resident (~1.5GB).
case "$JLENS_PRECISION" in
  nf4)  MIN_VRAM_GB=18 ;;   # ~8GB weights + 1.5GB lens + transients
  int8) MIN_VRAM_GB=22 ;;   # ~13GB weights + 1.5GB lens + transients
  bf16) MIN_VRAM_GB=36 ;;   # ~24GB weights + 1.5GB lens; fits a 48GB card, unlike Qwen
  *)    die "JLENS_PRECISION must be one of nf4|int8|bf16 (got '$JLENS_PRECISION')" ;;
esac

python - "$MIN_VRAM_GB" "$JLENS_PRECISION" <<'PY' || die "GPU preflight failed (see message above)"
import sys
import torch

min_gb, precision = float(sys.argv[1]), sys.argv[2]
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
if not torch.cuda.is_available():
    print(
        "CUDA is not available. This instance cannot run the dashboard.\n"
        "Check: the offer actually has a GPU attached, and the container was started\n"
        "from a CUDA image (vastai/pytorch:cuda-*). Destroy and relaunch on another offer.",
        file=sys.stderr,
    )
    raise SystemExit(1)
name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {name} | {total_gb:.1f} GiB | sm_{cap[0]}{cap[1]}")
if cap[0] < 8:
    print(
        f"{name} is sm_{cap[0]}{cap[1]} (pre-Ampere). bitsandbytes NF4 needs sm_80+.\n"
        "Destroy this instance and pick an offer with compute_cap >= 800.",
        file=sys.stderr,
    )
    raise SystemExit(1)
if total_gb < min_gb:
    print(
        f"{total_gb:.1f} GiB of VRAM is below the {min_gb:.0f} GiB needed for "
        f"JLENS_PRECISION={precision}.\n"
        "Either destroy and relaunch on a bigger card, or re-run with "
        "JLENS_PRECISION=nf4 (needs ~32 GiB).",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
stage "GPU preflight OK"

# ------------------------------------------------------------------- dependencies
if [ -f "$DEPS_STAMP" ]; then
  echo "deps already installed (remove $DEPS_STAMP to force reinstall)"
else
  # No causal_conv1d / fla: the Gated DeltaNet fast kernels are optional, the torch
  # fallback costs ~1-2s on a prefill-only workload, and causal-conv1d is sdist-only
  # (10-30min of billed CUDA compile). --no-deps on jlens: its unpinned `torch`
  # dependency would otherwise let pip replace the image's CUDA build with a CPU wheel.
  python -m pip install --no-cache-dir \
    'transformers==5.10.1' \
    'bitsandbytes>=0.46.1' \
    'accelerate>=1.1.0' \
    'huggingface_hub>=1.5.0,<2.0' \
    'hf-xet>=1.5.2' \
    'numpy>=1.17' \
    fastapi \
    'uvicorn[standard]'
  python -m pip install --no-cache-dir --no-deps -e "$REPO_ROOT"
  touch "$DEPS_STAMP"
fi
# The stamp lives on /workspace but site-packages live on the container filesystem,
# so the two can diverge. Say what fixes it instead of dying on a bare traceback.
python -c 'import transformers, bitsandbytes, jlens; print("transformers", transformers.__version__, "| bitsandbytes", bitsandbytes.__version__, "| jlens ok")' \
  || die "dependencies are stamped installed but do not import. Remove $DEPS_STAMP and re-run this script."
stage "dependencies ready"

# ---------------------------------------------------------------------- downloads
# Both repos are public; HF_TOKEN is optional and only raises the anonymous rate limit.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set; using it for downloads"
fi

# hf download is a no-op on a warm cache, so this whole block is idempotent.
# An `if`, not `[ -n ... ] && ...`: under `set -e` an AND-list that tests false is a
# trap this repo has already been bitten by once.
REV_ARGS=()
if [ -n "$JLENS_MODEL_REVISION" ]; then
  REV_ARGS=(--revision "$JLENS_MODEL_REVISION")
fi
hf download "$JLENS_MODEL_ID" "${REV_ARGS[@]}" --exclude '*.pth' '*.bin' '*.gguf' \
  || die "model download failed. Check network and that $JLENS_MODEL_ID is still public (gated repos need HF_TOKEN)."
stage "model downloaded ($JLENS_MODEL_ID${JLENS_MODEL_REVISION:+ @ $JLENS_MODEL_REVISION})"

hf download "$JLENS_LENS_REPO" "$JLENS_LENS_FILE" \
  || die "lens download failed. Check that $JLENS_LENS_REPO/$JLENS_LENS_FILE still exists."
stage "lens downloaded ($JLENS_LENS_FILE)"

# ------------------------------------------------------------------------- server
# Only kill a pid that is still the dashboard: /var/run may not be a tmpfs, so a
# stale pid file can name whatever process inherited that number after a restart.
OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -n "$OLD_PID" ] && grep -qa uvicorn "/proc/$OLD_PID/cmdline" 2>/dev/null; then
  echo "stopping previous server (pid $OLD_PID)"
  kill "$OLD_PID" 2>/dev/null || true
  sleep 3
fi

# Bind loopback only: reachable through the SSH tunnel, invisible to the public internet.
# setsid + nohup so it outlives this script and any SSH session. The server records its
# own pid because $! would be setsid's pid if setsid forks rather than execs.
rm -f "$PID_FILE"
cd "$DEPLOY_DIR"
JLENS_PID_FILE="$PID_FILE" setsid nohup bash -c \
  'echo $$ > "$JLENS_PID_FILE"; exec python -m uvicorn dashboard:app --host 127.0.0.1 --port 7860' \
  >> "$APP_LOG" 2>&1 < /dev/null &
disown || true   # a spurious "nothing to disown" must not abort a successful launch

# dashboard.py loads the model on a background thread, so uvicorn binds within a
# second or so. Probe the real endpoint: a bad module name shows up here as a
# specific failure instead of thirty minutes later as a dead port.
for _ in $(seq 30); do
  curl -sf -o /dev/null http://127.0.0.1:7860/status && break
  sleep 1
done
curl -sf -o /dev/null http://127.0.0.1:7860/status \
  || die "server never answered on 127.0.0.1:7860. Last lines of $APP_LOG:
$(tail -n 30 "$APP_LOG")"

stage "server started"

# ---------------------------------------------------------------------- dead man
# Nothing on the laptop can be relied on to stop the burn: closing the lid or killing
# the tunnel leaves the instance billing. This watchdog destroys the instance after
# JLENS_IDLE_KILL_S with no dashboard traffic ($APP_LOG mtime is the heartbeat -
# uvicorn writes an access line per request).
#
# It stays inert until a vast API key is dropped on the instance. The key deliberately
# does not travel in the onstart script (vast stores that server-side) and is never
# echoed here. Arm it from the laptop, no re-run needed:
#   scp -P <port> ~/.config/vastai/vast_api_key root@<host>:/workspace/.jlens/vast_api_key
KEY_FILE="$STAMP_DIR/vast_api_key"
ID_FILE="$STAMP_DIR/instance_id"
export JLENS_IDLE_KILL_S="${JLENS_IDLE_KILL_S:-7200}"

cat > "$STAMP_DIR/reaper.sh" <<'REAPER'
#!/usr/bin/env bash
set -uo pipefail
while sleep 300; do
  [ -s "$KEY_FILE" ] || continue
  ID="${CONTAINER_ID:-$(cat "$ID_FILE" 2>/dev/null || true)}"
  [ -n "$ID" ] || { echo "[$(date -Is)] armed but no instance id (CONTAINER_ID unset and $ID_FILE missing)"; continue; }
  LAST="$(stat -c %Y "$APP_LOG" 2>/dev/null || date +%s)"
  IDLE=$(( $(date +%s) - LAST ))
  [ "$IDLE" -ge "$JLENS_IDLE_KILL_S" ] || continue
  echo "[$(date -Is)] idle ${IDLE}s >= ${JLENS_IDLE_KILL_S}s: destroying instance $ID"
  curl -sS -X DELETE -H "Authorization: Bearer $(cat "$KEY_FILE")" \
    -H 'Content-Type: application/json' -d '{}' \
    "https://console.vast.ai/api/v0/instances/$ID/"
done
REAPER

OLD_REAPER="$(cat "$REAPER_PID_FILE" 2>/dev/null || true)"
if [ -n "$OLD_REAPER" ] && grep -qa reaper.sh "/proc/$OLD_REAPER/cmdline" 2>/dev/null; then
  kill "$OLD_REAPER" 2>/dev/null || true
fi
JLENS_REAPER_PID_FILE="$REAPER_PID_FILE" KEY_FILE="$KEY_FILE" ID_FILE="$ID_FILE" APP_LOG="$APP_LOG" \
  setsid nohup bash -c 'echo $$ > "$JLENS_REAPER_PID_FILE"; exec bash '"$STAMP_DIR"'/reaper.sh' \
  >> "$REAPER_LOG" 2>&1 < /dev/null &
disown || true

touch "$READY_FILE"

cat <<EOF

Dashboard listening on 127.0.0.1:7860 (pid $(cat "$PID_FILE")).
The model is still loading (~15 min); /status reports progress, the page polls it,
and /run answers 503 until it is ready.

  app log        tail -f $APP_LOG
  bootstrap log  tail -f $BOOTSTRAP_LOG
  setup log      tail -f $SETUP_LOG

Idle self-destruct: $([ -s "$KEY_FILE" ] && echo "ARMED, ${JLENS_IDLE_KILL_S}s of no traffic destroys this instance" || echo "NOT ARMED. scp a vast API key to $KEY_FILE to arm it.")

Reach it from your laptop with the tunnel script, then open http://localhost:7860
EOF
