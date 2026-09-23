#!/usr/bin/env bash
# Rent a vast.ai GPU and bring the J-lens dashboard up on it.
#   ./provision.sh            search, confirm, create, wait
#   ./provision.sh --yes      no interactive confirmation
#   ./provision.sh --offer N  use a specific offer id (pick one from the printed table)
#   ./provision.sh --status   state / uptime / accrued cost of the current instance
set -euo pipefail

API="https://console.vast.ai/api/v0"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$DEPLOY_DIR/.instance"
KNOWN_HOSTS="$DEPLOY_DIR/.known_hosts"   # shared with tunnel.sh
REPO_URL="https://github.com/Nathan2589/jacobian-lens"
# Which branch the instance clones. One model per branch (deploy/MODEL-BRANCHES.md),
# so the checked-out branch is the default: provisioning from model/gemma-4-12b-it
# should not silently rent a card sized for main's Qwen config.
BRANCH="${JLENS_BRANCH:-$(git -C "$DEPLOY_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
IMAGE="${JLENS_IMAGE:-vastai/pytorch:2.9.1-cu128-cuda-12.9-mini-py311-2026-08-21}"
DISK_GB=60
SSH_KEY="${JLENS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
VASTAI="${VASTAI:-$HOME/.local/bin/vastai}"
CREDIT_FLOOR="${JLENS_CREDIT_FLOOR:-2}"
MIN_RAM_MB=12000    # host RAM the bootstrap needs: ~1.5GB lens + single-shard loading
READY_TIMEOUT=900   # seconds to wait for the container to reach "running"
PORTS_GRACE=120     # seconds to wait for the direct port map before using the proxy
SSH_TIMEOUT=300     # seconds to wait for sshd after that

die() { echo "FATAL: $*" >&2; exit 1; }

command -v jq >/dev/null || die "jq not installed"
command -v curl >/dev/null || die "curl not installed"

# A detached HEAD has no branch name to clone, and guessing "main" here would rent a
# card sized for the wrong model. Make the operator say which branch they meant.
if [ "$BRANCH" = "HEAD" ]; then
  die "detached HEAD: no branch to clone. Check out a branch, or set JLENS_BRANCH=<branch>."
fi

KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
KEY="${KEY//[$'\n\r']/}"
[ -n "$KEY" ] || die "no vast.ai API key"

# The key goes in on stdin, never in argv - argv is world-readable via /proc.
# Non-2xx and empty bodies are errors: vast returns HTTP 200 + '"instances": null'
# for a missing instance, so an empty response really does mean the call failed.
api() { # api METHOD PATH [JSON_BODY] -> body on stdout
  local method="$1" path="$2" out code body
  local -a args=(-sS -X "$method" -H @- -w '\n%{http_code}')
  [ "$#" -ge 3 ] && args+=(-H 'Content-Type: application/json' -d "$3")
  out="$(printf 'Authorization: Bearer %s\n' "$KEY" | curl "${args[@]}" "$API$path")" \
    || { echo "curl failed on $method $path" >&2; return 1; }
  code="${out##*$'\n'}"; body="${out%$'\n'*}"
  case "$code" in 2*) ;; *) echo "HTTP $code from $method $path: $body" >&2; return 1 ;; esac
  [ -n "$body" ] || { echo "empty body from $method $path" >&2; return 1; }
  printf '%s' "$body"
}

# ---------------------------------------------------------------- args
MODE=create
OFFER_ID=""
ASSUME_YES=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --status) MODE=status; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --offer)  OFFER_ID="${2:-}"
              [[ "$OFFER_ID" =~ ^[0-9]+$ ]] || die "--offer needs a numeric offer id"
              shift 2 ;;
    -h|--help) sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ---------------------------------------------------------------- status
if [ "$MODE" = status ]; then
  [ -f "$STATE_FILE" ] || die "no state file at $STATE_FILE - nothing provisioned from here"
  ID="$(cat "$STATE_FILE")"
  INST="$(api GET "/instances/$ID/?owner=me" | jq '.instances')"
  [ "$INST" != "null" ] || die "instance $ID no longer exists (already destroyed?)"
  START="$(jq -r '.start_date // empty' <<<"$INST")"
  [ -n "$START" ] || START="$(stat -c %Y "$STATE_FILE")"
  jq -r --argjson now "$(date +%s)" --argjson start "$START" '
    "instance    : \(.id)
gpu         : \(.num_gpus)x \(.gpu_name)
rate        : $\(.dph_total)/hr
state       : \(.actual_status) (contract \(.cur_state), target \(.intended_status))
message     : \(.status_msg // "-" | rtrimstr("\n"))
uptime      : \((($now - $start) / 3600 * 100 | floor) / 100) hr
accrued     : $\((($now - $start) / 3600 * .dph_total * 100 | floor) / 100) (estimate)"' <<<"$INST"
  echo
  echo "destroy with: $DEPLOY_DIR/destroy.sh"
  exit 0
fi

# ---------------------------------------------------------------- preflight
[ -x "$VASTAI" ] || die "vastai CLI not found at $VASTAI (set VASTAI=/path/to/vastai)"
[ -f "$SSH_KEY" ] || die "ssh key not found at $SSH_KEY"

# Ask the account, not just the state file - a forgotten instance from another
# session (or one whose .instance was deleted by hand) is exactly the thing that
# quietly eats the budget. Note a *stopped* instance still bills for its disk.
LIVE="$(api GET "/instances/?owner=me" | jq '[.instances // [] | .[]]')"
if [ "$(jq 'length' <<<"$LIVE")" -gt 0 ]; then
  echo "you already have instances on this vast.ai account:" >&2
  jq -r '.[] | "  \(.id)  \(.gpu_name)  $\(.dph_total)/hr  \(.actual_status)  label=\(.label // "-")"' <<<"$LIVE" >&2
  die "destroy them first ($DEPLOY_DIR/destroy.sh --list). Stopped instances still bill for disk."
fi
rm -f "$STATE_FILE"

# The instance clones from GitHub, so the deploy scripts must already be pushed
# and the repo must be public. Catching this here saves a guaranteed-failed rental.
RAW_BASE="${REPO_URL/github.com/raw.githubusercontent.com}/$BRANCH/deploy"
STALE=0
for f in bootstrap.sh dashboard.py; do
  REMOTE="$(curl -sfL "$RAW_BASE/$f")" \
    || die "deploy/$f is not readable at $RAW_BASE/$f - push branch '$BRANCH' with deploy/ in it (and make sure the fork is public) before provisioning."
  [ "$REMOTE" = "$(cat "$DEPLOY_DIR/$f")" ] || { echo "warning: local deploy/$f differs from origin/$BRANCH." >&2; STALE=1; }
done
if [ "$STALE" = 1 ] && [ "$ASSUME_YES" -ne 1 ]; then
  die "the instance runs the PUSHED version, so those local edits would not apply. Push them, or rerun with --yes."
fi

CREDIT="$(api GET "/users/current/" | jq -r '.credit')"
[ "$CREDIT" != "null" ] && [ -n "$CREDIT" ] || die "could not read credit (bad API key?)"
printf 'credit remaining: $%.2f\n' "$CREDIT"
if [ "$(jq -n --argjson c "$CREDIT" --argjson f "$CREDIT_FLOOR" 'if $c < $f then 1 else 0 end')" = 1 ]; then
  die "credit \$$CREDIT is below the \$$CREDIT_FLOOR floor. Top up before provisioning."
fi

# ---------------------------------------------------------------- offers
# inet_down_cost is $/GB (there is no internet_down_cost_per_tb search key - the
# server rejects it with a 400 that the CLI prints to stdout and exits 0 on).
QUERY='gpu_ram >= 40 num_gpus=1 rentable=true reliability > 0.98 disk_space >= 100 compute_cap >= 800 cuda_max_good >= 12.4 direct_port_count >= 2 inet_down_cost < 0.02 inet_down >= 300 dph_total < 0.70'

# gpu_ram/cpu_ram come back in units of 1000 (vast's own offers_mult), not 1024.
# Effective container RAM is cpu_ram * gpu_frac. The gpu_name allow-list keeps out
# Turing and the crippled mining SKUs (CMP 170HX passes compute_cap >= 800 and is
# often the cheapest thing in the list).
NORMALISE='[ .[]
  | select((.cpu_ram * (.gpu_frac // 1)) >= MINRAM)
  | select(.gpu_name | test("^(A40|RTX A6000|RTX 6000|L40S?|RTX 4090|RTX 5090|A100|H100|H200|B200)"))
  | {id, dph: .dph_total, vram: ((.gpu_ram / 1000) | floor),
     gpu: .gpu_name, inet: (.inet_down | floor), geo: (.geolocation // "?"),
     ram: ((.cpu_ram * (.gpu_frac // 1) / 1000) | floor)} ]
  | sort_by(.dph)'
NORMALISE="${NORMALISE/MINRAM/$MIN_RAM_MB}"

echo "searching offers..."
# The CLI exits 0 and prints API errors to stdout, so pipefail buys nothing here:
# validate that what came back is actually a JSON array.
RAW_OFFERS="$("$VASTAI" search offers "$QUERY" --storage "$DISK_GB" -o dph --raw)" || true
jq -e 'type == "array"' >/dev/null 2>&1 <<<"$RAW_OFFERS" \
  || die "vastai search failed: ${RAW_OFFERS:-<no output>}"
MATCHED="$(jq 'length' <<<"$RAW_OFFERS")"
OFFERS="$(jq "$NORMALISE" <<<"$RAW_OFFERS")"
SURVIVED="$(jq 'length' <<<"$OFFERS")"
[ "$SURVIVED" -gt 0 ] \
  || die "$MATCHED offers matched the query, 0 survived the host-RAM (>= ${MIN_RAM_MB}MB) and GPU allow-list filters. Loosen MIN_RAM_MB or NORMALISE in $0."
echo "$MATCHED offers matched, $SURVIVED usable"

printf '\n%-10s %-8s %-6s %-18s %-9s %-7s %s\n' ID '$/HR' VRAM GPU INET_DOWN RAM_GB GEO
jq -r '.[:5][] | [.id, "$\(.dph)", "\(.vram)G", .gpu, "\(.inet)Mb", .ram, .geo] | @tsv' <<<"$OFFERS" \
  | while IFS=$'\t' read -r a b c d e f g; do printf '%-10s %-8s %-6s %-18s %-9s %-7s %s\n' "$a" "$b" "$c" "$d" "$e" "$f" "$g"; done

if [ -n "$OFFER_ID" ]; then
  CHOSEN="$(jq --argjson i "$OFFER_ID" '[.[] | select(.id == $i)] | first' <<<"$OFFERS")"
  [ "$CHOSEN" != null ] || die "offer $OFFER_ID is not in the list above - --offer can only pick from it. Widen QUERY in $0 to reach other offers."
else
  CHOSEN="$(jq '.[0]' <<<"$OFFERS")"
fi

ID="$(jq -r '.id' <<<"$CHOSEN")"
DPH="$(jq -r '.dph' <<<"$CHOSEN")"
RUNWAY="$(jq -n --argjson c "$CREDIT" --argjson d "$DPH" '(($c / $d) * 10 | floor) / 10')"

echo
jq -r '"chosen: offer \(.id)  \(.gpu) \(.vram)GB VRAM  $\(.dph)/hr  \(.geo)"' <<<"$CHOSEN"
echo "branch: $BRANCH  (the instance clones this; it decides which model is served)"
echo "runway: ~${RUNWAY} hours of continuous uptime against \$${CREDIT} of credit"
echo "note:   the \$/hr above already includes ${DISK_GB}GB of storage. Storage bills whether"
echo "        the instance is running or merely stopped - only destroy.sh stops the burn."
echo

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "create this instance and start spending? [y/N] " REPLY \
    || die "no terminal to confirm on - rerun with --yes if you meant to spend money"
  [[ "$REPLY" =~ ^[Yy]([Ee][Ss])?$ ]] || die "aborted, nothing created"
fi

# ---------------------------------------------------------------- create
# onstart is capped at 4048 chars, so it is a bootstrap: clone the repo, hand off.
# bootstrap.sh tees its own output into jlens-bootstrap.log, so only the clone is
# redirected here - redirecting both into one file would write every line twice.
ONSTART="#!/bin/bash
set -euo pipefail
mkdir -p /var/log/portal
{
  echo \"[\$(date -Is)] onstart begin\"
  rm -rf /workspace/jacobian-lens
  git clone --depth 1 --branch $BRANCH $REPO_URL /workspace/jacobian-lens || {
    mkdir -p /workspace/jacobian-lens
    curl -sfL $REPO_URL/archive/refs/heads/$BRANCH.tar.gz | tar xz -C /workspace/jacobian-lens --strip-components=1
  }
} >> /var/log/portal/jlens-setup.log 2>&1
bash /workspace/jacobian-lens/deploy/bootstrap.sh
echo \"[\$(date -Is)] onstart done\" >> /var/log/portal/jlens-setup.log
"

BODY="$(jq -n --arg image "$IMAGE" --arg onstart "$ONSTART" --argjson disk "$DISK_GB" '{
  client_id: "me", image: $image,
  env: {HF_HOME: "/workspace/hf", HF_XET_HIGH_PERFORMANCE: "1", HF_HUB_DISABLE_TELEMETRY: "1"},
  price: null, disk: ($disk | tonumber), label: "jlens", extra: null, onstart: $onstart,
  image_login: null, python_utf8: false, lang_utf8: false, use_jupyter_lab: false,
  jupyter_dir: null, force: false, cancel_unavail: true, template_hash_id: null,
  user: null, runtype: "ssh_direc ssh_proxy"}')"

echo "creating instance from offer $ID..."
CREATED_AT="$(date +%s)"
RESP="$(api PUT "/asks/$ID/" "$BODY")" || RESP=""
INSTANCE_ID="$(jq -r '.new_contract // empty' <<<"$RESP" 2>/dev/null || true)"

if [ -z "$INSTANCE_ID" ]; then
  # The create may have been accepted server-side even though we lost the reply.
  # Never die without checking: an unrecorded live instance bills forever.
  echo "no instance id in the create response - checking whether one was created anyway..." >&2
  ORPHAN="$(api GET "/instances/?owner=me" \
    | jq -r --argjson t "$CREATED_AT" \
        '[.instances // [] | .[] | select(.label == "jlens" and ((.start_date // 0) >= ($t - 60)))] | first | .id // empty' \
    2>/dev/null || true)"
  if [ -n "$ORPHAN" ]; then
    echo "$ORPHAN" > "$STATE_FILE"
    echo "############################################################" >&2
    echo "# instance $ORPHAN WAS created and IS BILLING" >&2
    echo "# destroy it with: $DEPLOY_DIR/destroy.sh" >&2
    echo "############################################################" >&2
  fi
  die "create failed: ${RESP:-<no response>}"
fi

echo "$INSTANCE_ID" > "$STATE_FILE"
echo
echo "############################################################"
echo "# instance $INSTANCE_ID is now BILLING at \$$DPH/hr"
echo "# stop it with:  $DEPLOY_DIR/destroy.sh"
echo "############################################################"
echo

# From here on, any failure or interrupt must be loud - the instance is live.
# INT/TERM must exit explicitly: a trap handler that just returns resumes the
# wait loop, leaving Ctrl-C unable to stop it.
WARNED=0
warn_live() {
  if [ "$WARNED" = 0 ]; then
    WARNED=1
    echo
    echo "!!! instance $INSTANCE_ID IS STILL RUNNING AND BILLING at \$$DPH/hr"
    echo "!!! destroy it with: $DEPLOY_DIR/destroy.sh"
  fi
}
trap warn_live EXIT
trap 'warn_live; exit 130' INT TERM

# ---------------------------------------------------------------- wait
echo "waiting for the container (up to $((READY_TIMEOUT / 60)) min; image pull dominates)..."
DEADLINE=$(( $(date +%s) + READY_TIMEOUT ))
RUNNING_SINCE=0
LAST=""
while :; do
  if ! INST="$(api GET "/instances/$INSTANCE_ID/?owner=me" | jq '.instances')"; then
    echo "  (poll failed, retrying)"; sleep 10; continue
  fi
  [ "$INST" != null ] || die "instance $INSTANCE_ID vanished"
  STATUS="$(jq -r '.actual_status // "creating"' <<<"$INST")"
  MSG="$(jq -r '(.status_msg // "") | rtrimstr("\n")' <<<"$INST")"
  if [ "$STATUS|$MSG" != "$LAST" ]; then echo "  [$STATUS] $MSG"; LAST="$STATUS|$MSG"; fi
  case "$STATUS" in
    running)
      # Ready means running AND the docker port map published. ssh_host alone is
      # always set, so accepting it immediately would silently drop to the slow proxy.
      if jq -e '.ports["22/tcp"]' >/dev/null <<<"$INST"; then break; fi
      [ "$RUNNING_SINCE" != 0 ] || RUNNING_SINCE="$(date +%s)"
      if [ $(( $(date +%s) - RUNNING_SINCE )) -ge "$PORTS_GRACE" ]; then
        echo "  no direct port map after ${PORTS_GRACE}s - falling back to the vast ssh proxy"
        break
      fi
      ;;
    exited|unknown|offline) die "instance reached terminal state '$STATUS' - it will never run. Destroy and retry." ;;
  esac
  # A host that fails to build the image reports 'loading' with the contract already
  # stopped, and sits there until the timeout. Seen as "docker_build() error writing
  # dockerfile". Stop waiting: it never comes up, and it bills meanwhile.
  if [ "$(jq -r '.cur_state // ""' <<<"$INST")" = stopped ] && [[ "$MSG" == *error* ]]; then
    die "host failed to start the container ('$MSG') - it will never run. Destroy and retry; a fresh provision picks a different host."
  fi
  [ "$(date +%s)" -lt "$DEADLINE" ] || die "timed out waiting for 'running'"
  sleep 10
done

# Direct SSH if the port map is published, otherwise the vast proxy.
if jq -e '.ports["22/tcp"]' >/dev/null <<<"$INST"; then
  SSH_HOST="$(jq -r '.public_ipaddr' <<<"$INST")"
  SSH_PORT="$(jq -r '.ports["22/tcp"][0].HostPort' <<<"$INST")"
else
  SSH_HOST="$(jq -r '.ssh_host' <<<"$INST")"
  SSH_PORT="$(jq -r 'if (.image_runtype // "" | test("jupyter")) then ((.ssh_port | tonumber) + 1) else (.ssh_port | tonumber) end' <<<"$INST")"
fi
echo "  ssh endpoint: root@$SSH_HOST:$SSH_PORT"

echo "waiting for sshd..."
DEADLINE=$(( $(date +%s) + SSH_TIMEOUT ))
until ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$KNOWN_HOSTS" \
          -o LogLevel=ERROR -o ConnectTimeout=10 -i "$SSH_KEY" -p "$SSH_PORT" \
          "root@$SSH_HOST" true 2>/dev/null; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || die "sshd did not come up within $SSH_TIMEOUT s"
  sleep 10
done

trap - EXIT INT TERM

# ---------------------------------------------------------------- done
cat <<EOF

instance $INSTANCE_ID is up. bootstrap.sh is now installing and downloading
the model (~24GB) in the background; expect 8-20 minutes before the dashboard answers.
Nothing listens on port 7860 until that finishes - watch the log first.

  watch setup:   $DEPLOY_DIR/tunnel.sh --logs
  open tunnel:   $DEPLOY_DIR/tunnel.sh        then browse http://localhost:7860
  check state:   $DEPLOY_DIR/provision.sh --status

  DESTROY WHEN DONE (this is the only thing that stops the \$$DPH/hr):
                 $DEPLOY_DIR/destroy.sh
EOF
