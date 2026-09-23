#!/usr/bin/env bash
# Keep trying to provision until vast.ai has a matching offer, then wait for the
# bootstrap to finish and open the tunnel. For when provision.sh says "0 survived".
#   ./wait.sh                  poll every 2 min, provision on the first match, tunnel
#   ./wait.sh --interval 300   poll every 5 min
#   ./wait.sh --no-tunnel      stop once the dashboard is up instead of tunnelling
# Everything after provisioning is the normal flow; Ctrl-C at any point after the
# instance exists leaves it running and billing - destroy.sh stops it.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KNOWN_HOSTS="$DEPLOY_DIR/.known_hosts"
SSH_KEY="${JLENS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL=120
TUNNEL=1

die() { echo "FATAL: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="${2:-}"; [[ "$INTERVAL" =~ ^[0-9]+$ ]] || die "--interval needs seconds"; shift 2 ;;
    --no-tunnel) TUNNEL=0; shift ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -f "$DEPLOY_DIR/.instance" ] && die "$DEPLOY_DIR/.instance exists: an instance may already be up. destroy.sh or provision.sh --status first."

# --- poll until provision.sh gets past the offer search ---------------------------
T0=$(date +%s)
N=0
while :; do
  N=$((N + 1))
  echo "[$(date +%H:%M:%S)] attempt $N"
  # pipefail is inherited by the substitution, so $? is provision.sh's status.
  set +e
  OUT="$("$DEPLOY_DIR/provision.sh" --yes 2>&1 | tee /dev/stderr)"
  RC=$?
  set -e
  [ "$RC" = 0 ] && break
  # provision.sh dies before spending when the search comes back empty. Anything
  # else (credit floor, API error, create failure after vast accepted) is not
  # something to retry blindly: an instance may exist.
  if [ -f "$DEPLOY_DIR/.instance" ]; then
    die "provision.sh failed after creating an instance (see above). Not retrying. destroy.sh stops it."
  fi
  if grep -q "0 survived\|vastai search failed" <<<"$OUT"; then
    echo "no usable offer yet; waited $(( ($(date +%s) - T0) / 60 )) min so far, next try in ${INTERVAL}s"
    sleep "$INTERVAL"
    continue
  fi
  die "provision.sh failed for a reason other than 'no offers' (see above). Not retrying."
done

# --- instance exists; wait for bootstrap.sh to touch its READY marker ------------
ENDPOINT="$(grep -o 'ssh endpoint: root@[^ ]*' <<<"$OUT" | tail -1 | sed 's/ssh endpoint: root@//')"
SSH_HOST="${ENDPOINT%:*}"
SSH_PORT="${ENDPOINT##*:}"
[ -n "$SSH_HOST" ] && [[ "$SSH_PORT" =~ ^[0-9]+$ ]] || die "could not read the ssh endpoint from provision.sh output"
SSH_OPTS=(-i "$SSH_KEY" -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new
  -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR)

echo
echo "instance is up after $(( ($(date +%s) - T0) / 60 )) min of polling. Waiting for bootstrap (8-20 min)..."
while :; do
  STATE="$(ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST" \
    'if [ -f /workspace/.jlens/READY ]; then echo ready; elif [ -f /workspace/.jlens/FAILED ]; then echo failed; cat /workspace/.jlens/FAILED; else tail -n 1 /var/log/portal/jlens-bootstrap.log 2>/dev/null || echo starting; fi' 2>/dev/null \
    || echo "ssh not answering yet")"
  case "$STATE" in
    ready) break ;;
    failed*) die "bootstrap failed on the instance: ${STATE#failed}. See: $DEPLOY_DIR/tunnel.sh --logs. The instance is still billing." ;;
  esac
  echo "  [$(date +%H:%M:%S)] $STATE"
  sleep 30
done

printf '\a'
echo
echo "dashboard is up on the instance. The model still loads for 4-8 min; the page shows it."
[ "$TUNNEL" = 1 ] || { echo "open it with: $DEPLOY_DIR/tunnel.sh"; exit 0; }
exec "$DEPLOY_DIR/tunnel.sh"
