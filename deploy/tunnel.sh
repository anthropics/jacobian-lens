#!/usr/bin/env bash
# Open the SSH tunnel from this laptop to the J-lens dashboard on the vast.ai instance.
#   ./tunnel.sh               forward localhost:7860 -> instance 127.0.0.1:7860
#   ./tunnel.sh --port 7861   use a different local port (remote stays 7860)
#   ./tunnel.sh --logs        tail the remote setup/app logs instead of tunnelling
#   ./tunnel.sh <instance-id> target an instance other than the one in .instance
set -euo pipefail

API="https://console.vast.ai/api/v0"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$DEPLOY_DIR/.instance"
# Dedicated known_hosts: vast recycles ip:port pairs, so a changed key is routine churn
# here and would otherwise poison the user's real ~/.ssh/known_hosts. Keeping a file
# still catches a key swap within one instance's life. If churn trips it, delete it.
KNOWN_HOSTS="$DEPLOY_DIR/.known_hosts"
SSH_KEY="${JLENS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_PORT=7860
LOCAL_PORT=7860
MODE=tunnel
INSTANCE_ID=""

die() { echo "FATAL: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) LOCAL_PORT="${2:-}"
            [[ "$LOCAL_PORT" =~ ^[0-9]+$ ]] || die "--port needs a number"; shift 2 ;;
    --logs) MODE=logs; shift ;;
    -h|--help) sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *) INSTANCE_ID="$1"; shift ;;
  esac
done

[ -f "$SSH_KEY" ] || die "ssh key not found at $SSH_KEY (set JLENS_SSH_KEY)"
command -v jq >/dev/null || die "jq not installed"

if [ -z "$INSTANCE_ID" ]; then
  [ -f "$STATE_FILE" ] || die "no instance id: $STATE_FILE is missing. Run provision.sh, or pass an id."
  INSTANCE_ID="$(tr -d '[:space:]' < "$STATE_FILE")"
  [ -n "$INSTANCE_ID" ] || die "$STATE_FILE is empty. Run '$DEPLOY_DIR/destroy.sh --list' to find the id."
fi

KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
KEY="${KEY//[$'\n\r']/}"
[ -n "$KEY" ] || die "no vast.ai API key"

RESP="$(curl -sS -w '\n%{http_code}' -H "Authorization: Bearer $KEY" \
        "$API/instances/$INSTANCE_ID/?owner=me")" || die "could not reach the vast.ai API"
CODE="${RESP##*$'\n'}"
BODY="${RESP%$'\n'*}"
# An auth error / 5xx also returns a body with a null .instances, so check the status
# first - otherwise a rotated key reads as "your instance is gone" while it still bills.
[ "$CODE" = 200 ] || die "vast API error (HTTP $CODE): $BODY"
INST="$(jq '.instances' <<<"$BODY")"
# A missing instance comes back as HTTP 200 with "instances": null, not a 404.
[ "$INST" != null ] && [ -n "$INST" ] || die "instance $INSTANCE_ID does not exist (already destroyed?)"

STATUS="$(jq -r '.actual_status // "unknown"' <<<"$INST")"
[ "$STATUS" = running ] || die "instance $INSTANCE_ID is '$STATUS', not running. Check: $DEPLOY_DIR/provision.sh --status"

# Direct SSH if the port map is published, otherwise the vast proxy.
if jq -e '.ports["22/tcp"][0].HostPort' >/dev/null <<<"$INST"; then
  SSH_HOST="$(jq -r '.public_ipaddr' <<<"$INST")"
  SSH_PORT="$(jq -r '.ports["22/tcp"][0].HostPort' <<<"$INST")"
else
  SSH_HOST="$(jq -r '.ssh_host' <<<"$INST")"
  SSH_PORT="$(jq -r 'if (.image_runtype // "" | test("jupyter")) then (.ssh_port + 1) else .ssh_port end' <<<"$INST")"
fi
[ -n "$SSH_HOST" ] && [ "$SSH_HOST" != null ] && [[ "$SSH_PORT" =~ ^[0-9]+$ ]] \
  || die "instance $INSTANCE_ID exposes no ssh endpoint yet"

SSH_OPTS=(-i "$SSH_KEY"
  -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15 -o LogLevel=ERROR)

if [ "$MODE" = logs ]; then
  # bootstrap.sh tees install and download progress to jlens-bootstrap.log, which is the
  # only log with anything in it during the 8-20 min setup window. Tail all three.
  echo "tailing /var/log/portal/jlens-{setup,bootstrap,app}.log on $INSTANCE_ID - Ctrl-C to stop"
  exec ssh -t "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST" \
    'tail -n 50 -F /var/log/portal/jlens-setup.log /var/log/portal/jlens-bootstrap.log /var/log/portal/jlens-app.log'
fi

# Fail with a sentence rather than an ssh backtrace when something already holds the port.
if (exec 3<>"/dev/tcp/127.0.0.1/$LOCAL_PORT") 2>/dev/null; then
  die "local port $LOCAL_PORT is already in use (another tunnel?). Re-run with --port 7861."
fi

# "running" only means the container is up. bootstrap.sh then spends 8-20 min installing
# and pulling ~24GB, and -L binds locally whether or not anything listens on the far end,
# so probe the remote port before promising the user a working URL.
if ! ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST" \
     "timeout 2 bash -c '</dev/tcp/127.0.0.1/$REMOTE_PORT'" 2>/dev/null; then
  echo "the dashboard is not listening yet - bootstrap is probably still installing and"
  echo "downloading (8-20 min from launch). Watch it with: $0 --logs"
  echo "Opening the tunnel anyway; reload the page once the log prints 'server started'."
  echo
fi

cat <<EOF
dashboard: http://localhost:$LOCAL_PORT
  via root@$SSH_HOST:$SSH_PORT -> 127.0.0.1:$REMOTE_PORT on instance $INSTANCE_ID
  Ctrl-C closes the tunnel. It does NOT stop the instance billing -
  that is $DEPLOY_DIR/destroy.sh

EOF

set +e
ssh -N "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes \
  -p "$SSH_PORT" -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "root@$SSH_HOST"
RC=$?
set -e
if [ "$RC" = 255 ]; then
  echo "ssh exited with an error (see its message above). If it was a changed host key," >&2
  echo "delete $KNOWN_HOSTS - vast recycles ip:port pairs across machines." >&2
fi
exit "$RC"
