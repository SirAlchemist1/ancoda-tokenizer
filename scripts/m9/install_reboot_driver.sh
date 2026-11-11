#!/bin/bash
# Install a user crontab @reboot entry for the full-run driver. No sudo.
set -euo pipefail
DRIVER="$HOME/m9/run_full.sh"
MARKER="# m9-full-driver"
# Marker must be a trailing comment so it does not comment out screen.
LINE="@reboot sleep 60; screen -dmS m9full bash -lc 'bash $DRIVER >> $HOME/m9/queue.log 2>&1' $MARKER"
existing=$(crontab -l 2>/dev/null || true)
filtered=$(printf '%s\n' "$existing" | grep -v 'm9-full-driver' || true)
printf '%s\n%s\n' "$filtered" "$LINE" | crontab -
echo "installed crontab:"
crontab -l | grep m9-full-driver
