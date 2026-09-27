#!/bin/bash
# Run inside WSL by the robot hub (robot_hub/tunnel.py). Edit freely: the hub
# restarts the tunnel automatically when this file changes.
# non-login shell: load the PATH that has brev on it
[ -f ~/.profile ] && . ~/.profile >/dev/null 2>&1
export PATH="$PATH:/usr/local/bin:$HOME/.local/bin:$HOME/bin:$HOME/.brev/bin"
PORT=8000
# Preferred GPU first; falls back to the next one if its model server is not answering yet.
CANDIDATES="qwen-72b-vlm qwen-32b-vlm"
echo "== brev refresh =="
timeout 40 brev refresh >/dev/null 2>&1 && echo "ok" || echo "brev refresh failed/timed out"
INST=""
for c in $CANDIDATES; do
  echo "== checking $c =="
  ans=$(timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new $c \
        "curl -s -m 8 localhost:$PORT/v1/models" 2>/dev/null | head -c 400)
  if echo "$ans" | grep -q '"id"'; then
    echo "$c serves: $(echo "$ans" | grep -o '"id":"[^"]*"' | head -1)"
    INST=$c; break
  else
    echo "$c: no model answering on port $PORT yet (${ans:0:120})"
  fi
done
[ -z "$INST" ] && { echo "no GPU is serving a model - retrying soon"; sleep 20; exit 1; }
echo "== using $INST =="
echo "== freeing local port $PORT =="
ssh -O exit $INST 2>/dev/null
for pid in $(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
  echo "killing old listener: $(ps -o pid=,args= -p $pid | cut -c1-120)"; kill $pid 2>/dev/null
done
sleep 1
ss -ltnH "sport = :$PORT" | grep -q . && echo "port $PORT STILL busy" || echo "port $PORT free"
echo "== connecting =="
ssh -N -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new -o LogLevel=INFO -o ControlMaster=no -o ControlPath=none \
  -L 127.0.0.1:$PORT:localhost:$PORT $INST 2>&1 </dev/null
echo "ssh exit $?"
