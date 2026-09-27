"""Keeps the SSH tunnel to the cloud GPU (Brev) running, so Qwen is reachable
at http://127.0.0.1:8000 without a separate terminal window.

Configured in config.json:
  "ai_tunnel": "wsl -d Ubuntu-22.04 -- bash -lc \"brev port-forward qwen-32b-vlm -p 8000:8000\""
  ("" or "off" disables it.)
Output goes to tunnel.log next to config.json and to the dashboard.
"""

import collections
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_WSL = "Ubuntu-22.04"
DEFAULT_INSTANCE = "qwen-32b-vlm"


def wsl_path(path):
    """C:\\Users\\x\\y -> /mnt/c/Users/x/y"""
    p = str(path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def default_command(script, distro=DEFAULT_WSL):
    return ["wsl", "-d", distro, "--", "bash", wsl_path(script)]


class Tunnel:
    def __init__(self, command, log_path, watch=None):
        if isinstance(command, str):
            command = shlex.split(command, posix=True)
        self.command = list(command)
        self.log_path = Path(log_path)
        self.lines = collections.deque(maxlen=60)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._wake = threading.Event()
        self._proc = None
        self.started_at = None
        self.restarts = 0
        self._thread = None
        self.watch = Path(watch) if watch else None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._done.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if self.watch:
            threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        """Restart the tunnel when its script is edited."""
        def mtime():
            try:
                return self.watch.stat().st_mtime
            except OSError:
                return None
        last = mtime()
        while not self._done.wait(2):
            now = mtime()
            if now != last:
                last = now
                self._log("-- tunnel script changed, restarting --")
                self._kill()
                self._wake.set()

    def restart(self):
        self._log("-- restart requested --")
        self._kill()
        self._wake.set()           # the loop starts it again straight away

    def stop(self):
        self._done.set()
        self._wake.set()
        self._kill()

    def _kill(self):
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":      # kill the wsl.exe tree too
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, timeout=10)
                else:
                    os.killpg(proc.pid, 9)
            except Exception:
                proc.kill()

    def _log(self, text):
        line = time.strftime("%H:%M:%S ") + text.rstrip()
        self.lines.append(line)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _run(self):
        delay = 3
        while not self._done.is_set():
            self._log("starting: " + " ".join(self.command))
            flags = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW
            try:
                proc = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        creationflags=flags,
                                        start_new_session=os.name != "nt")
            except Exception as exc:
                self._log(f"could not start tunnel: {exc}")
                self._done.wait(15)
                continue
            with self._lock:
                self._proc = proc
                self.started_at = time.time()
            for raw in iter(proc.stdout.readline, b""):
                text = raw.replace(b"\x00", b"").decode("utf-8", "replace").strip()
                if text:
                    self._log(text)
            code = proc.wait()
            ran = time.time() - (self.started_at or time.time())
            self._log(f"tunnel exited (code {code}) after {ran:.0f}s")
            with self._lock:
                self._proc = None
            self.restarts += 1
            delay = 3 if ran > 60 else min(30, delay * 2)
            self._wake.wait(delay)
            self._wake.clear()

    def state(self):
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
        age = None if not self.started_at else round(time.time() - self.started_at)
        return {"enabled": True, "running": running, "restarts": self.restarts, "age_s": age,
                "log": list(self.lines)[-15:]}
