import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _windows_networks():
    """[(ip, mask)] of this PC's IPv4 adapters from `ipconfig` (Windows only)."""
    import os
    import re
    import subprocess
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=5,
                             creationflags=0x08000000).stdout
    except Exception:
        return []
    nets, ip = [], None
    for line in out.splitlines():
        m = re.search(r"IPv4[^:]*:\s*([\d.]+)", line)
        if m:
            ip = m.group(1)
            continue
        m = re.search(r"Subnet Mask[^:]*:\s*([\d.]+)", line)
        if m and ip:
            nets.append((ip, m.group(1)))
            ip = None
    return nets


def local_prefixes():
    """'a.b.c.' prefixes worth scanning: every /24 of this PC's real networks (up to /22),
    or the neighbouring /24s on huge networks, plus the rover's own hotspot."""
    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    main = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))          # no packet is sent; picks the main adapter
            main = s.getsockname()[0]
            ips.add(main)
    except OSError:
        pass
    masks = dict(_windows_networks())
    prefixes = []

    def add(ip):
        a, b, c, _ = (int(x) for x in ip.split("."))
        mask = masks.get(ip, "255.255.255.0")
        bits = sum(bin(int(x)).count("1") for x in mask.split("."))
        if bits >= 24:
            third = [c]
        elif bits >= 22:                       # e.g. /22 = 4 x 256 addresses: scan all of it
            size = 2 ** (24 - bits)
            start = c - c % size
            third = list(range(start, start + size))
        else:                                  # very big network: the 3 nearest /24s each side
            third = [c + d for d in (0, -1, 1, -2, 2, -3, 3) if 0 <= c + d <= 255]
        for t in third:
            prefixes.append(f"{a}.{b}.{t}.")

    for ip in ([main] if main else []) + sorted(ips - {main}):
        if ip and not ip.startswith(("127.", "169.254.")):
            add(ip)
    seen, out = set(), []
    for p in prefixes + ["192.168.4."]:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def known_neighbours():
    """IPs Windows has recently talked to (ARP table) - the rover is usually among them."""
    import os
    import re
    import subprocess
    try:
        out = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=5,
                             creationflags=0x08000000 if os.name == "nt" else 0).stdout
    except Exception:
        return []
    ips = re.findall(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", out)
    return [ip for ip in dict.fromkeys(ips)
            if not ip.endswith((".255", ".0")) and not ip.startswith(("224.", "239.", "255.", "127."))]


def _is_rover(ip, opener, timeout):
    try:
        with opener.open(f"http://{ip}/", timeout=timeout) as r:
            body = r.read(400000)
        if b"RSSI" in body and b"VOLTAGE" in body:
            return f"http://{ip}"
    except Exception:
        return None
    return None


def find_rover(prefixes, opener, timeout=0.8, progress=None, neighbours=None):
    """Find the Waveshare rover web page: recently-seen devices first, then full scan."""
    near = neighbours if neighbours is not None else known_neighbours()
    if progress:
        progress(f"checking {len(near)} recently seen devices")
    with ThreadPoolExecutor(max_workers=64) as pool:
        for found in pool.map(lambda ip: _is_rover(ip, opener, timeout), near):
            if found:
                return found
    hosts = [p + str(i) for p in prefixes for i in range(1, 255)]
    if progress:
        nets = ", ".join(p + "x" for p in prefixes[:6]) + (" ..." if len(prefixes) > 6 else "")
        progress(f"scanning {len(hosts)} addresses ({nets})")
    with ThreadPoolExecutor(max_workers=192) as pool:
        for found in pool.map(lambda ip: _is_rover(ip, opener, timeout), hosts):
            if found:
                return found
    return None


class _Link:
    """One kept-open HTTP connection to the rover's ESP32 (no new TCP handshake per command:
    ~10-30 ms per command instead of 50-200 ms). Reconnects by itself if the rover closes it."""

    def __init__(self):
        self._conn = None
        self._host = None

    def get(self, base_url, path, timeout):
        import http.client
        host = base_url.split("//", 1)[-1].split("/", 1)[0]
        for attempt in (1, 2):
            if self._conn is None or self._host != host:
                self.close()
                self._conn = http.client.HTTPConnection(host, timeout=timeout)
                self._host = host
            try:
                self._conn.timeout = timeout
                if self._conn.sock is not None:
                    self._conn.sock.settimeout(timeout)
                self._conn.request("GET", path, headers={"Connection": "keep-alive"})
                resp = self._conn.getresponse()
                body = resp.read(4096)
                if resp.getheader("Connection", "").lower() == "close":
                    self.close()
                return body
            except (OSError, http.client.HTTPException):
                self.close()                  # stale kept-open socket: one fresh try
                if attempt == 2:
                    raise
        return b""

    def close(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None


# This rover's firmware (its own web page sends {"T":1,"L":1800,"R":1800} for full
# forward, scaled by 0.3/0.6/1.0) takes L/R wheel speeds from -1800 to +1800.
FULL_SPEED = 1800


def mix_speeds(linear, turn, maximum=FULL_SPEED):
    linear = max(-1.0, min(1.0, float(linear)))
    turn = max(-1.0, min(1.0, float(turn)))
    left = max(-1.0, min(1.0, linear + turn))
    right = max(-1.0, min(1.0, linear - turn))
    return round(left * maximum, 3), round(right * maximum, 3)


class RoverDriver:
    """Duration-bounded rover commands with a local refresh watchdog."""

    def __init__(self, base_url, max_speed, safety, opener=None, discover_prefixes=None, on_found=None):
        # "auto" (or empty) = find the rover on the network by itself.  A fixed
        # address is tried first; if it stops answering, the hub searches.
        self.base_url = "" if base_url in ("", "auto") else base_url.rstrip("/")
        self._prefixes = discover_prefixes
        self._searching = False
        self._last_search = 0.0
        self._fail_since = None
        self._last_io = 0.0
        # rover_max_speed: fraction of full speed (0-1), or an absolute value up to 1800.
        value = float(max_speed)
        self.max_speed = int(FULL_SPEED * max(0.0, value)) if value <= 1 else int(min(value, FULL_SPEED))
        self.accel = max(self.max_speed, 1) / 0.6    # 0 -> top speed in 0.6 s
        self.brake = max(self.max_speed, 1) / 0.3    # top speed -> 0 in 0.3 s
        self._current = (0, 0)
        self._hard_stop = False
        self.safety = safety
        self._custom_opener = opener is not None
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._link = None if opener is not None else _Link()
        self.on_found = on_found               # called with the new address when the search finds it
        self._fails = 0
        self._latency = None                    # ms, smoothed round trip of a drive command
        self._telemetry = {}                    # battery voltage, Wi-Fi signal, rover IP
        self._last_telemetry = 0.0
        self._last_info = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._done = threading.Event()
        self._desired = (0, 0)
        self._deadline = 0.0
        self._last_sent = None
        self._connected = False
        self._error = "Not contacted"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _url(self, left, right):
        command = json.dumps({"T": 1, "L": left, "R": right}, separators=(",", ":"))
        return f"{self.base_url}/js?{urllib.parse.urlencode({'json': command})}"

    def command(self, linear, turn, duration_ms):
        duration_ms = int(duration_ms)
        if not 50 <= duration_ms <= 500:
            raise ValueError("Rover duration must be between 50 and 500 ms")
        speeds = mix_speeds(linear, turn, self.max_speed)
        with self._lock:
            self._desired = speeds
            self._deadline = time.monotonic() + duration_ms / 1000
        self._wake.set()
        return speeds

    def stop(self):
        """Emergency stop: zero immediately, no ramp."""
        with self._lock:
            self._desired = (0, 0)
            self._deadline = 0
            self._hard_stop = True
        self._wake.set()

    def set_url(self, url):
        url = str(url).strip().rstrip("/")
        if url and not url.startswith("http"):
            url = "http://" + url
        if url and not url.split("//", 1)[1]:
            raise ValueError("Give the rover address, e.g. 10.86.102.15")
        self.base_url = url
        self._fail_since = None
        self._last_io = 0.0
        with self._lock:
            self._error = "Connecting..."
        self._wake.set()
        return url

    def _start_search(self):
        if self._searching or time.monotonic() - self._last_search < 10:
            return
        self._searching = True
        self._last_search = time.monotonic()
        with self._lock:
            self._error = "Searching the network for the rover..."
        threading.Thread(target=self._search, daemon=True).start()

    def _search(self):
        def progress(msg):
            with self._lock:
                self._error = "Searching for the rover: " + msg + "..."
        try:
            found = find_rover(self._prefixes or local_prefixes(), self.opener, progress=progress,
                               neighbours=[] if self._prefixes else None)
            if found and found != self.base_url:
                self.base_url = found
                self._fail_since = None
                self.safety.record("rover", "found", f"Rover found at {found}")
                if self.on_found:
                    try:
                        self.on_found(found)           # remember it for the next start
                    except Exception:
                        pass
            elif not found:
                with self._lock:
                    self._error = ("Rover not found on this laptop's network - is it switched on and on the "
                                   "same Wi-Fi? Or type its IP (shown on its screen) in the box")
        finally:
            self._searching = False

    def _get(self, obj, timeout=0.6):
        """Send one JSON command to the rover -> reply bytes."""
        if not self.base_url:
            raise OSError("Rover address unknown - searching")
        cmd = json.dumps(obj, separators=(",", ":"))
        if self._link is not None:
            return self._link.get(self.base_url, "/js?" + urllib.parse.urlencode({"json": cmd}), timeout)
        url = f"{self.base_url}/js?{urllib.parse.urlencode({'json': cmd})}"
        with self.opener.open(url, timeout=max(timeout, 2.0)) as response:
            return response.read(64)

    def _send(self, command):
        t0 = time.monotonic()
        self._get({"T": 1, "L": command[0], "R": command[1]})
        ms = (time.monotonic() - t0) * 1000
        self._latency = ms if self._latency is None else 0.7 * self._latency + 0.3 * ms

    def _poll_telemetry(self):
        """Idle: battery voltage (T:130) every 2 s, Wi-Fi signal / IP (T:405) every 10 s."""
        now = time.monotonic()
        if self._custom_opener or now - self._last_telemetry < 2.0:
            return
        self._last_telemetry = now
        try:
            fb = json.loads(self._get({"T": 130}, timeout=0.5) or b"{}")
            if isinstance(fb, dict) and "v" in fb:
                v = float(fb["v"])
                self._telemetry["battery_v"] = round(v / 100 if v > 100 else v, 2)
            if now - self._last_info > 10:
                self._last_info = now
                info = json.loads(self._get({"T": 405}, timeout=0.5) or b"{}")
                if isinstance(info, dict):
                    self._telemetry.update({k: info[k] for k in ("rssi", "ip", "mac") if k in info})
            self._last_io = now
            with self._lock:
                self._connected, self._error = True, ""
            self._fails = 0
        except Exception:
            pass

    def _ramp(self, want, dt):
        """Smooth speed changes so the rover (and anything on it) does not tip."""
        out = []
        for cur, w in zip(self._current, want):
            braking = abs(w) < abs(cur) or cur * w < 0
            step = (self.brake if braking else self.accel) * dt
            out.append(int(round(cur + max(-step, min(step, w - cur)))))
        self._current = tuple(out)
        return self._current

    def _run(self):
        last = time.monotonic()
        while not self._done.is_set():
            now = time.monotonic()
            dt, last = min(now - last, 0.3), now
            with self._lock:
                expired = now >= self._deadline
                want = (0, 0) if expired else self._desired
                if self._hard_stop:
                    self._current, self._hard_stop = (0, 0), False
            command = self._ramp(want, dt)
            if command == (0, 0):
                command = (0, 0)
            heartbeat = time.monotonic() - self._last_io > 2.0
            if command == (0, 0) and self._last_sent == (0, 0) and not heartbeat:
                self._poll_telemetry()                   # idle: read battery / signal instead
            elif command != self._last_sent or command != (0, 0) or heartbeat:
                self._last_io = time.monotonic()
                try:
                    self._send(command)
                    with self._lock:
                        self._connected = True
                        self._error = ""
                    self._fail_since = None
                    self._fails = 0
                    self._last_sent = command
                except Exception as exc:
                    self._fails += 1
                    self._last_sent = None
                    now = time.monotonic()
                    self._fail_since = self._fail_since or now
                    # one lost Wi-Fi packet: just send again on the next tick (the rover keeps its
                    # last command ~3 s by itself). Only a real outage counts as disconnected.
                    if self._fails >= 3 or now - self._fail_since > 0.8:
                        was = self._connected
                        with self._lock:
                            self._connected = False
                            self._error = str(exc) or type(exc).__name__
                            self._desired = (0, 0)       # don't resume driving on its own later
                            self._deadline = 0
                        if was:
                            self.safety.record("rover", "warning", "Rover link lost - stopped driving")
                        if now - self._fail_since > (4 if self.base_url else 0):
                            self._start_search()
                        self._wake.wait(.3)               # no need to hammer an absent rover
            self._wake.wait(.1)
            self._wake.clear()

    def state(self):
        with self._lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "last_sent": self._last_sent,
                "url": self.base_url,
                "max_speed": self.max_speed,
                "latency_ms": None if self._latency is None else round(self._latency),
                **self._telemetry,
            }

    def close(self):
        self.stop()
        deadline = time.monotonic() + .8
        while self._last_sent != (0, 0) and time.monotonic() < deadline:
            time.sleep(.03)
        self._done.set()
        self._wake.set()
        self._thread.join(timeout=1)
