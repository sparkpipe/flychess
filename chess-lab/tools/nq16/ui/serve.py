#!/usr/bin/env python3
"""Live games server :8077 — serves index.html + games.json from ~/games-live."""
import http.server, socketserver, os, threading, subprocess, time

HOME = os.path.expanduser("~/games-live")
PORT = 8077
BOX = "spec@192.168.50.4"
REMOTE = "/extnvme/active/matches/games.json"

def poll_forever():
    while True:
        try:
            subprocess.run(
                ["rsync", "-q", "-e", "ssh -o ConnectTimeout=10 -o BatchMode=yes",
                 f"{BOX}:{REMOTE}", f"{HOME}/games.json"],
                timeout=30)
            if os.path.exists(f"{HOME}/focus.json"):
                subprocess.run(
                    ["rsync", "-q", "-e", "ssh -o ConnectTimeout=10 -o BatchMode=yes",
                     f"{HOME}/focus.json", f"{BOX}:/extnvme/active/matches/focus.json"],
                    timeout=30)
        except Exception:
            pass
        time.sleep(12)

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HOME, **kw)
    def log_message(self, *a):
        pass
    def do_POST(self):
        if self.path == "/focus":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = self.rfile.read(n)
                open(f"{HOME}/focus.json", "wb").write(body)
                self.send_response(204)
            except Exception:
                self.send_response(500)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    os.makedirs(HOME, exist_ok=True)
    threading.Thread(target=poll_forever, daemon=True).start()
    with S(("127.0.0.1", PORT), H) as s:
        print(f"serving {HOME} on http://localhost:{PORT}/")
        s.serve_forever()
