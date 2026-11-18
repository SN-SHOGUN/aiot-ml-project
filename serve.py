"""
serve.py — Simple HTTP server that serves the webapp + figures.
Run from the project root: python serve.py
"""
import http.server, socketserver, os, pathlib, webbrowser, threading, time

PORT = 8080
ROOT = pathlib.Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)
    def log_message(self, fmt, *args):
        pass  # silence request logs

def open_browser():
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}/webapp/index.html")

print(f"[serve] AIoT ML Web App running at http://localhost:{PORT}/webapp/index.html")
print(f"[serve] Root: {ROOT}")
print("[serve] Press Ctrl+C to stop.\n")

threading.Thread(target=open_browser, daemon=True).start()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
