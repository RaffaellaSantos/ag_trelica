"""
serve.py — servidor HTTP local para trelica_3d.html
Executado por run_viewer.bat (ou: python serve.py)
"""
import http.server, webbrowser, threading, os, sys

PORT = 8080
ROOT = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)
    def log_message(self, fmt, *args):   # silencia logs de request
        pass

def open_browser():
    webbrowser.open(f"http://localhost:{PORT}/trelica_3d.html")

print(f"Servidor rodando em http://localhost:{PORT}/trelica_3d.html")
print("Pressione Ctrl+C para parar.\n")
threading.Timer(1.0, open_browser).start()

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
