import http.server, socketserver, os

PORT = 5050
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/site/index.html'
        elif self.path == '/board.html':
            self.path = '/site/board.html'
        elif self.path.startswith('/config.js'):
            self.path = '/site/config.js'
        elif self.path.startswith('/auth.js'):
            self.path = '/site/auth.js'
        return super().do_GET()
    def log_message(self, format, *args):
        pass

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Red Fox running at http://localhost:5050")
    httpd.serve_forever()
