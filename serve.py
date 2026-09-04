import http.server
import os
import urllib.error
import urllib.parse
import urllib.request


PORT = 5050
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path.startswith('/sandbox-data/'):
            self._serve_sandbox_data(path)
            return
        if path == '/board-sandbox.html':
            # Sandbox adapts only board data URLs. The tracked production board
            # owns the entire visual shell, including the sidebar.
            with open(os.path.join(ROOT, 'site', 'board.html'), encoding='utf-8') as source:
                body = source.read().replace("'/data/", "'/sandbox-data/").encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path in ('/', '/index.html'):
            self.path = '/site/board.html'
        elif path == '/landing.html':
            self.path = '/site/index.html'
        elif path == '/board.html':
            self.path = '/site/board.html'
        elif path.startswith('/config.js'):
            self.path = '/site/config.js'
        elif path.startswith('/auth.js'):
            self.path = '/site/auth.js'
        elif path.startswith('/preview.png'):
            self.path = '/site/preview.png'
        elif path.startswith('/logo.png'):
            self.path = '/site/logo.png'
        return super().do_GET()

    def _serve_sandbox_data(self, path):
        allowed = {
            'anomaly_board.csv', 'anomaly_events.csv', 'results_resolved.csv',
            'kpi_master_dataset.csv', 'freshness.json', 'book_lines.json', 'live_recent.csv',
        }
        filename = urllib.parse.unquote(path.removeprefix('/sandbox-data/'))
        is_detail = (
            filename.startswith('anomaly_event_details/')
            and filename.endswith('.json')
            and '/' not in filename[len('anomaly_event_details/'):]
            and '\\' not in filename
        )
        if filename not in allowed and not is_detail:
            self.send_error(404)
            return
        try:
            staging_root = os.path.join(ROOT, 'data', 'two_side_staging')
            use_staging = filename in {'anomaly_board.csv', 'anomaly_events.csv'} or is_detail
            local_path = os.path.join(staging_root, *filename.split('/')) if use_staging else ''
            if local_path and os.path.isfile(local_path):
                with open(local_path, 'rb') as source:
                    body = source.read()
            else:
                request = urllib.request.Request(
                    'https://redfoxmi.com/data/' + urllib.parse.quote(filename, safe='/'),
                    headers={'User-Agent': 'RedFoxSandbox/1.0'},
                )
                with urllib.request.urlopen(request, timeout=8) as upstream:
                    body = upstream.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8' if filename.endswith('.json') else 'text/csv; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except urllib.error.HTTPError as error:
            self.send_error(error.code, 'Requested board data is unavailable')
        except Exception as error:
            self.send_error(502, 'Unable to load current board data: ' + str(error))

    def log_message(self, format, *args):
        pass


with http.server.ThreadingHTTPServer(('', PORT), Handler) as httpd:
    print(f'Red Fox running at http://localhost:{PORT}')
    httpd.serve_forever()
