"""Server-side subscription check used by Nginx before serving paid board assets."""

from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUPABASE_URL = "https://nwvosippnquwhtuppmkw.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_Nr0RHkARHAR7Obu-N4O3yQ_WU3OrNjU"
ACCESS_COOKIE = "redfox_access_token"
RPC_URL = f"{SUPABASE_URL}/rest/v1/rpc/has_active_access"


def has_paid_access(cookie_header: str | None) -> bool:
    """Ask Supabase to evaluate access under the user's own authenticated JWT."""
    if not cookie_header:
        return False

    cookies = SimpleCookie()
    try:
        cookies.load(cookie_header)
    except (CookieError, ValueError):
        return False

    token_cookie = cookies.get(ACCESS_COOKIE)
    if not token_cookie or not token_cookie.value:
        return False

    request = Request(
        RPC_URL,
        data=b"{}",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {token_cookie.value}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status == HTTPStatus.OK and response.read().strip() == b"true"
    except (HTTPError, URLError, TimeoutError):
        return False


class AccessHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/verify":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        status = HTTPStatus.NO_CONTENT if has_paid_access(self.headers.get("Cookie")) else HTTPStatus.UNAUTHORIZED
        self.send_response(status)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Nginx performs this check for every paid asset; keep system logs quiet.
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 5051), AccessHandler).serve_forever()
