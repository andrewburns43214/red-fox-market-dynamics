#!/usr/bin/env python3
"""Authoritative Nginx auth_request verifier for protected Red Fox routes.

The browser supplies only its Supabase access token.  This process never trusts
client-supplied entitlement data: it forwards the token to the database RPC,
where `auth.uid()`, server time, grants, trials, and paid-through periods are
evaluated by `public.has_active_access()`.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request, urlopen
import json
import os
import re


def configured_value(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    # The anon key is intentionally public and already shipped to the browser.
    # Reading it from the deployed public config avoids duplicating credentials
    # in a systemd unit or an unmanaged server-side file.
    with open("/opt/red-fox-market-dynamics/site/config.js", encoding="utf-8") as config:
        source = config.read()
    script_name = "SUPABASE_URL" if name == "SUPABASE_URL" else "SUPABASE_ANON_KEY"
    match = re.search(rf"const {script_name} = '([^']+)'", source)
    if not match:
        raise RuntimeError(f"{name} is not configured")
    return match.group(1)


SUPABASE_URL = configured_value("SUPABASE_URL").rstrip("/")
SUPABASE_ANON_KEY = configured_value("SUPABASE_ANON_KEY")
LISTEN_HOST = os.environ.get("ACCESS_VERIFIER_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("ACCESS_VERIFIER_PORT", "5051"))


def access_token(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    cookies = SimpleCookie()
    try:
        cookies.load(cookie_header)
    except Exception:
        return None
    token = cookies.get("redfox_access_token")
    return unquote(token.value) if token and token.value else None


def has_active_access(token: str) -> bool:
    request = Request(
        f"{SUPABASE_URL}/rest/v1/rpc/has_active_access",
        data=b"{}",
        method="POST",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=4) as response:
            return response.status == 200 and json.loads(response.read()) is True
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


class AccessHandler(BaseHTTPRequestHandler):
    server_version = "RedFoxAccessVerifier/1"

    def log_message(self, format: str, *args: object) -> None:
        # Nginx already records request outcomes; avoid logging bearer material.
        return

    def do_GET(self) -> None:
        token = access_token(self.headers.get("Cookie"))
        if token and has_active_access(token):
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(401)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


if __name__ == "__main__":
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), AccessHandler).serve_forever()
