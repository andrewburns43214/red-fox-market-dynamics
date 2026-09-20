from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")


def test_protected_board_requests_refresh_an_expired_session_once():
    assert "window.redfoxAuthorizedFetch = async function" in BOARD
    assert "client.auth.refreshSession()" in BOARD
    assert "if (!accessRefreshPromise)" in BOARD
    assert "if (response.status !== 401) return response;" in BOARD
    assert "if (session) return fetch(input, init);" in BOARD


def test_failed_session_refresh_uses_existing_auth_recovery_route():
    assert "window.location.replace('/?auth=required');" in BOARD


def test_board_data_loaders_use_authorized_fetch_wrapper():
    assert "window.redfoxAuthorizedFetch('/data/freshness.json" in BOARD
    assert "window.redfoxAuthorizedFetch(PROP_PROJECTION_URL" in BOARD
    assert "window.redfoxAuthorizedFetch(FAVORITE_PERFORMANCE_URL" in BOARD
    assert "window.redfoxAuthorizedFetch(url,{cache:'no-store'})" in BOARD
    assert "window.redfoxAuthorizedFetch(url+'?_='+Date.now())" in BOARD

