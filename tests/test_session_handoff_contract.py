from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")
NGINX = (ROOT / "redfox-nginx-site.conf").read_text(encoding="utf-8")


def _function_body(source, name):
    start = source.index(f"function {name}")
    end = source.find("\n}\n", start) + 2
    return source[start:end]


def test_valid_persisted_session_repairs_cookie_before_returning_to_board():
    recovery = _function_body(INDEX, "recoverBoardAfterAuthRequired")
    assert "syncAccessCookie(session);" in recovery
    assert "await hasBoardAccess(session)" in recovery
    assert "window.location.replace(BOARD_RECOVERY_DESTINATION);" in recovery
    assert recovery.index("syncAccessCookie(session);") < recovery.index("window.location.replace")


def test_direct_board_visit_and_reload_use_the_guarded_recovery_destination():
    assert "error_page 401 = /index.html?auth=required;" in NGINX
    assert "const BOARD_RECOVERY_DESTINATION = '/board.html';" in INDEX
    assert "const BOARD_RECOVERY_GUARD_TTL_MS = 10000;" in INDEX
    assert "if (hasRecentBoardRecoveryAttempt()) return;" in INDEX
    assert "sessionStorage.removeItem('redfox_board_recovery_attempt')" in BOARD


def test_browser_restart_can_use_the_persisted_supabase_session_to_recover():
    on_load = INDEX[INDEX.index("// ── On load: check existing session ──"):]
    assert "await _supabase.auth.getSession()" in on_load
    assert "await recoverBoardAfterAuthRequired(session);" in on_load
    assert "persistSession: false" not in INDEX
    assert "autoRefreshToken: false" not in INDEX


def test_invalid_session_stays_on_sign_in_instead_of_retrying_board():
    recovery = _function_body(INDEX, "recoverBoardAfterAuthRequired")
    assert "if (!session)" in recovery
    assert "openModal();" in recovery
    assert recovery.index("openModal();") < recovery.index("syncAccessCookie(session);")


def test_explicit_sign_out_still_clears_auth_state_normally():
    sign_out = _function_body(INDEX, "handleSignOut")
    assert "await _supabase.auth.signOut();" in sign_out
    assert "syncAccessCookie(null);" in sign_out
    assert "window.location.href = '/';" in sign_out


def test_server_side_board_protection_remains_in_place():
    board_location = NGINX[NGINX.index("location = /board.html {"):]
    assert "auth_request /_internal/board-access;" in board_location
    assert "location ~ ^/data/" in NGINX
    assert NGINX.count("auth_request /_internal/board-access;") >= 2
