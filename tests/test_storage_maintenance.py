from datetime import datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ops" / "redfox-storage-maintenance.py"
SPEC = importlib.util.spec_from_file_location("redfox_storage_maintenance", MODULE_PATH)
maintenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(maintenance)


def test_removes_only_stale_favorite_shadow_atomic_temps(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    authoritative = data / "red_fox_favorite_shadow.csv"
    authoritative.write_text("authoritative", encoding="utf-8")
    stale = data / ".red_fox_favorite_shadow.csv.abandoned.tmp"
    stale.write_bytes(b"x" * 17)
    fresh = data / ".red_fox_favorite_shadow.csv.active.tmp"
    fresh.write_bytes(b"y" * 19)
    unrelated = data / ".performance_favorites.csv.abandoned.tmp"
    unrelated.write_bytes(b"z" * 23)

    now = datetime(2026, 11, 23, tzinfo=timezone.utc)
    old = (now - timedelta(hours=1)).timestamp()
    os.utime(stale, (old, old))

    result = maintenance.remove_stale_atomic_temps(data, now=now)

    assert result == {"files": 1, "bytes": 17}
    assert authoritative.read_text(encoding="utf-8") == "authoritative"
    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()
