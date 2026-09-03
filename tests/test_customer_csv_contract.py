from pathlib import Path
import re


def test_customer_csv_export_uses_the_two_sided_board_contract():
    """The public download must reflect the paired board, not legacy action fields."""
    source = Path("site/board.html").read_text(encoding="utf-8")
    match = re.search(
        r"function exportCSV\(\)\{(?P<body>.*?)\n\}\n\nfunction customerExportSides",
        source,
        flags=re.DOTALL,
    )
    assert match, "Customer CSV exporter was not found"
    export = match.group("body")

    for header in (
        "'Board Rank'", "'Game'", "'Sport'", "'Kickoff (ET)'", "'Market'",
        "'Side A'", "'Side A Bets'", "'Side A Money'", "'Side A Open'", "'Side A Current'", "'Side A Market Read'",
        "'Side B'", "'Side B Bets'", "'Side B Money'", "'Side B Open'", "'Side B Current'", "'Side B Market Read'",
        "'Market Rationale'", "'Data Quality'", "'Snapshot Time (ET)'",
    ):
        assert header in export

    for internal_field in ("action_type", "action_basis", "kpi_eligible", "observed_signal_side"):
        assert internal_field not in export.lower()

    assert "customerExportSides(r)" in export
    assert "hasCustomerSideFields(side)" in export
    assert "_boardFreshness?.dk_ts" in export
    assert "exported.forEach((row,index)=>{ row[0]=index+1; });" in export


def test_customer_csv_keeps_positive_moneyline_signs_and_requires_two_sides():
    source = Path("site/board.html").read_text(encoding="utf-8")
    assert "market==='MONEYLINE'" in source
    assert "return '=\"'+(line.startsWith('+')?line:'+'+line)+'\"';" in source
    assert "Array.isArray(sides)&&sides.length===2" in source
    assert "function customerKickoff(row)" in source
    assert "function customerSnapshotTime(raw)" in source
