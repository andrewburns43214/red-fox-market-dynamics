import pandas as pd

import main


def test_final_score_history_is_append_only_and_keeps_provenance(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([{
        "game_id": "old-game", "team1": "old away", "team1_score": "3",
        "team2": "old home", "team2_score": "2",
        "resolved_at_utc": "2026-09-01T00:00:00Z",
    }]).to_csv(data / "final_scores_history.csv", index=False)
    pd.DataFrame([{"game_id": "new-game"}]).to_csv(data / "decision_freeze_ledger.csv", index=False)
    pd.DataFrame([
        {
            "game_id": "new-game", "game": "Away Team @ Home Team", "side": "Away Team",
            "final_score_for": "20", "final_score_against": "17",
        },
        {
            "game_id": "new-game", "game": "Away Team @ Home Team", "side": "Home Team",
            "final_score_for": "17", "final_score_against": "20",
        },
    ]).to_csv(data / "snapshots.csv", index=False)
    monkeypatch.chdir(tmp_path)

    main.update_final_scores_history()
    history = pd.read_csv(data / "final_scores_history.csv", dtype=str, keep_default_na=False)

    assert set(history["game_id"]) == {"old-game", "new-game"}
    assert history["score_evidence_hash"].str.len().eq(64).all()
    assert history["score_source"].ne("").all()

    snapshots = pd.read_csv(data / "snapshots.csv", dtype=str)
    snapshots.loc[snapshots["side"].eq("Away Team"), "final_score_for"] = "99"
    snapshots.to_csv(data / "snapshots.csv", index=False)
    main.update_final_scores_history()
    unchanged = pd.read_csv(data / "final_scores_history.csv", dtype=str, keep_default_na=False)

    new_game = unchanged[unchanged["game_id"].eq("new-game")].iloc[0]
    assert new_game["team1_score"] == "20.0"
    assert len(unchanged) == 2
