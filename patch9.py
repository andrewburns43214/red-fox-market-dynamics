with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# --- CHANGE 1: market_read scoring block ---
# Add Reverse Pressure +8, Neutral -2, fix Contradiction +2 -> 0
old1 = """        mr = str(row.get("market_read") or "").strip()
        if mr == "Stealth Move":
            score += 8
        elif mr == "Freeze Pressure":
            score += 10
        elif mr == "Aligned Sharp":
            score += 6
        elif mr == "Contradiction":
            score += 2
        elif mr == "Public Drift":
            score -= 10"""

new1 = """        mr = str(row.get("market_read") or "").strip()
        if mr == "Stealth Move":
            score += 8
        elif mr == "Freeze Pressure":
            score += 10
        elif mr == "Aligned Sharp":
            score += 6
        elif mr == "Reverse Pressure":
            score += 8
        elif mr == "Contradiction":
            score += 0
        elif mr == "Neutral":
            score -= 2
        elif mr == "Public Drift":
            score -= 10"""

if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1
    print("CHANGE 1 OK: market_read scoring updated")
else:
    print("CHANGE 1 FAILED: anchor not found")

# --- CHANGE 2: divergence cap - market-specific ---
old2 = """        try:
            D = float(row.get("divergence_D")) if pd.notna(row.get("divergence_D")) else 0.0
        except Exception:
            D = 0.0
        score += min(12.0, abs(D) * 0.4)"""

new2 = """        try:
            D = float(row.get("divergence_D")) if pd.notna(row.get("divergence_D")) else 0.0
        except Exception:
            D = 0.0
        # v1.1: TOTAL divergence weighted lower (naturally extreme splits)
        if str(mkt).strip().upper() == "TOTAL":
            score += min(10.0, abs(D) * 0.3)
        else:
            score += min(12.0, abs(D) * 0.4)"""

if old2 in content:
    content = content.replace(old2, new2, 1)
    changes += 1
    print("CHANGE 2 OK: divergence cap market-specific")
else:
    print("CHANGE 2 FAILED: anchor not found")

# --- CHANGE 3: line move cap - market-specific ---
old3 = """        try:
            lm = float(row.get("line_move_open")) if pd.notna(row.get("line_move_open")) else 0.0
        except Exception:
            lm = 0.0
        score += min(8.0, abs(lm) * 2.0)"""

new3 = """        try:
            lm = float(row.get("line_move_open")) if pd.notna(row.get("line_move_open")) else 0.0
        except Exception:
            lm = 0.0
        # v1.1: SPREAD line move weighted higher (half-point moves are more meaningful)
        if str(mkt).strip().upper() == "SPREAD":
            score += min(10.0, abs(lm) * 3.0)
        else:
            score += min(8.0, abs(lm) * 2.0)"""

if old3 in content:
    content = content.replace(old3, new3, 1)
    changes += 1
    print("CHANGE 3 OK: line move cap market-specific")
else:
    print("CHANGE 3 FAILED: anchor not found")

# --- CHANGE 4: key_number_note - SPREAD gets +6, others +3 ---
old4 = """        if str(row.get("key_number_note") or "").strip():
            score += 3"""

new4 = """        if str(row.get("key_number_note") or "").strip():
            # v1.1: SPREAD key number crossings are stronger signal (all sports)
            if str(mkt).strip().upper() == "SPREAD":
                score += 6
            else:
                score += 3"""

if old4 in content:
    content = content.replace(old4, new4, 1)
    changes += 1
    print("CHANGE 4 OK: key_number_note SPREAD +6")
else:
    print("CHANGE 4 FAILED: anchor not found")

# --- CHANGE 5: _game_decision TOTAL net_edge threshold 10 -> 12 ---
old5 = """        if s >= 72 and ne >= 10 and bool(strong_eligible):
            return 'STRONG_BET'
        if s >= 72 and ne >= 10:
            return 'BET'"""

new5 = """        # v1.1: TOTAL requires higher net_edge for STRONG/BET certification
        _ne_threshold = 12 if getattr(_game_decision, '_mkt', '') == 'TOTAL' else 10
        if s >= 72 and ne >= _ne_threshold and bool(strong_eligible):
            return 'STRONG_BET'
        if s >= 72 and ne >= _ne_threshold:
            return 'BET'"""

# Note: _game_decision doesn't have market context directly
# Better approach: use market-aware net_edge in game_view apply
old5 = """    game_view['game_decision'] = game_view.apply(
        lambda r: _game_decision(
            r.get('game_confidence', 50),
            r.get('net_edge', 0),
            r.get('strong_eligible', False)
        ), axis=1
    )"""

new5 = """    game_view['game_decision'] = game_view.apply(
        lambda r: _game_decision(
            r.get('game_confidence', 50),
            r.get('net_edge', 0),
            r.get('strong_eligible', False),
            r.get('market_display', '')
        ), axis=1
    )"""

if old5 in content:
    content = content.replace(old5, new5, 1)
    changes += 1
    print("CHANGE 5a OK: game_view apply passes market_display")
else:
    print("CHANGE 5a FAILED: anchor not found")

# Update _game_decision signature to accept market
old5b = """    def _game_decision(score, net_edge, strong_eligible=False):
        try:
            s = float(score)
        except Exception:
            s = 50.0
        try:
            ne = float(net_edge)
        except Exception:
            ne = 0.0
        if s >= 72 and ne >= 10 and bool(strong_eligible):
            return 'STRONG_BET'
        if s >= 72 and ne >= 10:
            return 'BET'
        if s >= 62:
            return 'LEAN'
        return 'NO BET'"""

new5b = """    def _game_decision(score, net_edge, strong_eligible=False, market=''):
        try:
            s = float(score)
        except Exception:
            s = 50.0
        try:
            ne = float(net_edge)
        except Exception:
            ne = 0.0
        # v1.1: TOTAL requires net_edge >= 12 for BET/STRONG_BET (suppress low-edge totals)
        _ne_min = 12 if str(market).strip().upper() == 'TOTAL' else 10
        if s >= 72 and ne >= _ne_min and bool(strong_eligible):
            return 'STRONG_BET'
        if s >= 72 and ne >= _ne_min:
            return 'BET'
        if s >= 62:
            return 'LEAN'
        return 'NO BET'"""

if old5b in content:
    content = content.replace(old5b, new5b, 1)
    changes += 1
    print("CHANGE 5b OK: _game_decision market-aware net_edge threshold")
else:
    print("CHANGE 5b FAILED: anchor not found")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTotal changes applied: {changes}/6")
