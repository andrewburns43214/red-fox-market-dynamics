"""
Red Fox — Email Notifications for New BET / STRONG_BET Decisions
Sends styled HTML email to all users with active Supabase access.
"""

import os
import json
import requests
from datetime import datetime, timezone

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://nwvosippnquwhtuppmkw.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("NOTIFY_FROM", "Red Fox Alerts <onboarding@resend.dev>")


def _get_active_user_emails():
    """Query Supabase for users with active access."""
    if not SUPABASE_SERVICE_KEY:
        print("[notify] SUPABASE_SERVICE_KEY not set — skipping")
        return []
    # Call the has_active_access RPC for each user, or just get all auth users
    # For simplicity, get all auth users (admin endpoint)
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    # List all users via admin API
    resp = requests.get(
        f"{SUPABASE_URL}/auth/v1/admin/users?per_page=500",
        headers=headers,
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"[notify] Failed to fetch users: {resp.status_code} {resp.text[:200]}")
        return []
    data = resp.json()
    users = data.get("users", data) if isinstance(data, dict) else data
    emails = []
    for u in users:
        email = u.get("email", "")
        if email and u.get("email_confirmed_at"):
            emails.append(email)
    return emails


def _build_email_html(new_bets):
    """Build HTML email body showing new bet alerts."""
    now_et = datetime.now(timezone.utc).strftime("%b %d, %Y %I:%M %p ET")

    rows_html = ""
    for b in new_bets:
        decision = b.get("game_decision", "")
        dec_color = "#00965a" if decision in ("BET", "STRONG_BET") else "#d48000"
        dec_bg = "#f0fdf4" if decision == "STRONG_BET" else "#fafafa"
        sport = (b.get("sport", "") or "").upper()
        game = b.get("game", "") or f'{b.get("favored_side", "")} game'
        market = b.get("market_display", "")
        pick = b.get("favored_side", "")
        line = b.get("decision_line", "") or b.get("current_line", "")
        score = b.get("total_score", b.get("game_confidence", ""))
        edge = b.get("net_edge", "")
        pattern = (b.get("pattern_primary", "") or "NEUTRAL").replace("_", " ")
        timing = b.get("timing_bucket", "")
        sharp = b.get("sharp_score", "")
        consensus = b.get("consensus_score", "")

        try:
            score_f = f"{float(score):.1f}"
        except (ValueError, TypeError):
            score_f = str(score)
        try:
            edge_f = f"{float(edge):.1f}"
        except (ValueError, TypeError):
            edge_f = str(edge)

        rows_html += f"""
        <tr style="background:{dec_bg}">
          <td style="padding:12px 16px;border-bottom:1px solid #e5e7eb">
            <div style="font-weight:700;color:#111;font-size:15px">{pick}</div>
            <div style="color:#6b7280;font-size:12px;margin-top:2px">{sport} · {market} · {timing}</div>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e7eb;text-align:center">
            <div style="font-weight:700;color:{dec_color};font-size:14px">{decision.replace('_',' ')}</div>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e7eb;text-align:center">
            <div style="font-weight:700;font-size:16px;color:#111">{score_f}</div>
            <div style="color:#6b7280;font-size:11px">Score</div>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e7eb;text-align:center">
            <div style="font-weight:600;font-size:14px;color:#111">{edge_f}</div>
            <div style="color:#6b7280;font-size:11px">Edge</div>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e7eb;text-align:center;font-size:12px;color:#374151">
            {line}
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #e5e7eb;text-align:center;font-size:11px;color:#6b7280">
            {pattern}
          </td>
        </tr>"""

    count = len(new_bets)
    subject_games = ", ".join(set(
        b.get("favored_side", "").split(" ")[0] for b in new_bets[:3]
    ))
    if count > 3:
        subject_games += f" +{count - 3} more"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f3f4f6">
  <div style="max-width:640px;margin:0 auto;padding:20px">
    <div style="background:#111827;border-radius:12px 12px 0 0;padding:20px 24px;text-align:center">
      <div style="color:#f97316;font-weight:800;font-size:20px;letter-spacing:1px">RED FOX</div>
      <div style="color:#9ca3af;font-size:12px;margin-top:4px">Market Intelligence</div>
    </div>
    <div style="background:#ffffff;padding:24px;border-radius:0 0 12px 12px;box-shadow:0 1px 3px rgba(0,0,0,.1)">
      <div style="font-size:18px;font-weight:700;color:#111;margin-bottom:4px">New Bet Alert</div>
      <div style="font-size:13px;color:#6b7280;margin-bottom:20px">{count} new {"bet" if count == 1 else "bets"} detected · {now_et}</div>
      <table style="width:100%;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif">
        <thead>
          <tr style="background:#f9fafb">
            <th style="padding:8px 16px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">PICK</th>
            <th style="padding:8px 16px;text-align:center;font-size:11px;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">DECISION</th>
            <th style="padding:8px 16px;text-align:center;font-size:11px;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">SCORE</th>
            <th style="padding:8px 16px;text-align:center;font-size:11px;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">EDGE</th>
            <th style="padding:8px 16px;text-align:center;font-size:11px;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">LINE</th>
            <th style="padding:8px 16px;text-align:center;font-size:11px;color:#6b7280;font-weight:600;border-bottom:2px solid #e5e7eb">PATTERN</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
      <div style="margin-top:24px;text-align:center">
        <a href="https://redfoxintel.com/board.html" style="display:inline-block;background:#f97316;color:#fff;font-weight:700;padding:12px 32px;border-radius:8px;text-decoration:none;font-size:14px">View Dashboard</a>
      </div>
      <div style="margin-top:20px;padding-top:16px;border-top:1px solid #f3f4f6;text-align:center;font-size:11px;color:#9ca3af">
        Red Fox Market Intelligence · Automated Alert
      </div>
    </div>
  </div>
</body>
</html>"""

    return html, subject_games


def _send_email(to_emails, subject, html_body):
    """Send email via Resend API."""
    if not RESEND_API_KEY:
        print("[notify] RESEND_API_KEY not set — skipping email send")
        return False
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": FROM_EMAIL,
                "to": to_emails,
                "subject": subject,
                "html": html_body,
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            print(f"[notify] Email sent to {len(to_emails)} users")
            return True
        else:
            print(f"[notify] Email failed: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[notify] Email send error: {repr(e)}")
        return False


def notify_new_bets(new_ds, old_ledger_df=None):
    """
    Called after freeze ledger write.
    new_ds: DataFrame of current engine run decisions
    old_ledger_df: DataFrame of previous freeze ledger (before merge)

    Detects NEW BET/STRONG_BET decisions and emails all active users.
    """
    import pandas as pd
    import re

    # Filter to BET/STRONG_BET only
    bet_rows = new_ds[new_ds["game_decision"].isin(["BET", "STRONG_BET"])].copy()
    if bet_rows.empty:
        return

    # Find which are truly NEW (not already in old ledger)
    if old_ledger_df is not None and not old_ledger_df.empty:
        old_bets = old_ledger_df[old_ledger_df["game_decision"].isin(["BET", "STRONG_BET"])].copy()
        if not old_bets.empty:
            # Normalize sides for comparison
            def _norm(s):
                return re.sub(r"\s*[+-]?\d+\.?\d*\s*$", "", str(s)).strip()
            bet_rows["_nside"] = bet_rows["side"].apply(_norm)
            old_bets["_nside"] = old_bets["side"].apply(_norm)
            # Merge to find new-only
            merged = bet_rows.merge(
                old_bets[["sport", "game_id", "market_display", "_nside"]].drop_duplicates(),
                on=["sport", "game_id", "market_display", "_nside"],
                how="left",
                indicator=True,
            )
            bet_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge", "_nside"])
            if bet_rows.empty:
                return

    # Deduplicate: keep only the favored side row per game+market
    bet_rows["_is_fav"] = bet_rows.apply(
        lambda r: str(r.get("side", "")).strip() == str(r.get("favored_side", "")).strip(),
        axis=1,
    )
    fav_rows = bet_rows[bet_rows["_is_fav"]]
    if fav_rows.empty:
        fav_rows = bet_rows.drop_duplicates(subset=["sport", "game_id", "market_display"], keep="first")

    new_bets = fav_rows.to_dict("records")
    if not new_bets:
        return

    print(f"[notify] {len(new_bets)} new BET/STRONG_BET decisions detected")

    # Get active user emails
    emails = _get_active_user_emails()
    if not emails:
        print("[notify] No active users to notify")
        return

    # Build and send email
    html_body, games_summary = _build_email_html(new_bets)
    subject = f"🦊 New Bet Alert — {games_summary}"
    _send_email(emails, subject, html_body)
