(() => {
  "use strict";
  const base = "/data/historical_comparisons/";
  const labels = {SPREAD: "Spread", MONEYLINE: "Moneyline", TOTAL: "Total"};
  const sports = {nfl: "NFL", ncaaf: "CFB", nba: "NBA", ncaab: "NCAAB", nhl: "NHL", mlb: "MLB", ufc: "UFC"};
  const cohorts = {closest: "Closest same sport", broader: "Broader same sport", cross: "Cross-sport context"};
  const icon = '<svg width="17" height="17" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><rect x="3" y="11" width="3.4" height="5.5" rx=".65"/><rect x="8.3" y="7.5" width="3.4" height="9" rx=".65"/><rect x="13.6" y="3.4" width="3.4" height="13.1" rx=".65"/></svg>';
  const escape = value => String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
  const gameKey = row => String(row.sport) + "--" + String(row.game_id);
  const odds = value => value == null ? "—" : (value > 0 ? "+" : "") + value;
  const pct = value => value == null ? "—" : value + "%";
  const signed = value => value == null ? "—" : (value > 0 ? "+" : "") + value;
  const record = result => [result.win, result.loss, result.push].join("–");
  const line = (market, number, price, name = "") => {
    if (market === "MONEYLINE") return odds(price);
    if (number == null) return "—";
    const mark = market === "SPREAD" ? signed(number) : (name ? name + " " : "") + number;
    return mark + (price == null ? "" : " (" + odds(price) + ")");
  };
  const when = (value, time = true) => value ? new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", month: "short", day: "numeric",
    ...(time ? {hour: "numeric", minute: "2-digit", timeZoneName: "short"} : {year: "numeric"})
  }).format(new Date(value)) : "—";
  const minuteText = value => value >= 60 ? `${Math.floor(value / 60)}h ${Math.round(value % 60)}m`
    : `${Math.max(0, Math.round(value))} min`;
  const chips = (items, style) => items.map(item => `<span class="historic-chip ${style}">${escape(item)}</span>`).join("");
  let manifest = {}, manifestHash = "", active = null, request = 0, lastButton = null;
  const dialog = () => document.getElementById("historical-comparison");
  const authorized = (url) => window.redfoxAuthorizedFetch(url, {cache: "no-store"});

  async function refreshManifest() {
    try {
      const response = await fetch(base + "index.json", {cache: "no-store"});
      const current = response.ok ? await response.json() : {};
      const fresh = Date.now() - Date.parse(current.generated) < 40 * 60000;
      manifest = current.enabled && fresh && current.games ? current.games : {};
      const hash = JSON.stringify(Object.entries(manifest).sort());
      if (hash !== manifestHash) {
        manifestHash = hash;
        if (dialog()?.open) return;
        if (document.body.classList.contains("live-recent-active")) window.renderLiveRecent();
        else window.renderBoard?.(false);
      }
    } catch (_) {
      // The normal board is independent of the optional research endpoint.
    }
  }

  function available(row) {
    return Array.isArray(manifest[gameKey(row)]) && manifest[gameKey(row)].includes(row.market_display);
  }

  function button(row, index, name) {
    return `<button type="button" class="historic-launch" title="Historical Market Comparisons"
      aria-haspopup="dialog" aria-label="Historical Market Comparisons for ${escape(name)}"
      data-historic-game="${escape(gameKey(row))}" data-historic-market="${escape(row.market_display)}"
      data-historic-lane="${index}" data-historic-observation="${escape(row.source_latest_at)}"
      onclick="event.stopPropagation();window.redfoxHistory.open(this)">${icon}</button>`;
  }

  function liveButton(row) {
    if (!available(row)) return "";
    const name = row.flagged_side;
    const side = window.marketSideIdentity(row, row);
    const lanes = row.market_display === "TOTAL" ? ["OVER", "UNDER"] :
      String(row.game).split(/ @ |\s+vs\.?\s+/i).map(team => team.trim().toLowerCase());
    const index = lanes.indexOf(side);
    return index < 0 ? "" : button(row, index, name);
  }

  async function open(button) {
    const token = ++request;
    lastButton = button;
    const modal = dialog();
    modal.querySelector("[data-historic-title-icon]").innerHTML = `<span class="historic-mark">${icon}</span>`;
    modal.querySelector("[data-historic-context]").textContent = "";
    modal.querySelector("[data-historic-favorite]").innerHTML = "";
    modal.querySelector("[data-historic-body]").innerHTML = '<p class="historic-empty" role="status">Loading historical comparisons…</p>';
    if (!modal.open) modal.showModal();
    active = null;
    try {
      const game = button.dataset.historicGame;
      if (!/^[a-z]+--[A-Za-z0-9_-]{1,75}$/.test(game)) throw new Error();
      const response = await authorized(base + game + ".json");
      if (!response.ok) throw new Error();
      const payload = await response.json();
      if (!modal.open || token !== request) return;
      const market = button.dataset.historicMarket;
      if (!payload.markets?.[market]) throw new Error();
      const index = Number(button.dataset.historicLane);
      active = {payload, market, team: market === "TOTAL" ? 0 : index, total: market === "TOTAL" ? index : 0,
        boardObservation: button.dataset.historicObservation, boardMarket: market, cohort: "closest"};
      render();
    } catch (_) {
      if (modal.open && token === request) modal.querySelector("[data-historic-body]").innerHTML =
        '<p class="historic-empty" role="status">A verified comparison is temporarily unavailable for this market. You can still open the game’s Market Journey from the board.</p>';
    }
  }

  function strongest(cross) {
    if (!cross) return "Cross-market evidence unavailable";
    const value = cross.market === "MONEYLINE" ? cross.price_toward_implied : cross.line_toward;
    if (value == null) return "Cross-market movement unavailable";
    const amount = cross.market === "MONEYLINE" ? `${signed(Number((value * 100).toFixed(1)))} implied pts`
      : `${signed(Number(value.toFixed(2)))} points`;
    const minimum = cross.market === "MONEYLINE" ? .005 : .25;
    const direction = value >= minimum ? "toward this side" : value <= -minimum ? "against this side" : "little movement";
    return `${labels[cross.market]} ${direction} (${amount})${cross.sustained_toward ? " across recent captures" : ""}`;
  }

  function setupCopy(setup, side, market) {
    const move = market === "MONEYLINE" ? side.price_toward_implied : side.line_toward;
    const tol = market === "MONEYLINE" ? .005 : .25;
    const magnitude = move == null ? "" : market === "MONEYLINE" ?
      `${Math.round(Math.abs(move) * 1000) / 10} implied points` : `${Math.abs(move)} ${Math.abs(move) === 1 ? "point" : "points"}`;
    const direction = move == null ? "Movement is unavailable." : move >= tol ? `The market moved ${magnitude} toward this side.`
      : move <= -tol ? `The market moved ${magnitude} against this side.` : "The market showed little directional movement.";
    const other = setup.sides.find(item => item.index !== side.index);
    const signal = side.context || {};
    const breach = {ticket_only: "Ticket threshold breach", money_only: "Money threshold breach", both: "Ticket and money threshold breach"};
    const phases = {initial_confirmation: "initial confirmation", removal: "removal", reinstatement: "reinstatement"};
    const audit = side.split_audit;
    const tags = [...new Set([signal.reaction, signal.path, signal.evidence_role, signal.data_badge && signal.data_badge + " data",
      audit && breach[audit.split_breach] && breach[audit.split_breach] +
        (phases[audit.phase] ? " · " + phases[audit.phase] : ""),
      ...(signal.context_chips || "").split("|").map(item => item.trim())].filter(Boolean))];
    return `<section class="historic-setup"><h3>Market setup</h3><p><strong>${escape(side.name)}</strong>
      drew ${pct(side.tickets)} of tickets and ${pct(side.money)} of money. ${direction}
      ${other?.tickets != null && other?.money != null ? `${escape(other.name)} drew ${pct(other.tickets)} of tickets and ${pct(other.money)} of money.` : ""}</p>
      <dl><div><dt>Opening → observed price</dt><dd>${escape(line(market, side.open_line, side.open_price))} → ${escape(line(market, side.line, side.price))}</dd></div>
      <div><dt>First paired capture → observed splits</dt><dd>${pct(side.opening_tickets)} / ${pct(side.opening_money)} → ${pct(side.tickets)} / ${pct(side.money)}</dd></div>
      <div><dt>Time to scheduled start</dt><dd>${minuteText(setup.lead)} at observation</dd></div>
      <div><dt>Cross-market evidence</dt><dd>${escape(market === "TOTAL" ? "Totals are a separate outcome" : strongest(side.cross_market))}</dd></div></dl>
      ${tags.length ? `<div class="historic-chips">${chips(tags, "")}</div>` : ""}</section>`;
  }

  function tile(label, value, sub = "", tone = "") {
    return `<div class="historic-metric"><span>${escape(label)}</span>
      <strong class="${tone}">${escape(value)}</strong>${sub ? `<small>${escape(sub)}</small>` : ""}</div>`;
  }

  function examples(result, market) {
    if (!result.examples.length) return "";
    const resultLabel = grade => ({win: market === "SPREAD" ? "Covered" : "Won",
      loss: market === "SPREAD" ? "Did not cover" : "Lost", push: "Push"}[grade] || grade);
    return `<section class="historic-examples"><h3>Closest verified examples</h3>
      ${result.examples.map(item => `<details class="historic-example">
        <summary><span><strong>${escape(item.game)}</strong><small>${escape(sports[item.sport])} · ${escape(when(item.date, false))}</small></span>
        <span><strong>${escape(item.selection || item.side.side)}</strong><small>${escape(line(market, item.side.line, item.side.price))} · ${pct(item.side.tickets)} tickets · ${pct(item.side.money)} money</small></span>
        <span class="historic-result ${item.grade === "win" ? "positive" : item.grade === "loss" ? "negative" : ""}">${escape(resultLabel(item.grade))}</span></summary>
        <div class="historic-example-detail"><p>${escape(item.score || "Verified settlement")} · Entry captured ${escape(when(item.entry))}.</p>
        <p>Opening ${escape(line(market, item.side.open_line, item.side.open_price))} → entry ${escape(line(market, item.side.line, item.side.price))}.
        ${escape(market === "TOTAL" ? "" : strongest(item.side.cross_market) + ".")}</p>
        ${item.matched.length ? `<p><strong>Similar</strong> ${escape(item.matched.join(", "))}.</p>` : ""}
        ${item.differences.length ? `<p><strong>Differences</strong> ${escape(item.differences.join(", "))}.</p>` : ""}</div>
      </details>`).join("")}</section>`;
  }

  function resultCopy(side, market) {
    const report = side.comparison;
    if (report.reason) return `<p class="historic-empty"><strong>Insufficient history.</strong> ${escape(report.reason)}</p>`;
    const result = report[active.cohort];
    if (!result) return '<p class="historic-empty">Cross-sport context is unavailable. Red Fox currently compares normalized moneyline setups only between comparable football or basketball leagues.</p>';
    const tone = value => value == null ? "" : value > 0 ? "positive" : value < 0 ? "negative" : "";
    const message = result.status === "none" ? "No comparable games in this group."
      : result.status === "limited" ? "Limited sample. Small records can swing sharply." : "";
    return `
      ${active.cohort === "cross" ? `<p class="historic-note">Normalized moneyline context from ${escape(report.cross_sports.map(s => sports[s]).join(", "))}.</p>` : ""}
      <div class="historic-metrics">${tile("Matched games", result.matched, result.graded + " priced and graded")}
        ${tile("Record", record(result), "Wins · losses · pushes")}
        ${tile("Win rate", pct(result.win_rate), "Excludes pushes")}
        ${tile("Avg. entry odds", odds(result.odds), result.line != null ? "Avg. line " + (market === "SPREAD" ? signed(result.line) : result.line) : "")}
        ${tile("Net units", result.units == null ? "—" : signed(result.units) + "u", "", tone(result.units))}
        ${tile("ROI", result.roi == null ? "—" : signed(result.roi) + "%", "", tone(result.roi))}</div>
      ${message ? `<p class="historic-notice">${escape(message)}</p>` : ""}
      ${result.unresolved || result.missing_price || result.void ? `<p class="historic-note">${result.unresolved} unresolved · ${result.missing_price} missing entry prices · ${result.void} void. These are excluded from performance.</p>` : ""}
      ${result.matched ? `<section><h3>What matched</h3><div class="historic-chips">${chips(["Comparable decision time", ...result.matches], "match")}
       ${chips(result.differences.map(item => "Varied: " + item), "difference")}</div></section>` : ""}
      ${examples(result, market)}`;
  }

  function control(label, value, group, on, disabled = false) {
    return `<button type="button" data-historic-action="${group}" data-value="${escape(value)}"
      aria-pressed="${on}"${disabled ? " disabled" : ""}>${escape(label)}</button>`;
  }

  function render() {
    const {payload, market, team, total, cohort} = active;
    const modal = dialog(), setup = payload.markets[market], index = market === "TOTAL" ? total : team;
    const side = setup.sides[index];
    const body = modal.querySelector("[data-historic-body]");
    const focus = modal.contains(document.activeElement) ? document.activeElement.dataset.historicAction : null;
    const selectedFocus = focus ? document.activeElement.dataset.value : null;
    modal.querySelector("[data-historic-context]").textContent = `${sports[payload.sport]} · ${payload.game}`;
    modal.querySelector("[data-historic-favorite]").innerHTML = side.favorite ?
      '<img src="/assets/red-fox-favorite.png" alt="Red Fox Favorite">' : "";
    const stale = !payload.frozen && market === active.boardMarket &&
      Date.parse(active.boardObservation) > Date.parse(setup.observed);
    const marketButtons = Object.entries(labels).filter(([key]) => payload.markets[key]);
    body.innerHTML = `
      <div class="historic-controls">
        <div><span>${market === "TOTAL" ? "Outcome" : payload.sport === "ufc" ? "Fighter" : "Team"}</span>
          <div class="historic-options">${setup.sides.map((item, i) => control(item.name, i, "side", i === index)).join("")}</div></div>
        <div><span>Market</span><div class="historic-options markets">${marketButtons.map(([key, label]) =>
          control(label, key, "market", key === market)).join("")}</div></div>
      </div>
      <p class="historic-note">${payload.frozen ? "Frozen pregame capture" : stale ? "Latest processed capture · a newer board observation is pending" : "Market observation"} · ${escape(when(setup.observed))}</p>
      ${setupCopy(setup, side, market)}
      <div class="historic-cohorts" aria-label="Historical comparison group">
        ${Object.entries(cohorts).map(([key, label]) => control(label, key, "cohort", key === cohort)).join("")}
      </div>
      <p class="historic-note">${cohort === "closest" ? "Tighter matching on splits, price, and move direction within the same sport."
        : cohort === "broader" ? "Relaxes tolerances for splits, price, and movement. Includes any games in the closest group."
          : "Compares price and market movement in implied probability, with similar betting splits and timing."}
        ${side.comparison.since ? `Coverage for this market and window starts ${escape(side.comparison.since)}.` : ""}</p>
      ${resultCopy(side, market)}
      <section class="historic-by-market"><h3>Similar results by market</h3>
      <p class="historic-note">Each market has its own closest same-sport group.</p>
      <div>${marketButtons.map(([key, label]) => {
        const current = payload.markets[key].sides[key === "TOTAL" ? total : team];
        const metric = current.comparison.closest;
        const summary = !metric || !metric.graded ? "No graded matches"
          : `${record(metric)} · ${metric.graded} graded · ${pct(metric.win_rate)} wins · ${signed(metric.roi)}% ROI`;
        return `<button type="button" data-historic-action="market" data-value="${key}" aria-pressed="${key === market}">
          <strong>${escape(label + (key === "TOTAL" ? " · " + current.name : ""))}</strong>
          <span>${escape(summary)}</span></button>`;
      }).join("")}</div></section>
      <p class="historic-method">Historical results include Favorites and non-Favorites. Net units and ROI risk one unit at each recorded entry price. Pushes return stake. Voids are excluded. Historical performance does not guarantee future results.</p>`;
    if (focus) {
      const element = [...body.querySelectorAll("[data-historic-action]")].find(el =>
        el.dataset.historicAction === focus && el.dataset.value === selectedFocus);
      element?.focus({preventScroll: true});
    }
  }

  function init() {
    const modal = dialog();
    if (!modal) return;
    modal.addEventListener("click", event => {
      const picked = event.target.closest("[data-historic-action]");
      if (!picked || !active) return;
      const {historicAction, value} = picked.dataset;
      if (historicAction === "market") {
        active.market = value;
        active.cohort = "closest";
      } else if (historicAction === "cohort") active.cohort = value;
      else active[active.market === "TOTAL" ? "total" : "team"] = Number(value);
      render();
    });
    modal.addEventListener("close", () => {
      request++;
      active = null;
      lastButton?.focus({preventScroll: true});
    });
    refreshManifest();
    window.setInterval(() => { if (!document.hidden) refreshManifest(); }, 120000);
  }

  window.redfoxHistory = {available, button, liveButton, open};
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})();
