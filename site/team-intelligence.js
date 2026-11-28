/* Team Intelligence consumes the separate, read-only archive snapshot on demand. */
(() => {
  "use strict";
  const markets = ["SPREAD", "MONEYLINE", "TOTAL"];
  const sports = new Set(["nfl", "ncaaf", "nba", "ncaab", "mlb", "nhl", "ufc"]);
  const labels = { nfl: "NFL", ncaaf: "CFB", nba: "NBA", ncaab: "NCAAB", mlb: "MLB", nhl: "NHL", ufc: "UFC" };
  const cache = new Map();
  const date = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" });
  const et = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  const markup = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  const when = (value, format = date) => value && Number.isFinite(Date.parse(value)) ? format.format(new Date(value)) : "Unavailable";
  const display = (number, digits = 1) => Number.isFinite(number) ? number.toFixed(digits) : "—";
  const signed = (number, digits = 0) => Number.isFinite(number) ? (number > 0 ? "+" : "") + display(number, digits).replace("-", "−") : "—";
  const pct = (n) => n === "" || n == null || !Number.isFinite(Number(n)) ? "—" : Math.round(Number(n)) + "%";
  const keyName = (name) => String(name).normalize("NFC").trim().replace(/\s+/g, " ").toLowerCase();
  const day = 86400000;
  let dialog, state, previousFocus, request = 0;

  async function load(sport) {
    const hit = cache.get(sport);
    if (hit && hit.expires > Date.now()) return hit.promise;
    const endpoint = "/data/team-intelligence/" + sport + ".json";
    const fetcher = window.redfoxAuthorizedFetch || fetch;
    const promise = fetcher(endpoint, { cache: "no-cache", signal: AbortSignal.timeout(10000) }, { required: false })
      .then(response => {
        if (!response.ok) throw new Error("Team snapshot unavailable");
        return response.json();
      }).then(snapshot => {
        if (snapshot.version !== 1 || snapshot.sport !== sport || !snapshot.fixtures || !snapshot.games || !snapshot.teams) {
          throw new Error("Team snapshot is invalid");
        }
        return snapshot;
      }).catch(error => {
        cache.delete(sport);
        throw error;
      });
    cache.set(sport, { promise, expires: Date.now() + 120000 });
    return promise;
  }

  function fixture(snapshot, identity) {
    const row = Object.hasOwn(snapshot.fixtures, identity.fixture) && snapshot.fixtures[identity.fixture];
    if (!row || !Array.isArray(row.names) || !row.names.every((name, i) => keyName(name) === keyName(identity.game.split(/\s+@\s+|\s+vs?\.?\s+/i)[i]))) return null;
    if (identity.start && !(row.published_starts || [row.start]).some(start => Date.parse(start) === Date.parse(identity.start))) return null;
    return row;
  }

  function ensureDialog() {
    if (dialog) return;
    dialog = document.createElement("dialog");
    dialog.className = "team-intelligence";
    dialog.setAttribute("aria-labelledby", "ti-title");
    dialog.innerHTML = '<header class="ti-header"><div><p>Red Fox · Team Intelligence</p><h2 id="ti-title"></h2><div class="ti-matchup"></div></div><button type="button" class="ti-close" aria-label="Close Team Intelligence" autofocus>×</button></header><div class="ti-body"></div>';
    dialog.querySelector(".ti-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
      request++;
      previousFocus?.focus();
    });
    dialog.addEventListener("click", event => {
      const control = event.target.closest("[data-ti-action]");
      if (!control || !state?.fixture) return;
      const { tiAction, tiValue } = control.dataset;
      if (tiAction === "team") {
        state.team = Number(tiValue);
        if (state.tab === "compare") state.tab = "overview";
      }
      if (tiAction === "tab") state.tab = tiValue;
      if (tiAction === "market") state.market = tiValue;
      if (tiAction === "all-games") state.expanded = !state.expanded;
      if (tiAction === "total") state.total = tiValue;
      render(tiAction, tiValue);
    });
    document.body.append(dialog);
  }

  function control(label, type, value, selected, extraClass = "") {
    return '<button type="button" class="' + extraClass + '" data-ti-action="' + type + '" data-ti-value="' + markup(value) + '" aria-pressed="' + !!selected + '">' + markup(label) + '</button>';
  }

  function scoredGames(key) {
    const { snapshot, fixture } = state;
    const floor = Math.min(Date.parse(fixture.start), Date.parse(snapshot.generated));
    return (snapshot.teams[key] || []).map(id => snapshot.games[id]).filter(game =>
      Date.parse(game.start) < floor
      && (fixture.start > snapshot.generated || (game.source !== "nflverse" && Date.parse(game.source_at) < floor))
    );
  }

  function seasonView(games) {
    const { snapshot, fixture } = state;
    const source = snapshot.sources.nflverse;
    const league = !!source?.complete_regular_schedules?.includes(fixture.season) && fixture.start > snapshot.generated
      && !source.error && Date.parse(snapshot.generated) - Date.parse(source.at) <= 3 * day;
    const selected = games.filter(game => game.season === fixture.season && (!league || (game.source === "nflverse" && game.stage === "REG")));
    const end = ["nba", "nhl", "ncaab"].includes(snapshot.sport) ? "–" + String(fixture.season + 1).slice(2) : "";
    const span = snapshot.sport === "ufc" ? "Calendar year " + fixture.season : fixture.season + end;
    return { league, games: selected, name: league ? span + " NFL regular season" : span + " Red Fox coverage" };
  }

  function result(game, key) {
    if (game.winner) return game.winner === key ? "Win" : "Loss";
    const i = game.keys.indexOf(key);
    if (i < 0 || !game.scores) return "";
    const differential = game.scores[i] - game.scores[1 - i];
    return differential > 0 ? "Win" : differential < 0 ? "Loss" : "Tie";
  }

  function form(games, key) {
    const count = { Win: 0, Loss: 0, Tie: 0 };
    const scores = [];
    for (const game of games) {
      count[result(game, key)]++;
      if (game.scores) {
        const i = game.keys.indexOf(key);
        scores.push([game.scores[i], game.scores[1 - i]]);
      }
    }
    const average = (i) => scores.length ? scores.reduce((sum, pair) => sum + pair[i], 0) / scores.length : null;
    return { record: games.length ? [count.Win, count.Loss, ...(count.Tie ? [count.Tie] : [])].join("–") : "—",
      scored: average(0), allowed: average(1), games: games.length };
  }

  const fact = (value, label) => '<div class="ti-fact"><strong>' + markup(value) + '</strong><span>' + markup(label) + '</span></div>';
  const note = text => '<p class="ti-note">' + markup(text) + '</p>';
  const heading = label => '<h3>' + markup(label) + '</h3>';
  const marketLabel = (market) => ({ SPREAD: "Spread", MONEYLINE: "Moneyline", TOTAL: "Total" })[market];

  function recentRows(games, team) {
    return '<div class="ti-list">' + games.map(game => {
      const i = game.keys.indexOf(team);
      const gameResult = result(game, team);
      const line = game.scores ? game.scores.join("–") : (game.winner ? game.names[game.keys.indexOf(game.winner)] + " won" : "Result unavailable");
      return '<div class="ti-score-row"><div><strong>' + markup(game.names.join(state.snapshot.sport === "ufc" ? " vs " : " @ "))
         + '</strong><small title="Verified ' + markup(when(game.source_at, et)) + ' ET">' + markup(when(game.start))
        + ' · ' + markup(game.source === "nflverse" ? "nflverse" : "Verified archive")
        + '</small></div><span title="' + markup(game.names[i]) + ' selected">' + markup(line) + '</span><span class="ti-result" data-result="'
        + markup(gameResult.toLowerCase()) + '">' + markup(gameResult) + '</span></div>';
    }).join("") + '</div>';
  }

  function overview(games, team) {
    const scope = seasonView(games), f = form(scope.games, team);
    const recordLabel = state.snapshot.sport === "nhl" ? "Wins–losses (OT included)" : state.snapshot.sport === "ufc" ? "Verified wins–losses" : "Record";
    let html = heading(scope.name) + '<div class="ti-facts">' + fact(f.record, recordLabel) + fact(f.games, "Verified results")
      + (state.snapshot.sport === "ufc" ? "" : fact(display(f.scored), "Scored per game") + fact(display(f.allowed), "Allowed per game")) + '</div>';
    if (scope.league) {
      const home = scope.games.filter(game => game.venue === "home" && game.keys[1] === team);
      const away = scope.games.filter(game => game.venue === "home" && game.keys[0] === team);
      html += '<div class="ti-facts ti-venue">' + fact(form(home, team).record, "Home (" + home.length + ")")
        + fact(form(away, team).record, "Away (" + away.length + ")") + '</div>' + note("Venue records exclude neutral and unverified sites.");
    } else {
      html += note(state.snapshot.sport === "ufc"
        ? "Includes archive bouts with a verified winner and individual bout timing. Draws and no contests are outside this view."
        : "Record and scoring cover verified results in the Red Fox archive. Season coverage varies by team.");
    }
    const late = games.slice(0, 5);
    html += heading(state.snapshot.sport === "ufc" ? "Last five verified bouts" : "Last five verified games");
    html += late.length ? recentRows(late, team) : note("No verified results for this team before the selected game.");
    return html;
  }

  function quoteText(market, line, price) {
    if (market === "MONEYLINE") return signed(price);
    const quoteLine = Number.isFinite(line) ? (market === "SPREAD" && line > 0 ? "+" : "") + String(line).replace("-", "−") : "—";
    return quoteLine + (Number.isFinite(price) ? " (" + signed(price) + ")" : "");
  }

  function movementText(market, quote) {
    const line = quote.line_toward, price = Number.isFinite(quote.price_toward_implied) ? 100 * quote.price_toward_implied : null;
    const direction = value => value > 0 ? " toward selection" : " against selection";
    const output = [];
    if (market !== "MONEYLINE" && Number.isFinite(line)) {
      output.push(!line ? "Line unchanged" : display(Math.abs(line)) + " line pts" + direction(line));
    }
    if (price !== null) {
      output.push(Math.abs(price) < 0.05 ? (price === 0 ? "Price unchanged" : "Price moved <0.1 implied pt" + direction(price))
        : display(Math.abs(price)) + " implied pts" + direction(price));
    }
    return output.join(" · ");
  }

  function officialHistory(games, market, team, total) {
    const previous = games.flatMap(game => (game.favorites || [])
      .filter(row => row.market === market && row.decision_price && row.side_key === (market === "TOTAL" ? total : team))
      .map(row => ({ game, row })));
    if (!previous.length) return "";
    return heading("Previous official Favorites") + '<div class="ti-list">' + previous.slice(0, 12).map(({game, row}) => {
      const entry = (row.decision_line ? row.decision_line + " @ " : "") + row.decision_price;
      const grade = ({ W: "Win", L: "Loss", P: "Push", V: "Void" })[row.grade] || row.grade || "Pending";
      const source = row.decision_line_source ? " · " + row.decision_line_source.replaceAll("_", " ") : "";
      const close = row.closing_price && row.closing_price_source ? " · ledger close "
        + (row.closing_line ? row.closing_line + " @ " : "") + row.closing_price
        + " (" + row.closing_price_source.replaceAll("_", " ") + ")" : "";
      return '<div class="ti-quote-row"><div><strong>' + markup(game.names.join(state.snapshot.sport === "ufc" ? " vs " : " @ "))
        + '</strong><small>' + markup(when(game.start) + " · " + row.side) + '</small></div><div>Entry ' + markup(entry)
        + '<small>' + markup(when(row.decision_quote_observed_at, et) + " ET" + source + close) + '</small></div><span class="ti-result" data-result="'
        + markup(grade.toLowerCase()) + '">' + markup(grade) + '</span></div>';
    }).join("") + '</div>' + (previous.length > 12 ? note("Showing the 12 most recent official Favorites.") : "");
  }

  function historical(games, team) {
    const { fixture, market, snapshot, total } = state;
    const subject = market === "TOTAL" ? total : team;
    const sample = games.filter(game => game.quotes?.[market]?.[subject]).slice(0, 12);
    const counts = { win: 0, loss: 0, push: 0 };
    let net = 0;
    for (const game of sample) {
      const quote = game.quotes[market][subject];
      counts[quote.grade]++;
      net += quote.units;
    }
    const current = fixture.markets?.[market];
    let html = '<div class="ti-selectors">' + (snapshot.sport === "ufc" ? "" : markets.map(value => control(marketLabel(value), "market", value, value === market)).join(""))
      + (market === "TOTAL" ? '<span class="ti-separator"></span>' + ["over", "under"].map(value => control(value === "over" ? "Over" : "Under", "total", value, value === total)).join("") : "") + '</div>';
    html += heading("Selected matchup");
    if (current?.length) {
      html += '<div class="ti-list">' + current.map(row => '<div class="ti-quote-row ti-current-quote"><strong>' + markup(row.flagged_side)
        + '</strong><span>' + markup(row.open_line || "—") + ' → ' + markup(row.current_line || "—")
        + '</span><small>' + pct(row.bets_pct) + ' bets · ' + pct(row.money_pct) + ' money'
        + (row.source_latest_at ? ' · ' + markup(when(row.source_latest_at, et)) + ' ET' : '') + '</small></div>').join("") + '</div>';
      const retained = current.some(row => /retained|stale/i.test(row.quality));
      html += note(retained ? "Last retained market quote. See Game Detail for its current status."
        : current.some(row => row.frozen) ? "Frozen pregame record for this matchup."
          : "Latest market quote in this snapshot. See Game Detail for the newest board capture.");
    } else {
      html += note("A current quote for this market is unavailable.");
    }
    html += heading("Verified pregame market history");
    const official = officialHistory(games, market, team, total);
    if (!sample.length) return html + note("No priced, graded archive quotes for this selection yet. Completed scores may still be available.") + official;
    html += '<div class="ti-facts">' + fact(sample.length, "Priced games")
      + fact([counts.win, counts.loss, counts.push].join("–"), "Wins–losses–pushes")
      + fact(signed(net, 2) + "u", "Net units") + fact(signed(100 * net / sample.length, 1) + "%", "Flat-unit ROI") + '</div>';
    html += note("Up to 12 recent priced results. Each game uses one verified pregame checkpoint, which can precede the closing quote. Net units and ROI use one unit risked per game.");
    html += '<div class="ti-list">' + sample.map(game => {
      const quote = game.quotes[market][subject];
      const from = quoteText(market, quote.open_line, quote.open_price);
      const to = quoteText(market, quote.line, quote.price);
      const primary = market === "TOTAL" ? (total === "over" ? "Over" : "Under") : game.names[game.keys.indexOf(team)];
      const grade = quote.grade[0].toUpperCase() + quote.grade.slice(1);
      return '<div class="ti-quote-row ti-grade-quote"><div><strong>' + markup(game.names.join(snapshot.sport === "ufc" ? " vs " : " @ ")) + '</strong><small>'
        + markup(when(game.start)) + ' · ' + markup(primary) + '</small></div><div>' + markup(from + ' → ' + to)
        + '<small>' + pct(quote.tickets) + ' bets · ' + pct(quote.money) + ' money</small></div><span class="ti-result" data-result="'
        + quote.grade + '">' + grade + '</span><div class="ti-quote-meta"><small>Quote observed ' + markup(when(quote.at, et)) + ' ET'
        + (game.scores ? ' · Final ' + markup(game.scores.join("–")) : '') + '</small>'
        + (movementText(market, quote) ? '<small>' + markup(movementText(market, quote)) + '</small>' : '')
        + (quote.path_direction_changes > 0 ? '<small>' + markup(quote.path_direction_changes + " direction changes in the observed path") + '</small>' : '')
        + '</div></div>';
    }).join("") + '</div>';
    return html + official;
  }

  function compare() {
    const { fixture, snapshot } = state;
    const sides = fixture.keys.map((key, i) => {
      const games = scoredGames(key), scope = seasonView(games);
      return { name: fixture.names[i], form: form(scope.games.slice(0, 5), key), chart: scope, record: form(scope.games, key) };
    });
    let html = '<div class="ti-compare">' + sides.map(side => '<section><h4>' + markup(side.name) + '</h4>'
      + note(side.chart.name) + '<div class="ti-facts">' + fact(side.record.record, "Record · " + side.record.games)
      + fact(side.form.record, "Recent form · " + side.form.games)
      + (snapshot.sport === "ufc" ? "" : fact(display(side.form.scored), "Recent scoring") + fact(display(side.form.allowed), "Recent allowed"))
      + '</div></section>').join("") + '</div>';
    const [a, b] = sides;
    if (snapshot.sport !== "ufc" && a.form.games === 5 && b.form.games === 5) {
      const gap = a.form.scored - b.form.scored;
      const read = Math.abs(gap) < 0.05 ? "Each team averaged " + display(a.form.scored) + " per game across five verified results."
        : (gap > 0 ? a.name : b.name) + " scored " + display(Math.abs(gap)) + " more per game on average across the last five verified results.";
      const defense = a.form.allowed - b.form.allowed;
      const pair = defense > 0 ? [b, a] : [a, b];
      const defenseRead = Math.abs(defense) < 0.05 ? "The teams allowed the same average in those samples."
        : pair[0].name + " allowed " + display(pair[0].form.allowed) + " per game, compared with " + display(pair[1].form.allowed)
          + " for " + pair[1].name + ".";
      const recentDates = fixture.keys.map(key => scoredGames(key)[0]?.start);
      const reference = Math.min(Date.parse(fixture.start), Date.parse(snapshot.generated));
      const gaps = recentDates.map(value => (reference - Date.parse(value)) / day);
      const schedule = gaps.every(value => Number.isFinite(value))
        ? " Their most recent verified games were " + when(recentDates[0]) + " and " + when(recentDates[1]) + ", respectively." : "";
      html += heading("What the results show") + note(read + " " + defenseRead + schedule
        + " These comparisons reflect different opponent schedules and venues.");
    } else {
      html += note(snapshot.sport === "ufc" ? "The comparison uses verified bout results. Fighter style, opponent changes and camp news require separate evidence."
        : "Five verified results per team are needed for a recent scoring comparison.");
    }
    return html;
  }

  function render(action, value) {
    const { fixture, snapshot, tab, team, expanded } = state;
    const key = fixture.keys[team], games = scoredGames(key);
    dialog.querySelector("#ti-title").textContent = tab === "compare" ? "Matchup comparison" : fixture.names[team];
    dialog.querySelector(".ti-matchup").textContent = labels[snapshot.sport] + " · " + fixture.game + " · " + when(fixture.start);
    const recent = games.slice(0, expanded ? games.length : 10);
    const tabs = [["Overview", "overview"], ["Recent results", "results"], ["Market history", "markets"], ["Compare", "compare"]];
    let content;
    if (tab === "compare") content = compare();
    else if (tab === "markets") content = historical(games, key);
    else if (tab === "results") content = heading("Verified results")
      + (recent.length ? recentRows(recent, key) : note("No verified results for this team before the selected game."))
      + (games.length > 10 ? control(expanded ? "Show latest 10" : "Show all " + games.length, "all-games", "toggle", expanded) : "");
    else content = overview(games, key);
    const source = snapshot.sources.nflverse;
    const sourceLink = source?.at && /^https:\/\/github\.com\/nflverse\//.test(source.url)
      ? ' · <a href="' + markup(source.url) + '" target="_blank" rel="noopener noreferrer">nflverse schedule</a>' : "";
    const stale = Date.now() - Date.parse(snapshot.generated) > 4 * 3600000;
    dialog.querySelector(".ti-body").innerHTML = '<div class="ti-selectors ti-teams" aria-label="Team">' + fixture.names.map((name, i) => control(name, "team", i, team === i && tab !== "compare")).join("")
      + '</div><nav class="ti-selectors ti-tabs" aria-label="Team Intelligence views">' + tabs.map(([label, value]) => control(label, "tab", value, tab === value)).join("")
      + '</nav><div class="ti-content">' + content + '</div><footer class="ti-source">' + (stale ? "Last available snapshot · " : "Snapshot updated · ")
      + markup(when(snapshot.generated, et)) + ' ET · Red Fox verified archive' + sourceLink + '</footer>';
    if (action) {
      Array.from(dialog.querySelectorAll("[data-ti-action]")).find(el => el.dataset.tiAction === action && el.dataset.tiValue === value)?.focus({ preventScroll: true });
    }
  }

  document.addEventListener("click", async event => {
    const button = event.target.closest("[data-team-intelligence]");
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const identity = button.dataset;
    if (!sports.has(identity.sport)) return;
    ensureDialog();
    previousFocus = button;
    state = null;
    const id = ++request;
    dialog.querySelector("#ti-title").textContent = button.textContent;
    dialog.querySelector(".ti-matchup").textContent = identity.game;
    const body = dialog.querySelector(".ti-body");
    body.innerHTML = '<p class="ti-status" role="status">Loading verified team evidence…</p>';
    if (!dialog.open) dialog.showModal();
    try {
      const snapshot = await load(identity.sport);
      if (id !== request) return;
      const match = fixture(snapshot, identity);
      if (!match) {
        body.innerHTML = '<p class="ti-status" role="status">This matchup isn’t in the current verified snapshot. Try again after the next update.</p>';
        return;
      }
      state = { snapshot, fixture: match, team: Number(identity.team), tab: "overview", expanded: false,
        market: snapshot.sport === "ufc" ? "MONEYLINE" : (markets.includes(identity.market) ? identity.market : "SPREAD"), total: "over" };
      render();
      body.scrollTop = 0;
    } catch (error) {
      if (id === request) body.innerHTML = '<p class="ti-status" role="status">Team Intelligence is temporarily unavailable. The Market Board is still available.</p>';
    }
  }, true);
})();
