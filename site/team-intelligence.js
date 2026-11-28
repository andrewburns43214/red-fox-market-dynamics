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
      previousFocus?.setAttribute("aria-expanded", "false");
      previousFocus?.focus();
    });
    dialog.addEventListener("toggle", event => {
      const details = event.target;
      if (!details.open || !state?.fixture) return;
      const more = details.querySelector("[data-ti-more]");
      if (more && !more.childElementCount) {
        const team = state.fixture.keys[state.team];
        more.innerHTML = recentRows(scoredGames(team).slice(5),team);
      }
      if (details.matches("[data-ti-observations]")) void loadObservations(details);
    },true);
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

  const fact = (value, label, tone = "") => '<div class="ti-fact" data-tone="' + tone + '"><strong>' + markup(value) + '</strong><span>' + markup(label) + '</span></div>';
  const note = text => '<p class="ti-note">' + markup(text) + '</p>';
  const heading = label => '<h3>' + markup(label) + '</h3>';
  const marketLabel = (market) => ({ SPREAD: "Spread", MONEYLINE: "Moneyline", TOTAL: "Total" })[market];

  const course = () => ({nhl:"Goals",mlb:"Runs"})[state.snapshot.sport] || "Points";

  function scoreAgainst(game, team) {
    if (!game.scores) return "";
    const i = game.keys.indexOf(team);
    return game.scores[i] + "–" + game.scores[1 - i];
  }

  function streak(games, team) {
    if (!games.length) return "—";
    const grade = result(games[0], team);
    const end = games.findIndex(game => result(game, team) !== grade);
    return grade[0] + (end < 0 ? games.length : end);
  }

  function seasonDetails(games, team) {
    const scope = seasonView(games);
    const year = form(scope.games, team);
    const home = scope.games.filter(g => g.venue === "home" && g.keys[1] === team);
    const away = scope.games.filter(g => g.venue === "home" && g.keys[0] === team);
    const last = games.slice(0, 5);
    const recent = form(last, team);
    const margin = Number.isFinite(year.scored) ? year.scored - year.allowed : null;
    return {scope, year, last, recent, margin, home: form(home, team), away: form(away, team), streak: streak(games,team)};
  }

  function nflContext(team, scope) {
    if (!scope.league) return {};
    if (!state.nflContext) {
      const { snapshot, fixture } = state;
      const before = Math.min(Date.parse(fixture.start), Date.parse(snapshot.generated));
      const league = Object.values(snapshot.games).filter(game => game.source === "nflverse" && game.stage === "REG"
        && game.season === fixture.season && Date.parse(game.start) < before);
      const opponents = new Map();
      for (const game of league) {
        for (const key of game.keys) {
          if (!opponents.has(key)) opponents.set(key,[]);
          opponents.get(key).push(game);
        }
      }
      const data = new Map();
      for (const [key,games] of opponents) {
        games.sort((a,b) => b.start.localeCompare(a.start));
        const quality = games.map(game => {
          const opponent = game.keys.find(other => other !== key);
          const sample = (opponents.get(opponent) || []).filter(row => !row.keys.includes(key));
          const points = sample.reduce((sum,row) => sum + ({Win:1,Tie:0.5,Loss:0}[result(row,opponent)] || 0),0);
          return {game,total:sample.length,pct:sample.length ? points / sample.length : null};
        });
        const average = rows => rows.length && rows.every(row => row.pct != null)
          ? rows.reduce((sum,row) => sum+row.pct,0)/rows.length : null;
        const strong = quality.filter(row => row.pct != null && row.pct >= 0.5).map(row => row.game);
        const weak = quality.filter(row => row.pct != null && row.pct < 0.5).map(row => row.game);
        const stats = games.map(game => {
          const values = game.sporting?.efficiency;
          const other = game.keys.find(value => value !== key);
          return values?.[key] && values?.[other] ? [values[key],values[other]] : null;
        }).filter(Boolean);
        const epa = index => {
          const plays = stats.reduce((sum,row) => sum + row[index].plays,0);
          return plays ? stats.reduce((sum,row) => sum + row[index].epa,0)/plays : null;
        };
        const winPct = games.reduce((sum,game)=>sum + ({Win:1,Tie:0.5,Loss:0}[result(game,key)] || 0),0) / games.length;
        data.set(key,{sos:average(quality), recentSos:average(quality.slice(0,5)), winPct, strong:form(strong,key),
          weak:form(weak,key), games:games.length, epa:epa(0), allowedEpa:epa(1), epaGames:stats.length});
      }
      const all = [...data.values()];
      for (const item of all) {
        const rank = field => all.length === 32 && Number.isFinite(item[field]) && all.every(other => Number.isFinite(other[field]))
          ? 1 + all.filter(other => field === "allowedEpa" ? other[field] < item[field]-1e-8 : other[field] > item[field]+1e-8).length
          : null;
        item.sosRank = rank("sos");
        item.winPctRank = rank("winPct");
        if (all.every(row => row.epaGames === row.games)) {
          item.epaRank = rank("epa");
          item.defRank = rank("allowedEpa");
        }
      }
      state.nflContext = data;
    }
    return state.nflContext.get(team) || {};
  }

  const numberWithRank = (value, rank, format = display) => value == null ? "—"
    : format(value) + (rank ? " · #" + rank : "");
  const opponentPct = value => display(value*100) + "%";

  function availability(team, participant = "") {
    const reports = state.fixture.sporting?.availability?.[team];
    if (!reports?.length) return "";
    const timestamp = state.fixture.sporting?.availability_at;
    if (!timestamp || Date.parse(timestamp) > Math.min(Date.parse(state.fixture.start),Date.parse(state.snapshot.generated))
        || Date.parse(state.snapshot.generated) - Date.parse(timestamp) > 3*day) return "";
    const item = row => '<li><strong>' + markup(row.name) + '</strong><span>' + markup([row.status,row.position,row.injury].filter(Boolean).join(" · ")) + '</span></li>';
    return '<section class="ti-panel ti-availability">' + heading(participant ? participant + " availability" : "Reported availability")
      + '<ul>' + reports.slice(0,3).map(item).join("") + '</ul>'
      + (reports.length > 3 ? '<details><summary>View all ' + reports.length + ' reported players</summary><ul>'
        + reports.slice(3).map(item).join("") + '</ul></details>' : "")
            + note("Game-week reports updated " + when(timestamp, et) + " ET. A missed practice does not establish game status.")
      + '</section>';
  }

  function badge(label, tone) {
    return '<span class="ti-result" data-result="' + markup(tone) + '">' + markup(label) + '</span>';
  }

  function marketResult(market, grade, side) {
    if (grade === "push") return "Push";
    if (market === "SPREAD") return grade === "win" ? "Covered" : "Did not cover";
    if (market === "MONEYLINE") return grade === "win" ? "ML win" : "ML loss";
    return (side === "under" ? "Under" : "Over") + (grade === "win" ? " hit" : " missed");
  }

  function recentRows(games, team) {
    return '<div class="ti-list ti-recent-list">' + games.map(game => {
      const other = game.keys.findIndex(key => key !== team);
      const outcome = result(game, team);
      const spread = game.quotes?.SPREAD?.[team];
      return '<div class="ti-score-row"><div><strong>' + markup(
        (game.venue === "neutral" || state.snapshot.sport === "ufc" ? "vs " : (other === 1 ? "at " : "vs ")) + game.names[other])
        + '</strong><small>' + markup(when(game.start)) + ' · ' + markup(game.source === "nflverse" ? "nflverse" : "Verified archive")
        + (game.venue === "neutral" ? ' · Neutral site' : '') + '</small></div>'
        + '<strong class="ti-score-value">' + markup(scoreAgainst(game,team) || outcome) + '</strong>'
        + '<div class="ti-row-results">' + badge(outcome,outcome.toLowerCase())
        + (spread ? badge(marketResult("SPREAD",spread.grade),"ats-" + spread.grade) : '') + '</div></div>';
    }).join("") + '</div>';
  }

  function overview(games, team) {
    const summary = seasonDetails(games, team);
    const {scope,year,last,recent,margin} = summary;
    const context = nflContext(team,scope);
    const fight = state.snapshot.sport === "ufc";
    const scopeLabel = scope.league ? "Independent season data" : fight ? "Verified archive bouts" : "Red Fox archive coverage";
    let html = '<section class="ti-overview"><div class="ti-section-head">' + heading(scope.name) + '<span>'
      + markup(scopeLabel) + '</span></div><div class="ti-facts ti-overview-kpis">'
      + fact(year.record, scope.league ? "Season record" : "Archive record") + fact(recent.record, "Last " + last.length + " record")
      + fact(summary.streak, "Current streak") + (fight ? fact(year.games, "Verified bouts") : fact(signed(margin,1), course() + " margin / game"))
      + '</div></section>';
    if (!scope.league) html += note(fight
      ? "These are verified Red Fox bouts. Complete career and UFC records are unavailable."
      : "Archive coverage varies by team. Results count includes " + year.games + " verified games in this period.");
    if (scope.league && (summary.home.games || summary.away.games)) {
      html += '<div class="ti-quick-stats">' + fact(summary.home.record,"Home · " + summary.home.games)
        + fact(summary.away.record,"Away · " + summary.away.games)
        + (context.sos == null ? '' : fact(numberWithRank(context.sos,context.sosRank,opponentPct),"Opponent win %")) + '</div>';
    }
    html += availability(team);
    html += '<section class="ti-panel ti-recent">' + heading(fight ? "Recent verified bouts" : "Recent games")
      + (last.length ? note(fight ? "Selected fighter result" : "Scores show the selected team first. ATS is graded separately.") + recentRows(last, team)
        : note("No verified results before the selected matchup."));
    if (games.length > 5) {
      html += '<details class="ti-more"><summary>View all ' + games.length + ' results</summary><div data-ti-more></div></details>';
    }
    return html + '</section>';
  }

  function baseballSeason(team) {
    const source = state.snapshot.sporting?.baseball;
    const season = source?.teams?.[team];
    if (!season) return "";
    const record = [season.wins,season.losses,...(season.tie?[season.tie]:[])].join("–");
    const stats = [
      fact(record,"Season record · " + season.games),
      fact(season.home_wins+"–"+season.home_losses,"Designated home"),
      fact(season.away_wins+"–"+season.away_losses,"Designated away"),
      fact(signed((season.scored-season.allowed)/season.games,2),"Run differential / game"),
      fact(display(season.scored/season.games,2),"Runs scored / game"),
      fact(display(season.allowed/season.games,2),"Runs allowed / game")
    ];
    return '<section class="ti-panel">' + heading(source.season+" MLB regular season") + '<div class="ti-facts">'
      + stats.join("") + '</div><p class="ti-note">The information used here was obtained free of charge from and is copyrighted by '
      + '<a href="https://www.retrosheet.org/gamelogs/">Retrosheet</a>. Interested parties may contact Retrosheet at retrosheet.org.</p></section>';
  }

  function teamStats(games, team) {
    const {scope,year,recent,margin,home,away} = seasonDetails(games,team);
    const context = nflContext(team,scope);
    const fight = state.snapshot.sport === "ufc";
    let html = '<section class="ti-panel">' + heading(scope.name);
    if (fight) {
      const profile = state.snapshot.sporting?.fighters?.[team];
      html += '<div class="ti-facts">' + fact(year.record, "Archive record") + fact(year.games, "Verified bouts") + '</div>';
      if (profile?.height_cm) {
        html += '<div class="ti-quick-stats">' + fact(display(profile.height_cm,0) + " cm", "Published height") + '</div>'
          + '<p class="ti-note">Fighter identity and height from <a href="https://www.wikidata.org/wiki/' + markup(profile.item) + '">Wikidata</a>.</p>';
      }
      return html + note("Complete career record, reach, stance and fight-level striking and grappling metrics are unavailable.") + '</section>';
    }
    const scoring = course();
    html += '<div class="ti-facts">'
      + fact(display(year.scored),scoring + " scored / game")
      + fact(display(year.allowed),scoring + " allowed / game")
      + fact(signed(margin,1),scoring + " differential / game")
      + fact(year.games,"Verified games") + '</div></section>';
    html += '<section class="ti-panel">' + heading("Recent form") + '<div class="ti-facts">'
      + fact(recent.record,"Last " + recent.games + " record")
      + fact(display(recent.scored),scoring + " scored / game")
      + fact(display(recent.allowed),scoring + " allowed / game")
      + fact(Number.isFinite(recent.scored) ? signed(recent.scored-recent.allowed,1) : "—",scoring + " differential / game") + '</div></section>';
    if (scope.league) {
      html += '<section class="ti-panel">' + heading("Schedule and opponent quality")
        + '<div class="ti-facts">' + fact(home.record,"Home record") + fact(away.record,"Away record")
        + fact(numberWithRank(context.sos,context.sosRank,opponentPct),"Opponent win %")
        + fact(context.recentSos == null ? "—" : opponentPct(context.recentSos),"Recent opponent win %")
        + fact(numberWithRank(context.winPct,context.winPctRank,opponentPct),"NFL win % rank") + '</div>';
      html += note("Average opponent win percentage entering this matchup. Excludes each opponent’s games against the selected team. Ties count as half a win and repeated opponents count per matchup. Rank 1 indicates the highest opponent win percentage. Home and away records exclude neutral sites.")
        + '<div class="ti-quick-stats">' + fact(context.strong?.record || "—","Against opponents at .500 or better")
        + fact(context.weak?.record || "—","Against opponents below .500") + '</div></section>';
      if (context.epaGames) html += '<section class="ti-panel">' + heading("NFL offensive and defensive efficiency")
        + '<div class="ti-facts">' + fact(numberWithRank(context.epa,context.epaRank,value=>signed(value,3)),"Offense EPA / play · higher")
        + fact(numberWithRank(context.allowedEpa,context.defRank,value=>signed(value,3)),"Defense EPA allowed / play · lower") + '</div>'
        + note(context.epaGames + " matched nflverse box scores. EPA per play uses pass attempts, sacks and rushing attempts.") + '</section>';
    } else html += note("Verified league standings, strength of schedule and opponent-adjusted efficiency are unavailable for this profile.");
    if (state.snapshot.sport === "mlb") html += baseballSeason(team);
    return html;
  }

  function quoteText(market, line, price) {
    if (market === "MONEYLINE") return signed(price);
    const quoteLine = Number.isFinite(line) ? (market === "SPREAD" && line > 0 ? "+" : "") + String(line).replace("-", "−") : "—";
    return quoteLine + (Number.isFinite(price) ? " (" + signed(price) + ")" : "");
  }

  function extremeSplit(...values) {
    return values.some(value => value !== "" && value != null && [0,100].includes(Number(value)));
  }

  const splitNote = () => note("0% and 100% are source-reported endpoints. The archive does not independently verify the underlying wagers.");

  function oppositeSide(market, selected, names) {
    if (market === "TOTAL") return selected.toLowerCase() === "under" ? "Over" : "Under";
    return names.find(name=>keyName(name)!==keyName(selected)) || "the opponent";
  }

  function movementText(market, quote, side, names) {
    const line = quote.line_toward;
    const price = quote.price_toward_implied;
    const other = oppositeSide(market,side,names);
    const output = [];
    if (market !== "MONEYLINE" && Number.isFinite(line)) {
      output.push(!line ? "Line unchanged" : display(Math.abs(line)) + (Math.abs(line) === 1 ? " pt toward " : " pts toward ") + (line>0?side:other));
    }
    if (Number.isFinite(quote.price) && Number.isFinite(quote.open_price)) {
      if (quote.price === quote.open_price) output.push("Price unchanged");
      else if (Number.isFinite(price)) output.push(price>0?"Price shortened on "+side:price<0?"Price eased on "+side:"Equivalent price");
    }
    return output.join(" · ");
  }

  function observationDetails(game, market, side, quote) {
    const evidence = '<small>Observed ' + markup(when(quote.at,et)) + ' ET · Checkpoint ' + markup(when(quote.checkpoint,et)) + ' ET</small>'
      + (quote.path_direction_changes > 0 ? '<small>' + markup(quote.path_direction_changes + " direction changes before checkpoint") + '</small>' : '')
      + (extremeSplit(quote.tickets,quote.money) ? splitNote() : '');
    const identity = game.vendor && /^[A-Za-z0-9_.%+-]{1,90}$/.test(game.vendor);
    return '<details class="ti-evidence" ' + (identity ? 'data-ti-observations data-sport="' + markup(state.snapshot.sport) + '" data-game="' + markup(game.vendor)
      + '" data-market="' + markup(market) + '" data-side="' + markup(side) + '" data-cutoff="' + markup(quote.at)
      + '" data-price="' + markup(quote.price) + '" data-line="' + markup(quote.line) + '"' : '') + '><summary>Quote details</summary>'
      + evidence + (identity ? '<div class="ti-observations"></div>' : '') + '</details>';
  }

  const timelines = new Map();

  async function loadObservations(control) {
    if (control.dataset.loaded) return;
    control.dataset.loaded = "true";
    const host = control.querySelector(".ti-observations");
    const {sport,game,market,side,cutoff,price,line} = control.dataset;
    const id = sport + "--" + game;
    host.textContent = "Checking quote observations…";
    let promise = timelines.get(id);
    if (!promise) {
      const url = "/data/anomaly_event_details/" + encodeURIComponent(id) + ".json";
      promise = (window.redfoxAuthorizedFetch || fetch)(url,{signal:AbortSignal.timeout(10000)},{required:false})
        .then(response=>response.ok?response.json():null).catch(()=>null);
      if (timelines.size >= 3) timelines.delete(timelines.keys().next().value);
      timelines.set(id,promise);
    }
    const payload = await promise;
    if (!payload) {
      timelines.delete(id);
      delete control.dataset.loaded;
    }
    const asNum = value => value === "" || value == null || !Number.isFinite(Number(value)) ? null : Number(value);
    const at = Date.parse(cutoff);
    const identify = row => window.marketSideIdentity(row,{sport,market_display:market});
    const keep = row => String(row.game_id) === game && row.sport === sport && row.market_display === market
      && Date.parse(row.timestamp) <= at && row.line_display;
    const rows = Array.isArray(payload) ? payload.filter(keep) : [];
    const observed = rows.filter(row => Date.parse(row.timestamp)===at);
    const selected = identify({flagged_side:side});
    const expected = observed.find(row => identify(row)===selected && asNum(row.price_odds)===asNum(price)
      && (market==="MONEYLINE"||asNum(row.line_value)===asNum(line)));
    if (!expected || !selected || new Set(observed.map(identify)).size!==2 || observed.length!==2) {
      host.textContent = "The archived timeline could not be linked to this exact quote. The verified checkpoint is shown above.";
      return;
    }
    const sorted = rows.filter(row=>identify(row)===selected)
      .sort((a,b)=>Date.parse(b.timestamp)-Date.parse(a.timestamp));
    host.innerHTML = '<div class="ti-timeline"><table><thead><tr><th scope="col">Observed (ET)</th><th scope="col">Quote</th>'
      + '<th scope="col">Bets</th><th scope="col">Money</th></tr></thead><tbody></tbody></table></div>'
      + '<button type="button" class="ti-older-observations">Show older observations</button>';
    const body = host.querySelector("tbody"), button = host.querySelector("button");
    let visible = 0;
    const append = () => {
      const upcoming = sorted.slice(visible, visible+50);
      body.insertAdjacentHTML("beforeend",upcoming.map(row=>'<tr><td>'+markup(when(row.timestamp,et))+'</td><td>'+markup(row.line_display)
        +'</td><td>'+pct(row.bets_pct)+'</td><td>'+pct(row.money_pct)+'</td></tr>').join(""));
      visible += upcoming.length;
      button.hidden = visible >= sorted.length;
      button.textContent = "Show older observations ? " + (sorted.length-visible) + " remaining";
    };
    button.onclick = append;
    append();
  }

  function currentMove(row, market, mates) {
    const read = value => {
      const text = String(value || "").replaceAll("−","-").trim();
      if (market === "MONEYLINE") return {price:Number(text.match(/([+-]\d+)$/)?.[1])};
      const values = text.match(/([+-]?\d+(?:\.\d+)?)\s*\(([+-]\d+)\)$/);
      return {line:Number(values?.[1]),price:Number(values?.[2])};
    };
    const early = read(row.open_line), later = read(row.current_line);
    if (!Number.isFinite(early.price) || !Number.isFinite(later.price)) return "";
    const implied = odds => odds<0?-odds/(100-odds):100/(100+odds);
    const side = market==="TOTAL" ? (/^u/i.test(row.flagged_side||"")?"Under":"Over")
      : state.fixture.names.find(name=>keyName(row.flagged_side||"").startsWith(keyName(name))) || row.flagged_side;
    const lineMove = Number.isFinite(early.line) && Number.isFinite(later.line)
      ? (later.line-early.line)*(market==="TOTAL"&&side==="Over"?1:-1) : null;
    return movementText(market,{line_toward:lineMove,price_toward_implied:implied(later.price)-implied(early.price),
      open_price:early.price,price:later.price},side,mates);
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

  function selectedMatchup() {
    const {fixture,market} = state;
    const current = fixture.markets?.[market];
    let html = '<section class="ti-panel ti-current">' + heading("Current " + marketLabel(market).toLowerCase());
    if (current?.length) {
      html += '<div class="ti-list">' + current.map(row => {
        const move = currentMove(row,market,fixture.names);
        return '<div class="ti-quote-row ti-current-quote"><strong>' + markup(row.flagged_side)
          + '</strong><span>' + markup(row.open_line || "—") + ' → ' + markup(row.current_line || "—")
          + '</span><small><b>' + pct(row.bets_pct) + ' bets · ' + pct(row.money_pct) + ' money</b>'
          + (move ? ' · ' + markup(move) : '') + '</small></div>';
      }).join("") + '</div>';
      const retained = current.some(row => /retained|stale/i.test(row.quality));
      const stamps = current.map(row=>row.source_latest_at).filter(Boolean).sort();
      const time = stamps.length ? " · " + when(stamps.at(-1),et) + " ET" : "";
      html += note((retained ? "Last retained quote" : current.some(row => row.frozen) ? "Frozen pregame quote" : "Latest quote") + time);
      if (current.some(row => /^split risk$/i.test(row.quality))) {
        html += note("The board flags this capture for split risk. The quotes remain visible for context.");
      } else if (current.some(row => extremeSplit(row.bets_pct,row.money_pct))) html += splitNote();
    } else {
      html += note("Current market quote unavailable.");
    }
    return html + '</section>';
  }

  function historical(games, team) {
    const { market, snapshot, total } = state;
    const subject = market === "TOTAL" ? total : team;
    const sample = games.filter(game => game.quotes?.[market]?.[subject]).slice(0, 12);
    const counts = { win: 0, loss: 0, push: 0 };
    let net = 0;
    for (const game of sample) {
      const quote = game.quotes[market][subject];
      counts[quote.grade]++;
      net += quote.units;
    }
    let html = '<div class="ti-selectors">' + (snapshot.sport === "ufc" ? "" : markets.map(value => control(marketLabel(value), "market", value, value === market)).join(""))
      + (market === "TOTAL" ? '<span class="ti-separator"></span>' + ["over", "under"].map(value => control(value === "over" ? "Over" : "Under", "total", value, value === total)).join("") : "") + '</div>';
    html += selectedMatchup();
    html += heading("Verified pregame market history");
    const official = officialHistory(games, market, team, total);
    if (!sample.length) return html + note("No priced, graded archive quotes for this selection yet. Completed scores may still be available.") + official;
    const tone = net>0?"positive":net<0?"negative":"";
    html += '<div class="ti-facts">' + fact(sample.length, "Priced games")
      + fact([counts.win, counts.loss, counts.push].join("–"), "Wins–losses–pushes")
      + fact(signed(net, 2) + "u", "Net units",tone) + fact(signed(100 * net / sample.length, 1) + "%", "Flat-unit ROI",tone) + '</div>';
    html += note("Up to 12 recent priced results. Each game uses one verified pregame checkpoint, which can precede the closing quote. Net units and ROI use one unit risked per game.");
    html += '<div class="ti-list">' + sample.map(game => {
      const quote = game.quotes[market][subject];
      const from = quoteText(market, quote.open_line, quote.open_price);
      const to = quoteText(market, quote.line, quote.price);
      const primary = market === "TOTAL" ? (total === "over" ? "Over" : "Under") : game.names[game.keys.indexOf(team)];
      const selected = marketResult(market,quote.grade,total);
      const score = market === "TOTAL" ? (game.scores ? game.names[0]+" "+game.scores[0]+" · "+game.names[1]+" "+game.scores[1] : "")
        : scoreAgainst(game,team);
      const move = movementText(market,quote,primary,game.names);
      return '<article class="ti-game-evidence"><div class="ti-history-top"><div><strong>'
        + markup(game.names.join(snapshot.sport === "ufc" ? " vs " : " @ ")) + '</strong><small>'
        + markup(when(game.start) + " · " + primary) + '</small></div>' + badge(selected,quote.grade) + '</div>'
        + '<div class="ti-history-numbers"><div><strong>' + markup(from + " → " + to) + '</strong>'
        + (move ? '<small>' + markup(move) + '</small>' : '') + '</div><div><strong>' + pct(quote.tickets) + ' bets · '
        + pct(quote.money) + ' money</strong>' + (extremeSplit(quote.tickets,quote.money) ? '<small>Source-reported endpoint</small>' : '')
        + (score ? '<small>' + markup("Final "+score+(market!=="TOTAL"?" · selected team first":"")) + '</small>' : '')
        + '</div></div>' + observationDetails(game,market,primary,quote) + '</article>';
    }).join("") + '</div>';
    return html + official;
  }

  function compare() {
    const { fixture, snapshot } = state;
    const sides = fixture.keys.map((key, i) => {
      const games = scoredGames(key);
      const data = seasonDetails(games,key);
      const lastStart = games[0]?.start;
      const days = lastStart ? (Date.parse(fixture.start)-Date.parse(lastStart))/day : null;
      const priorMarket = games.filter(game=>game.quotes?.[state.market]?.[key]).slice(0,12);
      const absolute = priorMarket.reduce((sum,game)=>sum + game.quotes[state.market][key].units,0);
      return {key,name:fixture.names[i],data,context:nflContext(key,data.scope),days,
        market:priorMarket.length?"Unadjusted " + signed(100*absolute/priorMarket.length,1)+"% · "+priorMarket.length+" graded":"—"};
    });
    const fight = snapshot.sport === "ufc";
    const row = (label, values) => '<tr><th scope="row">' + markup(label) + '</th>'
      + values.map(value => '<td>' + markup(value ?? "—") + '</td>').join("") + '</tr>';
    const two = (label, value) => row(label, sides.map(value));
    let html = '<section class="ti-panel ti-matchup-panel"><table class="ti-compare-table"><colgroup><col><col><col></colgroup>'
      + '<thead><tr><th scope="col">Metric</th>' + sides.map(side=>'<th scope="col">'+markup(side.name)+'</th>').join("") + '</tr></thead><tbody>'
      + two("Coverage",s=>s.data.scope.league?"Regular season":"Verified archive")
      + two(fight ? "Verified bout record" : "Record",s=>s.data.year.record+" · "+s.data.year.games)
      + two("Recent record",s=>s.data.recent.record+" · "+s.data.recent.games)
      + two("Streak",s=>s.data.streak);
    if (!fight) {
      html += two(course()+" / game",s=>display(s.data.year.scored))
        + two(course()+" allowed / game",s=>display(s.data.year.allowed))
        + two("Scoring differential",s=>signed(s.data.margin,1));
      if (sides.every(s=>s.data.scope.league)) {
        html += two("Home / away",s=>s.data.home.record+" / "+s.data.away.record)
          + two("Opponent win %",s=>numberWithRank(s.context.sos,s.context.sosRank,opponentPct))
          + two("Recent opponent win %",s=>s.context.recentSos == null ? "—" : opponentPct(s.context.recentSos));
        if (sides.some(s=>s.context.epaGames)) html += two("Offense EPA / play ↑",s=>numberWithRank(s.context.epa,s.context.epaRank,value=>signed(value,3)))
          + two("Defense EPA allowed / play ↓",s=>numberWithRank(s.context.allowedEpa,s.context.defRank,value=>signed(value,3)));
      }
    }
    const baseball = snapshot.sporting?.baseball;
    if (baseball?.teams && sides.every(side=>baseball.teams[side.key])) {
      html += two(baseball.season+" MLB regular season",side=>{
        const t = baseball.teams[side.key];
        return t.wins+"–"+t.losses+(t.tie?"–"+t.tie:"");
      });
    }
    const fighters = snapshot.sporting?.fighters;
    if (fight && fighters && sides.some(side=>fighters[side.key])) {
      html += two("Published height",side=>fighters[side.key] ? display(fighters[side.key].height_cm,0)+" cm" : "—");
    }
    html += two(fight?"Days since verified bout":"Days between starts",s=>s.days==null?"—":display(s.days,1));
    if (state.market !== "TOTAL") html += two(marketLabel(state.market)+" historical ROI",s=>s.market);
    html += '</tbody></table></section>';
    const reported = sides.map(s=>availability(s.key,s.name)).filter(Boolean);
    if (reported.length) html += '<div class="ti-paired-panels">' + reported.join("") + '</div>';
    const compareMarkets = markets.filter(market=>fixture.markets?.[market]?.length);
    if (compareMarkets.length > 1) html += '<div class="ti-selectors" aria-label="Current matchup market">'
      + compareMarkets.map(value=>control(marketLabel(value),"market",value,state.market===value)).join("") + '</div>';
    html += selectedMatchup();
    if (baseball?.teams && sides.every(side=>baseball.teams[side.key])) {
      html += '<p class="ti-note">The information used here was obtained free of charge from and is copyrighted by '
        + '<a href="https://www.retrosheet.org/gamelogs/">Retrosheet</a>. Interested parties may contact Retrosheet at retrosheet.org.</p>';
    }
    const [a,b] = sides;
    const notes = [];
    const advanced = sides.every(s=>s.context.epaGames>=5);
    if (advanced) {
      const gap = a.context.epa - b.context.epa;
      const stronger = gap >= 0 ? a : b, other = gap >= 0 ? b : a;
      if (Math.abs(gap)>=0.03) notes.push(stronger.name+" generated "+display(Math.abs(gap),3)
        +" more expected points per offensive play across "+stronger.context.epaGames+" verified box scores. "
        +other.name+" allowed "+signed(other.context.allowedEpa,3)+" EPA per play across "+other.context.epaGames+" games.");
      if (a.context.sos != null && b.context.sos != null && Math.abs(a.context.sos-b.context.sos)>=0.04) {
        const harder = a.context.sos > b.context.sos ? a : b;
        notes.push(harder.name+" faced the stronger opponent record by the schedule measure. EPA here is unadjusted for opposition.");
      }
    } else if (!fight && sides.every(s=>s.data.recent.games>=5)) {
      const [one,two] = sides.map(s=>s.data.recent.scored-s.data.recent.allowed);
      if (Math.abs(one-two)>=0.2) notes.push(a.name+" averaged a "+signed(one,1)+" scoring margin over five verified games. "
        +b.name+" averaged "+signed(two,1)+". Opponent strength is unavailable for this sample, so the gap describes recent results.");
    }
    if (sides.every(s=>s.days!=null) && Math.abs(a.days-b.days)>=1.5) {
      const rested = a.days>b.days ? a:b;
      notes.push(rested.name+" has "+display(Math.abs(a.days-b.days),1)+" more days between recorded starts. Training, travel and availability are separate factors.");
    }
    const support = notes.length ? notes.map(note).join("") : note(fight
      ? "Verified bout results and time between starts are available. Fighter style and opponent quality are unverified."
      : "There is insufficient verified opponent or efficiency evidence to identify a material matchup difference.");
    return '<section class="ti-panel ti-what-matters">' + heading("What matters in this matchup") + support + '</section>' + html;
  }

  function render(action, value) {
    const { fixture, snapshot, tab, team } = state;
    const key = fixture.keys[team], games = scoredGames(key);
    dialog.querySelector("#ti-title").textContent = tab === "compare" ? "Matchup comparison" : fixture.names[team];
    dialog.querySelector(".ti-matchup").textContent = labels[snapshot.sport] + " · " + fixture.game + " · " + when(fixture.start);
    const tabs = [["Overview", "overview"], [snapshot.sport === "ufc" ? "Fighter stats" : "Team stats", "stats"],
      ["Market history", "markets"], ["Compare matchup", "compare"]];
    let content;
    if (tab === "compare") content = compare();
    else if (tab === "markets") content = historical(games, key);
    else if (tab === "stats") content = teamStats(games,key);
    else content = overview(games, key);
    const source = snapshot.sources.nflverse;
    const sourceLink = source?.at && /^https:\/\/github\.com\/nflverse\//.test(source.url)
      ? ' · <a href="' + markup(source.url) + '" target="_blank" rel="noopener noreferrer">nflverse schedule</a>' : "";
    const stale = Date.now() - Date.parse(snapshot.generated) > 4 * 3600000;
    dialog.querySelector(".ti-body").innerHTML = '<div class="ti-selectors ti-teams" aria-label="Team">' + fixture.names.map((name, i) => control(name, "team", i, team === i && tab !== "compare")).join("")
      + '</div><nav class="ti-selectors ti-tabs" aria-label="Team Intelligence views">' + tabs.map(([label, value]) => control(label, "tab", value, tab === value)).join("")
      + '</nav><div class="ti-content">' + content + '</div><footer class="ti-source">' + (stale ? "Last available snapshot · " : "Snapshot updated · ")
      + markup(when(snapshot.generated, et)) + ' ET · Red Fox verified archive' + sourceLink
      + (snapshot.sporting?.nfl_team_stats?.at ? ' · <a href="https://nflreadr.nflverse.com/reference/load_team_stats.html">nflverse team stats</a>' : '')
      + (snapshot.sporting?.nfl_availability?.at ? ' · <a href="https://nflreadr.nflverse.com/articles/dictionary_injuries.html">nflverse injury reports</a>' : '')
      + '</footer>';
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
    button.setAttribute("aria-expanded", "true");
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
      state = { snapshot, fixture: match, team: Number(identity.team), tab: "overview",
        market: snapshot.sport === "ufc" ? "MONEYLINE" : (markets.includes(identity.market) ? identity.market : "SPREAD"), total: "over" };
      render();
      body.scrollTop = 0;
    } catch (error) {
      if (id === request) body.innerHTML = '<p class="ti-status" role="status">Team Intelligence is temporarily unavailable. The Market Board is still available.</p>';
    }
  }, true);
})();
