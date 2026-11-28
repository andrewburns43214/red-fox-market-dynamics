/* Team Intelligence consumes the separate, read-only archive snapshot on demand. */
(() => {
  "use strict";
  const markets = ["SPREAD", "MONEYLINE", "TOTAL"];
  const sports = new Set(["nfl", "ncaaf", "nba", "ncaab", "mlb", "nhl", "ufc"]);
  const labels = { nfl: "NFL", ncaaf: "CFB", nba: "NBA", ncaab: "NCAAB", mlb: "MLB", nhl: "NHL", ufc: "UFC" };
  const cache = new Map();
  const date = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" });
  const et = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  const shortDate = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
  const dateKey = new Intl.DateTimeFormat("sv-SE", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" });
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
      if (tiAction === "team" || tiAction === "tab") dialog.querySelector(".ti-body").scrollTop = 0;
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
    return {league, games:selected};
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

  function seasonDetails(games, team) {
    const scope = seasonView(games);
    const year = form(scope.games, team);
    const home = scope.games.filter(g => g.venue === "home" && g.keys[1] === team);
    const away = scope.games.filter(g => g.venue === "home" && g.keys[0] === team);
    return {scope, year, home: form(home, team), away: form(away, team)};
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
        const faced = new Set(games.map(game=>game.keys.find(other=>other!==key)));
        const record = [0,0,0];
        for (const opponent of faced) {
          for (const game of opponents.get(opponent) || []) {
            if (game.keys.includes(key)) continue;
            const index = {Win:0,Loss:1,Tie:2}[result(game,opponent)];
            if (index!==undefined) record[index]++;
          }
        }
        const sampleSize = record.reduce((sum,count)=>sum+count,0);
        const stats = games.map(game => {
          const values = game.sporting?.efficiency;
          const other = game.keys.find(value => value !== key);
          return values?.[key] && values?.[other] ? [values[key],values[other]] : null;
        }).filter(Boolean);
        const efficiency = (index, numerator, denominator) => {
          const complete = stats.filter(row=>Number.isFinite(row[index][numerator]) && Number.isFinite(row[index][denominator]));
          const plays = complete.reduce((sum,row) => sum + row[index][denominator],0);
          return plays && complete.length === stats.length
            ? complete.reduce((sum,row) => sum + row[index][numerator],0)/plays : null;
        };
        data.set(key,{opponentRecord:sampleSize?record:null, opponentWinPct:sampleSize?(record[0]+record[2]/2)/sampleSize:null,
          games:games.length, epa:efficiency(0,"epa","plays"),
          allowedEpa:efficiency(1,"epa","plays"), epaGames:stats.length,
          passEpa:efficiency(0,"pass_epa","dropbacks"), passAllowed:efficiency(1,"pass_epa","dropbacks"),
          rushEpa:efficiency(0,"rush_epa","rushes"), rushAllowed:efficiency(1,"rush_epa","rushes"),
          sackRate:efficiency(0,"sacks","dropbacks"), sackMade:efficiency(1,"sacks","dropbacks")});
      }
      const all = [...data.values()];
      for (const item of all) {
        const rank = field => all.length === 32 && Number.isFinite(item[field]) && all.every(other => Number.isFinite(other[field]))
          ? 1 + all.filter(other => field === "allowedEpa" ? other[field] < item[field]-1e-8 : other[field] > item[field]+1e-8).length
          : null;
        if (all.every(row => row.epaGames === row.games)) {
          item.epaRank = rank("epa");
          item.defRank = rank("allowedEpa");
        }
      }
      state.nflContext = data;
    }
    return state.nflContext.get(team) || {};
  }

  const ordinal = value => Number.isInteger(value) && value > 0
    ? value + ([11,12,13].includes(value%100) ? "th" : ({1:"st",2:"nd",3:"rd"}[value%10] || "th")) : "—";
  const numberWithRank = (value, rank, format = display) => Number.isFinite(value)
    ? format(value) + (rank ? " (" + ordinal(rank) + ")" : "") : "—";
  const opponentPct = value => Number.isFinite(value) ? display(value*100) + "%" : "—";

  function availability(team, participant = "") {
    const reports = state.fixture.sporting?.availability?.[team];
    if (!reports?.length) return "";
    const timestamp = state.fixture.sporting?.availability_at;
    if (!timestamp || Date.parse(timestamp) > Math.min(Date.parse(state.fixture.start),Date.parse(state.snapshot.generated))
        || Date.parse(state.snapshot.generated) - Date.parse(timestamp) > 3*day) return "";
    const item = row => '<li><strong>' + markup(row.name) + '</strong><span>' + markup([
      row.status === "Did not practice" ? "Practice · Did not participate" : row.status,
      row.position,row.injury].filter(Boolean).join(" · ")) + '</span></li>';
    return '<section class="ti-panel ti-availability">' + heading(participant ? participant + " availability" : "Reported availability")
      + '<ul>' + reports.slice(0,3).map(item).join("") + '</ul>'
      + (reports.length > 3 ? '<details><summary>View all ' + reports.length + ' reported players</summary><ul>'
        + reports.slice(3).map(item).join("") + '</ul></details>' : "")
      + note("Updated " + when(timestamp, et) + " ET")
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

  function independentSource() {
    const { fixture } = state;
    const source = fixture.sporting?.independent;
    const observed = Date.parse(source?.at), now = Date.now();
    return observed <= now && now - observed <= 7 * day && now < Date.parse(fixture.start) ? source : null;
  }

  function currentFbsRanking() {
    const { fixture } = state;
    const source = fixture.sporting?.cfb_elo;
    const now = Date.now(), observed = Date.parse(source?.at);
    return now < Date.parse(fixture.start) && observed <= now && now - observed <= 2 * day ? source : null;
  }

  function independentFighter(team) {
    return Date.now() < Date.parse(state.fixture.start) ? state.fixture.sporting?.ufc_independent?.[team] : null;
  }

  const identity = value => keyName(value || "").normalize("NFD").replace(/\p{M}/gu,"").replace(/[^\p{L}\p{N}]/gu,"");
  const recordText = record => record?.length ? record.slice(0, record[2] ? 3 : 2).join("–") : "—";
  const fightOutcome = {win:"Win",loss:"Loss",draw:"Draw",no_contest:"No contest"};
  const fighterRecord = (fighter, scope) => {
    const record = fighter?.records?.[scope];
    if (!record) return "—";
    const nc = fighter.records[scope+"_nc"];
    return record.join("–") + (nc ? " · "+nc+" NC" : "");
  };

  function mergedResults(games, key, independent) {
    const fights = state.snapshot.sport === "ufc";
    const results = games.map(game => {
      const side = game.keys.indexOf(key), other = 1 - side;
      return {date:dateKey.format(new Date(game.start)), start:game.start, opponent:game.names[other],
        aliases:new Set([game.names[other],game.keys[other]].map(identity)),
        venue:fights ? "" : game.venue === "neutral" ? "Neutral" : game.venue === "home" ? (side ? "Home" : "Away") : "",
        scores:game.scores?.length === 2 ? [game.scores[side],game.scores[other]] : null,
        outcome:result(game,key), ats:game.quotes?.SPREAD?.[key] || game.ats?.[side]};
    });
    const issues = [], verified = new Set();
    for (const row of independent || []) {
      const date = row.at || row.date;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date||"") || !row.opponent) continue;
      const scores = row.scores?.length === 2 ? row.scores : null;
      const outcome = scores ? (scores[0]>scores[1] ? "Win" : scores[0]<scores[1] ? "Loss" : "Tie") : fightOutcome[row.result];
      if (!outcome) continue;
      const aliases = new Set([row.opponent,row.opponent_key].map(identity).filter(Boolean));
      const venue = ({neutral:"Neutral",home:"Home",away:"Away"})[row.venue] || "";
      const matched = game => [...aliases].some(alias=>game.aliases.has(alias));
      const sameDay = results.filter(g=>g.date===date && (!g.independent || matched(g)));
      const candidates = sameDay.filter(matched);
      const agreement = candidates.filter(g=>!verified.has(g) && g.outcome===outcome && (!scores || g.scores?.every((v,i)=>v===scores[i])));
      if (agreement.length === 1) {
        agreement[0].method = row.method;
        agreement[0].venue ||= venue;
        agreement[0].ats ||= row.ats;
        verified.add(agreement[0]);
      } else if (candidates.length || sameDay.length) {
        // With no shared event ID, an unresolved same-day match cannot safely add a second result.
        const conflict = candidates.length===1 && !verified.has(candidates[0]) && candidates[0].outcome && (
          candidates[0].outcome!==outcome || (scores && candidates[0].scores && !candidates[0].scores.every((v,i)=>v===scores[i]))
        );
        issues.push({date,opponent:row.opponent, conflict, external:scores?.join("–") || outcome,
          archive:conflict ? candidates[0].scores?.join("–") || candidates[0].outcome : ""});
      } else {
        results.push({date, opponent:row.opponent, scores, outcome, method:row.method, ats:row.ats,
          venue, aliases, independent:true});
      }
    }
    return {results:results.sort((a,b)=>b.date.localeCompare(a.date)||(b.start||"").localeCompare(a.start||"")),issues};
  }

  function recentForm(rows, fights) {
    const count = rows.filter(row=>row.outcome).length;
    const wins = rows.filter(row=>row.outcome==="Win").length;
    const losses = rows.filter(row=>row.outcome==="Loss").length;
    const ties = rows.filter(row=>["Tie","Draw"].includes(row.outcome)).length;
    const nc = rows.filter(row=>row.outcome==="No contest").length;
    const scored = rows.filter(row=>row.scores?.every(Number.isFinite));
    const average = index => scored.length ? scored.reduce((sum,row)=>sum+row.scores[index],0)/scored.length : null;
    return {record:count ? [wins,losses,...((ties||fights)?[ties]:[])].join("–")+(nc?" · "+nc+" NC":"") : "—",
      scored:average(0), allowed:average(1)};
  }

  function baseballData(key) {
    const source = state.snapshot.sporting?.baseball;
    const row = source?.teams?.[key];
    if (!row?.games) return null;
    return {season:source.season, at:source.at, games:row.games,
      record:recordText([row.wins,row.losses,row.tie || 0]),
      scored:row.scored/row.games, allowed:row.allowed/row.games,
      sites:{home:recordText([row.home_wins,row.home_losses]),away:recordText([row.away_wins,row.away_losses])}};
  }

  function teamView(key) {
    state.views ||= new Map();
    if (state.views.has(key)) return state.views.get(key);
    const {snapshot,fixture} = state;
    const fights = snapshot.sport === "ufc", games = scoredGames(key);
    const archive = seasonDetails(games,key), nfl = nflContext(key,archive.scope);
    const independent = independentSource(), team = independent?.teams?.[key], fighter = independentFighter(key);
    const {results,issues} = mergedResults(games,key,fights ? fighter?.recent : team?.recent);
    const recent = recentForm(results.slice(0,5),fights);
    const conflict = issues.some(item=>item.conflict), source = conflict ? null : team;
    const standings = source && independent.standings?.[key];
    const baseball = snapshot.sport==="mlb" ? baseballData(key) : null;
    // A complete Retrosheet season can be used as the headline record after that season closes.
    const closedBaseball = baseball?.season===fixture.season && Date.parse(fixture.start)>=Date.UTC(baseball.season,10,1);
    const preferred = source && source.games >= archive.year.games ? source : null;
    const year = closedBaseball ? baseball : preferred || archive.year;
    const official = !!(standings || archive.scope.league || closedBaseball);
    const record = standings ? [standings.wins,standings.losses,standings.otLosses].join("–")
      : year===preferred ? recordText(preferred.record) : year.record;
    const sites = closedBaseball ? baseball.sites : preferred
      ? Object.fromEntries(Object.entries(preferred.sites||{}).map(([k,v])=>[k,recordText(v)]))
      : {home:archive.home.record,away:archive.away.record};
    const first = results[0], streakCount = first && first.outcome && results.findIndex(row=>row.outcome!==first.outcome);
    const streak = first && ["Win","Loss","Tie","Draw"].includes(first.outcome)
      ? first.outcome[0]+(streakCount<0 ? results.length : streakCount) : "—";
    // Day-only provider results cannot establish decimal rest days.
    const days = first ? first.start ? (Date.parse(fixture.start)-Date.parse(first.start))/day
      : (Date.parse(dateKey.format(new Date(fixture.start)))-Date.parse(first.date))/day : null;
    const view = {key, name:fixture.names[fixture.keys.indexOf(key)], games, results, issues, recent, fights, fighter, year, record,
      complete:official, sites, designated: !!(closedBaseball || preferred && !preferred.venue_verified),
      standings, baseball, selectedBaseball:closedBaseball, streak, days, dateOnly:first&&!first.start,
      nfl, source, independent, ranking:currentFbsRanking()?.teams[key], rankingSize:currentFbsRanking()?.field,
      height:fighter?.height_inches ? fighter.height_inches+" in" : snapshot.sporting?.fighters?.[key]?.height_cm
        ? display(snapshot.sporting.fighters[key].height_cm,0)+" cm" : "",
      coverage:conflict ? "Sources disagree on a recent result. Using verified archive totals."
        : standings && year.games<standings.gamesPlayed ? "Scoring sample covers "+year.games+" of "+standings.gamesPlayed+" games"
        : official ? "" : year.games ? "Season summary covers "+year.games+" available games" : "Season results unavailable"};
    state.views.set(key,view);
    return view;
  }

  const margin = row => Number.isFinite(row.scored)&&Number.isFinite(row.allowed) ? row.scored-row.allowed : null;
  const panel = (label, content, classes="") => '<section class="ti-panel '+classes+'">'+heading(label)+content+'</section>';
  const facts = values => '<div class="ti-facts">'+values.filter(([value])=>value!=null&&value!==""&&value!=="—")
    .map(([value,label,tone])=>fact(value,label,tone)).join("")+'</div>';
  const expansion = (label,content) => '<details class="ti-more"><summary>'+markup(label)+'</summary>'+content+'</details>';
  const valueCell = value => '<td>'+markup(value ?? "—")+'</td>';

  function resultsTable(rows, fights) {
    const abbreviated = {Win:"W",Loss:"L",Tie:"T",Draw:"D","No contest":"NC"};
    const headers = ["Date","Opponent",fights?"Method":"Score","Result",...(fights?[]:["ATS"])];
    return '<table class="ti-data-table ti-results-table"><thead><tr>'+headers.map(label=>'<th scope="col">'+label+'</th>').join("")+'</tr></thead><tbody>'
      +rows.map(row=>'<tr>'+valueCell(when(row.date,shortDate))+'<th scope="row">'+markup(row.opponent)
      +(row.venue?' <small>'+markup(row.venue)+'</small>':"")+'</th>'+valueCell(fights ? row.method : row.scores?.join("–"))
      +'<td>'+(row.outcome?badge(abbreviated[row.outcome]||row.outcome,row.outcome.toLowerCase()):"—")+'</td>'
      +(fights?"":'<td>'+(row.ats?.grade
        ? '<span title="'+markup((row.ats.source || "Red Fox pregame quote")+" · "+quoteText("SPREAD",row.ats.line,row.ats.price))+'">'
          +badge(({win:"Cover",loss:"No cover",push:"Push"})[row.ats.grade]||row.ats.grade,"ats-"+row.ats.grade)+'</span>'
          +(row.ats.source?'<small class="ti-ats-line">'+markup(quoteText("SPREAD",row.ats.line))+'</small>':"")
        :'<span class="ti-note">No line</span>')+'</td>')+'</tr>').join("")
      +'</tbody></table>';
  }

  function overview(view) {
    const {key,fights,results,record,year,recent,streak,standings,fighter,ranking} = view;
    const summary = fights ? [[fighterRecord(fighter,"pro_mma"),"Professional MMA"],[fighterRecord(fighter,"ufc"),"UFC record"],
      [recent.record,"Recent form"],[streak,"Current streak"]] : [
      [record,standings?"Season record · W–L–OTL":view.complete?"Season record":"Record"],
      [recent.record,results.length>=5?"Last five":"Recent record"],[streak,"Current streak"],[signed(margin(year),1),course()+" margin / game"],
      [standings?ordinal(standings.rank):ordinal(ranking?.rank),standings?"League standing":"FBS rank"],
      [view.sites.home,view.designated?"Designated home":"Home"],[view.sites.away,view.designated?"Designated away":"Away"]
    ];
    const recentContent = results.length ? (fights ? "" : note("Score is shown with the selected team first."))
      + resultsTable(results.slice(0,5),fights)
      +(results.length>5?expansion("View all results ("+results.length+")",resultsTable(results.slice(5),fights)):"")
      : note(fights?"No verified bouts for this profile.":"No verified results for this profile.");
    return panel(fights?"Fighter snapshot":"Season snapshot",facts(summary)
      +(fights&&!fighter?.records?.pro_mma?note("Career record unavailable."):""),"ti-overview")
      +panel(fights?"Recent fights":"Recent results",recentContent)
      +availability(key);
  }

  const scoring = which => course() + " " + which;
  const metric = (label, value) => [label,value];
  const venueMetrics = [
    metric("Home record",v=>v.designated ? "—" : v.sites.home),
    metric("Away record",v=>v.designated ? "—" : v.sites.away),
    metric("Designated home record",v=>v.designated ? v.sites.home : "—"),
    metric("Designated away record",v=>v.designated ? v.sites.away : "—"),
    metric("Neutral-site record",v=>v.sites.neutral),
  ];
  const opponentRate = v=>v.source?.schedule_strength?.win_pct ?? v.nfl.opponentWinPct;
  const fbsRank = v=>v.ranking ? ordinal(v.ranking.rank)+" of "+v.rankingSize : "—";
  const opponentMetrics = [
    metric("Opponents’ combined record",v=>recordText(v.source?.schedule_strength?.record || v.nfl.opponentRecord)),
    metric("Opponent win percentage",v=>opponentPct(opponentRate(v))),
    metric("Pregame opponent win percentage",v=>v.source?.schedule_strength || v.nfl.opponentRecord
      ? "—" : opponentPct(v.source?.opponent_win_pct))
  ];
  const epa = (value,rank) => numberWithRank(value,rank,n=>signed(n,3));
  const offenseMetrics = [
    metric("Offensive EPA / play",v=>epa(v.nfl.epa,v.nfl.epaRank)),
    metric("Passing EPA / dropback",v=>epa(v.nfl.passEpa)),
    metric("Rushing EPA / rush",v=>epa(v.nfl.rushEpa)),
    metric("Sacks allowed / dropback",v=>opponentPct(v.nfl.sackRate)),
  ];
  const defenseMetrics = [
    metric("Defensive EPA allowed / play",v=>epa(v.nfl.allowedEpa,v.nfl.defRank)),
    metric("Passing EPA allowed / dropback",v=>epa(v.nfl.passAllowed)),
    metric("Rushing EPA allowed / rush",v=>epa(v.nfl.rushAllowed)),
    metric("Sacks made / opponent dropback",v=>opponentPct(v.nfl.sackMade)),
  ];
  const hockeyMetrics = [
    metric("Expected goals for",v=>display(v.standings?.expectedGoalsFor)),
    metric("Expected goals against",v=>display(v.standings?.expectedGoalsAgainst)),
    metric("Expected-goal differential",v=>Number.isFinite(v.standings?.expectedGoalsFor)&&Number.isFinite(v.standings?.expectedGoalsAgainst)
      ? signed(v.standings.expectedGoalsFor-v.standings.expectedGoalsAgainst,1):"—"),
    metric("5-on-5 Corsi",v=>Number.isFinite(v.standings?.corsiPct)?display(v.standings.corsiPct)+"%":"—"),
    metric("5-on-5 Fenwick",v=>Number.isFinite(v.standings?.fenwickPct)?display(v.standings.fenwickPct)+"%":"—"),
    metric("Standings points",v=>v.standings?.points),
    metric("League standing",v=>ordinal(v.standings?.rank)),
  ];
  const fighterMetrics = [
    metric("Height",v=>v.height), metric("Reach",v=>v.fighter?.reach_inches ? v.fighter.reach_inches+" in" : "—"),
    metric("Stance",v=>v.fighter?.stance), metric("Weight class",v=>v.fighter?.weight_class)
  ];

  function tableGroups(headers, groups) {
    const columns = headers.length+1;
    const body = groups.map(([title,rows])=>{
      const cells = rows.filter(([,values])=>values.some(value=>value != null && value!=="" && value!=="—"));
      return cells.length ? '<tbody>'+(title?'<tr class="ti-table-group"><th colspan="'+columns+'" scope="rowgroup">'+markup(title)+'</th></tr>':"")
        +cells.map(([label,values])=>'<tr><th scope="row">'+markup(label)+'</th>'+values.map(valueCell).join("")+'</tr>').join("")+'</tbody>' : "";
    }).join("");
    return body ? '<table class="ti-data-table ti-compare-table"><thead><tr><th scope="col">Metric</th>'
      + headers.map(label=>'<th scope="col">'+markup(label)+'</th>').join("")+'</tr></thead>'+body+'</table>' : "";
  }

  const metricRows = (metrics, views) => metrics.map(([label,read])=>[label,views.map(read)]);
  const restText = v => v.days == null ? "—" : display(v.days,v.dateOnly ? 0 : 1)+" days";
  const retrospective = v => {
    const b = v.baseball;
    if (!b || v.selectedBaseball) return "";
    return expansion(b.season+" regular-season reference",tableGroups(["Season"],[["",[
      ["Record",[b.record]],["Runs scored / game",[display(b.scored)]],["Runs allowed / game",[display(b.allowed)]],
      ["Designated home",[b.sites.home]],["Designated away",[b.sites.away]]
    ]]]));
  };

  function teamStats(view) {
    const {fights,year,recent,nfl} = view;
    if (fights) {
      const table = tableGroups(["Profile"],[["",metricRows([...fighterMetrics,metric("Days since last verified bout",restText)],[view])]]);
      return table ? panel("Fighter profile",table) : note("Verified measurements and fight profile details are unavailable.");
    }
    const scoringRows = [metric(scoring("scored / game"),r=>display(r.scored)),metric(scoring("allowed / game"),r=>display(r.allowed)),
      metric("Scoring margin / game",r=>signed(margin(r),1))];
    const performance = tableGroups(["Season",view.results.length>=5?"Last five":"Recent"],[["",metricRows(scoringRows,[year,recent])]]);
    const schedule = tableGroups(["Season"],[["Venues",metricRows(venueMetrics,[view])],
      ["Strength of schedule",metricRows(opponentMetrics,[view])],["Team ranking",metricRows([metric("FBS rank",fbsRank)],[view])]]);
    const advanced = tableGroups(["Season"],[["Offense",metricRows(offenseMetrics,[view])],
      ["Defense",metricRows(defenseMetrics,[view])],["Hockey",metricRows(hockeyMetrics,[view])]]);
    return (performance?panel("Scoring and recent form",performance):"")
      +(schedule?panel(view.ranking?"Schedule and ranking":"Schedule",schedule):"")
      +(advanced?panel(state.snapshot.sport==="nhl"?"Standings and shot quality":"Team efficiency",advanced
        +(nfl.epaGames?note(nfl.epaGames+" matched box scores"):"")):"")
      +retrospective(view)
      +(!performance&&!schedule&&!advanced&&!view.baseball?note("Team statistics are unavailable for this fixture."):"");
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
      else if (Number.isFinite(price)) {
        const change = price > 0 ? "Price shortened on " : "Price eased on ";
        output.push(price ? change + side + " (" + display(Math.abs(price)*100) + " implied pts)" : "Equivalent price");
      }
    }
    return output.join(" · ");
  }

  function observationDetails(game, market, side, quote) {
    const evidence = '<small>Observed ' + markup(when(quote.at,et)) + ' ET · Checkpoint ' + markup(when(quote.checkpoint,et)) + ' ET</small>'
      + (quote.path_direction_changes > 0 ? '<small>' + markup(quote.path_direction_changes + " direction changes before checkpoint") + '</small>' : '')
      + (extremeSplit(quote.tickets,quote.money) ? splitNote() : '');
    const identity = game.vendor && /^[A-Za-z0-9_.%+-]{1,90}$/.test(game.vendor);
    const quotes = Object.values(game.quotes?.[market] || {});
    const peer = quotes.length===2 ? quotes.find(other=>other!==quote && other.at===quote.at) : null;
    return '<details class="ti-evidence" ' + (identity ? 'data-ti-observations data-sport="' + markup(state.snapshot.sport) + '" data-game="' + markup(game.vendor)
      + '" data-market="' + markup(market) + '" data-side="' + markup(side) + '" data-cutoff="' + markup(quote.at)
      + '" data-price="' + markup(quote.price) + '" data-line="' + markup(quote.line) + '"'
      + (peer ? ' data-peer-price="' + markup(peer.price) + '" data-peer-line="' + markup(peer.line) + '"' : "") : '') + '><summary>Quote details</summary>'
      + evidence + (identity ? '<div class="ti-observations"></div>' : '') + '</details>';
  }

  const timelines = new Map();

  async function loadObservations(control) {
    if (control.dataset.loaded) return;
    control.dataset.loaded = "true";
    const host = control.querySelector(".ti-observations");
    const {sport,game,market,side,cutoff,price,line,peerPrice,peerLine} = control.dataset;
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
    const sameQuote = (row, odds, number) => asNum(odds)!=null && asNum(row.price_odds)===asNum(odds)
      && (market==="MONEYLINE"||(asNum(number)!=null && asNum(row.line_value)===asNum(number)));
    const selected = identify({flagged_side:side});
    let expected = observed.find(row=>identify(row)===selected && sameQuote(row,price,line));
    if (!expected && observed.length===2 && peerPrice && observed.every(row=>row.timestamp===cutoff)) {
      // A verified two-sided checkpoint can link abbreviated team labels when both quotes identify one pairing.
      const paired = observed.filter(row=>sameQuote(row,price,line)
        && observed.some(other=>other!==row && sameQuote(other,peerPrice,peerLine)));
      if (paired.length===1) expected = paired[0];
    }
    if (!expected || !identify(expected) || new Set(observed.map(identify)).size!==2 || observed.length!==2) {
      host.textContent = "The archived timeline could not be linked to this exact quote. The verified checkpoint is shown above.";
      return;
    }
    const sorted = rows.filter(row=>identify(row)===identify(expected))
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
      button.textContent = "Show older observations · " + (sorted.length-visible) + " remaining";
    };
    button.onclick = append;
    append();
  }

  function currentMove(row, market, mates) {
    const read = value => {
      const text = String(value || "").replaceAll("−","-").trim();
      if (market === "MONEYLINE") return {price:Number(text.match(/([+-]\d+)$/)?.[1])};
      const values = text.match(/([+-]?\d+(?:\.\d+)?)(?:\s*\(([+-]\d+)\))?$/);
      return {line:Number(values?.[1]),price:Number(values?.[2])};
    };
    const early = read(row.open_line), later = read(row.current_line);
    const implied = odds => odds<0?-odds/(100-odds):100/(100+odds);
    const side = market==="TOTAL" ? (/^u/i.test(row.flagged_side||"")?"Under":"Over")
      : state.fixture.names.find(name=>keyName(row.flagged_side||"").startsWith(keyName(name))) || row.flagged_side;
    const lineMove = Number.isFinite(early.line) && Number.isFinite(later.line)
      ? (later.line-early.line)*(market==="TOTAL"&&side==="Over"?1:-1) : null;
    const priceMove = Number.isFinite(early.price)&&Number.isFinite(later.price) ? implied(later.price)-implied(early.price) : null;
    return movementText(market,{line_toward:lineMove,price_toward_implied:priceMove,
      open_price:early.price,price:later.price},side,mates);
  }

  function officialHistory(games, market, team, total) {
    const previous = games.flatMap(game => (game.favorites || [])
      .filter(row => row.market === market && row.decision_price && row.side_key === (market === "TOTAL" ? total : team))
      .map(row => ({ game, row })));
    if (!previous.length) return "";
    return expansion("Previous official Favorites",'<div class="ti-list">' + previous.slice(0, 12).map(({game, row}) => {
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
    }).join("") + '</div>' + (previous.length > 12 ? note("Showing the 12 most recent official Favorites.") : ""));
  }

  function marketControls(includeTotal) {
    const {snapshot,fixture,market,total} = state;
    const available = markets.filter(value=>fixture.markets?.[value]?.length || value===market);
    if (!includeTotal && available.length < 2 || snapshot.sport === "ufc") return "";
    return '<div class="ti-selectors ti-market-controls" aria-label="Market selection">'
      + (includeTotal ? markets : available).map(value=>control(marketLabel(value),"market",value,value===market)).join("")
      +(includeTotal&&market==="TOTAL"?'<span class="ti-separator"></span>'+["over","under"].map(value=>
        control(value==="over"?"Over":"Under","total",value,value===total)).join(""):"")+'</div>';
  }

  function selectedMatchup() {
    const {fixture,market} = state;
    const current = fixture.markets?.[market];
    let html = '<section class="ti-panel ti-current">' + heading("Current market")+marketControls(false);
    if (current?.length) {
      html += '<table class="ti-data-table ti-market-table"><thead><tr><th scope="col">Side</th><th scope="col">Open</th>'
        +'<th scope="col">Observed</th><th scope="col">Bets / Money</th></tr></thead><tbody>' + current.map(row => {
        const move = currentMove(row,market,fixture.names);
        return '<tr><th scope="row">'+markup(row.flagged_side)+'</th>'+valueCell(row.open_line || "—")
          +valueCell(row.current_line || "—")+valueCell(pct(row.bets_pct)+" / "+pct(row.money_pct))+'</tr>'
          +(move?'<tr class="ti-market-movement"><td colspan="4">'+markup(move)+'</td></tr>':"");
      }).join("") + '</tbody></table>';
      const retained = current.some(row => /retained|stale/i.test(row.quality));
      const stamps = current.map(row=>row.source_latest_at).filter(Boolean).sort();
      const time = stamps.length ? " · " + when(stamps.at(-1),et) + " ET" : "";
      html += note((retained ? "Last retained quote" : current.some(row => row.frozen) ? "Frozen pregame quote" : "Latest quote") + time);
      if (current.some(row => /^split risk$/i.test(row.quality))) {
        html += note("Split data flagged for review.");
      } else if (current.some(row => extremeSplit(row.bets_pct,row.money_pct))) html += expansion("Split data note",splitNote());
    } else {
      html += note("Current market quote unavailable.");
    }
    return html + '</section>';
  }

  function historical(games, team) {
    const {market,snapshot,total} = state;
    const subject = market === "TOTAL" ? total : team;
    const sample = games.filter(game=>game.quotes?.[market]?.[subject]).slice(0,12);
    const counts = {win:0,loss:0,push:0};
    let net = 0;
    for (const game of sample) {
      const quote = game.quotes[market][subject];
      counts[quote.grade]++;
      net += quote.units;
    }
    const tone = net>0?"positive":net<0?"negative":"";
    const performance = sample.length ? facts([
      [sample.length,"Graded games"],[[counts.win,counts.loss,counts.push].join("–"),"Market record"],
      [signed(net,2)+"u","Net units",tone],[signed(100*net/sample.length,1)+"%","ROI",tone]
    ]) + (sample.length<=5?note("Small sample · Up to 12 recent verified pregame quotes."):"")
      : note("No graded games at a verified pregame price.");
    const rows = sample.map(game => {
      const quote = game.quotes[market][subject];
      const primary = market==="TOTAL" ? (total==="over"?"Over":"Under") : game.names[game.keys.indexOf(team)];
      const opponent = game.names[1-game.keys.indexOf(team)];
      const score = market==="TOTAL" && game.scores
        ? game.names.map((name,i)=>name+" "+game.scores[i]).join(" · ") : scoreAgainst(game,team);
      const move = movementText(market,quote,primary,game.names);
      return '<article class="ti-game-evidence"><div class="ti-history-top"><div><strong>'
        +markup(market==="TOTAL"?game.names.join(" @ "):opponent)+'</strong><small>'+markup(when(game.start)+" · "+(market==="TOTAL"?primary:
          game.venue==="neutral"?"Neutral site":game.venue==="home"?(game.keys[0]===team?"Away":"Home"):"Verified pregame"))+'</small></div>'
        +badge(marketResult(market,quote.grade,total),quote.grade)+'</div>'
        +'<div class="ti-history-numbers"><div><strong>'+markup(quoteText(market,quote.open_line,quote.open_price)
          +" → "+quoteText(market,quote.line,quote.price))+'</strong>'+(move?'<small>'+markup(move)+'</small>':"")
        +'</div><div><strong>'+pct(quote.tickets)+' bets · '+pct(quote.money)+' money</strong>'
        +(score?'<small>'+markup("Final "+score)+'</small>':"")+'</div></div>'
        +observationDetails(game,market,primary,quote)+'</article>';
    }).join("");
    const official = officialHistory(games,market,team,total);
    const gamesContent = rows ? (market!=="TOTAL"&&snapshot.sport!=="ufc"?note("Scores show the selected team first."):"")+rows
      : note("Supporting games will appear when graded quotes are available.");
    return panel("Historical performance",marketControls(true)+performance)+panel("Historical games",gamesContent+official);
  }

  function compare() {
    const {fixture,market} = state;
    const sides = fixture.keys.map(teamView);
    const fights = sides[0].fights;
    const matchSamples = sides.map(v=>v.results.slice(0,5).length);
    const formRows = fights ? [
      metric("Professional MMA record",v=>fighterRecord(v.fighter,"pro_mma")),
      metric("UFC record",v=>fighterRecord(v.fighter,"ufc")),
      metric("Recent form",v=>v.recent.record)
    ] : [
      metric(sides.every(v=>v.complete)?"Season record":"Record",v=>v.record),
      metric(matchSamples.every(n=>n===5)?"Last-five record":"Recent record",v=>v.recent.record),
      metric("Streak",v=>v.streak),
      metric(scoring("scored / game"),v=>display(v.year.scored)),
      metric(scoring("allowed / game"),v=>display(v.year.allowed)),
      metric("Scoring margin / game",v=>signed(margin(v.year),1)),
      metric("FBS rank",fbsRank),
    ];
    const table = tableGroups(sides.map(v=>v.name),[
      [fights?"Record and form":"Season and recent form",metricRows(formRows,sides)],
      ["Strength of schedule",metricRows(opponentMetrics,sides)],
      ["Offense",metricRows(offenseMetrics,sides)],
      ["Defense",metricRows(defenseMetrics,sides)],
      ["Standings and shot quality",metricRows(hockeyMetrics,sides)],
      ["Fighter measurements",metricRows(fighterMetrics,sides)],
      [fights?"Layoff":"Rest",metricRows([metric(fights?"Days since last verified bout":"Days between starts",restText)],sides)],
    ]);
    const reports = sides.map(v=>availability(v.key,v.name)).filter(Boolean);
    const [a,b] = sides, observations = [];
    const add = (label,text) => observations.push('<li><strong>'+markup(label)+'</strong><span>'+markup(text)+'</span></li>');
    const [aRate,bRate] = sides.map(opponentRate);
    if (Number.isFinite(aRate) && Number.isFinite(bRate) && Math.abs(aRate-bRate)>=0.04) {
      const higher = aRate > bRate ? a : b, lower = higher===a ? b : a;
      add("Strength of schedule",higher.name+"’s opponents had a "+opponentPct(opponentRate(higher))+
        " win rate, compared with "+opponentPct(opponentRate(lower))+" for "+lower.name+".");
    }
    if (sides.every(v=>v.nfl.epaGames>=5 && Number.isFinite(v.nfl.epa))) {
      const better = a.nfl.epa>=b.nfl.epa ? a : b, other = better===a ? b : a;
      if (Math.abs(better.nfl.epa-other.nfl.epa)>=0.03 && Number.isFinite(other.nfl.allowedEpa)) {
        add("Efficiency",better.name+" averaged "+signed(better.nfl.epa,3)+" offensive EPA per play. "+other.name+" allowed "+signed(other.nfl.allowedEpa,3)+".");
      }
    } else if (matchSamples.every(n=>n===5) && sides.every(v=>Number.isFinite(margin(v.recent)))) {
      const [one,two] = sides.map(v=>margin(v.recent));
      if (Math.abs(one-two)>=0.2) add("Recent form",a.name+" averaged a "+signed(one,1)+" scoring margin over five games. "+b.name+" averaged "+signed(two,1)+".");
    }
    if (sides.every(v=>Number.isFinite(v.days)) && Math.abs(a.days-b.days)>=1.5) {
      const rested = a.days>b.days ? a : b;
      add(fights?"Layoff":"Rest",rested.name+" has "+display(Math.abs(a.days-b.days),sides.some(v=>v.dateOnly)?0:1)+" more days between starts.");
    }
    const marketRows = fixture.markets?.[market] || [];
    const move = marketRows.filter(row=>!/retained|stale/i.test(row.quality||"")).map(row=>currentMove(row,market,fixture.names))
      .find(value=>/^[\d.]+ pts? toward /.test(value));
    if (move) add("Market movement",move.split(" · ")[0]+".");
    const insights = observations.length ? '<ul class="ti-insights">'+observations.join("")+'</ul>' : "";
    return (table ? panel(fights?"Fighter comparison":"Team comparison",table,"ti-matchup-panel") : "")
      +(insights||reports.length ? panel("What matters in this matchup",insights
        +(reports.length?expansion("Reported availability",'<div class="ti-paired-panels">'+reports.join("")+'</div>'):""),"ti-what-matters"):"")
      +selectedMatchup();
  }

  function sourceFooter(views) {
    const {snapshot,tab} = state;
    const provenance = new Map([
      ["Red Fox archive",snapshot.generated],
      ...views.filter(v=>v.independent).map(v=>[v.independent.provider,v.independent.at]),
      ...views.filter(v=>v.standings && v.independent?.standings_at).map(v=>[v.independent.provider+" standings",v.independent.standings_at]),
      ...views.some(v=>v.fighter)?[["UFCalendar",snapshot.sporting?.ufcalendar?.at]]:[],
      ...views.some(v=>v.ranking)?[["College Football Digest",currentFbsRanking()?.at]]:[],
      ...snapshot.sources.nflverse?.at?[["nflverse schedule",snapshot.sources.nflverse.at]]:[],
      ...snapshot.sporting?.nfl_team_stats?.at?[["nflverse box scores",snapshot.sporting.nfl_team_stats.at]]:[],
      ...snapshot.sporting?.nfl_availability?.at?[["nflverse availability",snapshot.sporting.nfl_availability.at]]:[],
      ...views.filter(v=>v.height && !v.fighter?.height_inches).map(()=>["Wikidata",snapshot.sporting?.wikidata?.at]),
      ...views.filter(v=>v.baseball).map(v=>["Retrosheet "+v.baseball.season,v.baseball.at]),
    ]);
    const list = Array.from(provenance,([name,stamp])=>'<li>'+markup(name)+(stamp?' <small>'+markup(when(stamp,et)+" ET")+'</small>':"")+'</li>').join("");
    const externalAts = new Set(views.flatMap(v=>v.results.map(row=>row.ats?.source).filter(Boolean)));
    const issues = views.flatMap(v=>v.issues.map(issue=>v.name+" vs "+issue.opponent+" · "+when(issue.date,shortDate)
      +(issue.conflict ? " · Archive "+issue.archive+" / provider "+issue.external : " · Same-day fixture could not be matched")));
    const coverage = views.filter(v=>v.coverage).map(v=>note(v.name+" · "+v.coverage)).join("");
    const epaSamples = views.filter(v=>v.nfl.epaGames).map(v=>v.name+" "+v.nfl.epaGames);
    const details = '<ul class="ti-source-list">'+list+'</ul>'+coverage
      +(views.some(v=>v.ranking)?'<p class="ti-note"><a href="https://collegefootballdigest.com/developers#what-we-serve">College Football Digest</a> supplies the FBS power ranking using CollegeFootballData.com results. <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></p>':"")
      +(snapshot.sport==="nfl"?note("EPA means expected points added. Dropbacks use pass attempts plus sacks. EPA uses matched box scores without adjusting for opponents."
        + (epaSamples.length ? " Matched box scores: "+epaSamples.join(", ")+ "." : "")):"")
      +(snapshot.sport==="ufc"?note("Professional and UFC records have different scopes. Days between starts are approximate for date-only results."):"")
      +(views.some(v=>v.source)?note("Recent results merge only verified date, opponent and outcome matches."):"")
      +(views.some(v=>v.source?.schedule_strength?.record || v.nfl.opponentRecord)
        ?note("Strength of schedule counts each opponent once. Opponent records exclude games against the selected team and include only results available before this matchup. Ties count as half a win."):"")
      +(externalAts.size?note("ATS uses the Red Fox pregame quote where available. Closing-line fallbacks are labeled at the result and sourced from "+[...externalAts].join(", ")+"."):"")
      +(issues.length?'<p class="ti-note">Result checks</p><ul class="ti-source-list">'+[...new Set(issues)].map(issue=>'<li>'+markup(issue)+'</li>').join("")+'</ul>':"")
      +note("Historical ROI uses one unit risked per graded game at its verified pregame checkpoint. The checkpoint may precede the closing quote.");
    const age = Date.now()-Date.parse(snapshot.generated);
    const stale = age > 4*3600000 || views.some(v=>v.independent && Date.now()-Date.parse(v.independent.at)>2*day);
    const credit = views.some(v=>v.selectedBaseball || (tab==="stats" && v.baseball)) ? '<p class="ti-attribution">The information used here was obtained free of charge from and is copyrighted by '
      +'<a href="https://www.retrosheet.org/gamelogs/">Retrosheet</a>. Interested parties may contact Retrosheet at www.retrosheet.org.</p>' : "";
    return '<footer class="ti-source">'+(stale?note("Last available snapshot · "+when(snapshot.generated,et)+" ET"):"")
      +credit+expansion("Sources and methodology",details)+'</footer>';
  }

  function render(action, value) {
    const { fixture, snapshot, tab, team } = state;
    const view = teamView(fixture.keys[team]);
    dialog.querySelector("#ti-title").textContent = tab === "compare" ? "Matchup comparison" : fixture.names[team];
    dialog.querySelector(".ti-matchup").textContent = labels[snapshot.sport] + " · " + fixture.game + " · " + when(fixture.start);
    const tabs = [["Overview", "overview"], [snapshot.sport === "ufc" ? "Fighter stats" : "Team stats", "stats"],
      ["Market history", "markets"], ["Compare matchup", "compare"]];
    let content;
    if (tab === "compare") content = compare();
    else if (tab === "markets") content = historical(view.games, view.key);
    else if (tab === "stats") content = teamStats(view);
    else content = overview(view);
    dialog.querySelector(".ti-body").innerHTML = '<div class="ti-selectors ti-teams" aria-label="Team">' + fixture.names.map((name, i) => control(name, "team", i, team === i && tab !== "compare")).join("")
      + '</div><nav class="ti-selectors ti-tabs" aria-label="Team Intelligence views">' + tabs.map(([label, value]) => control(label, "tab", value, tab === value)).join("")
      + '</nav><div class="ti-content">' + content + '</div>' + sourceFooter(tab==="compare"?fixture.keys.map(teamView):[view]);
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
