(() => {
  "use strict";
  const base = "/data/historical_comparisons/";
  const labels = {SPREAD: "Spread", MONEYLINE: "Moneyline", TOTAL: "Total"};
  const sports = {nfl: "NFL", ncaaf: "CFB", nba: "NBA", ncaab: "NCAAB", nhl: "NHL", mlb: "MLB", ufc: "UFC"};
  const cohorts = {closest: "Closest same sport", broader: "Broader same sport", cross: "Cross-sport context"};
  const icon = '<svg width="17" height="17" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><rect x="3" y="11" width="3.4" height="5.5" rx=".65"/><rect x="8.3" y="7.5" width="3.4" height="9" rx=".65"/><rect x="13.6" y="3.4" width="3.4" height="13.1" rx=".65"/></svg>';
  const clockIcon = '<svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="10" cy="10" r="7"/><path d="M10 5.5v4.8l3 1.7" stroke-linecap="round"/></svg>';
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
  let manifest = {}, availability = {}, pending = {}, payloads = {}, frozenGames = new Set();
  let manifestHash = "", manifestState = "", active = null, request = 0, lastButton = null;
  const dialog = () => document.getElementById("historical-comparison");
  const authorized = (url) => window.redfoxAuthorizedFetch(url, {cache: "no-store"});
  const defaultCohort = report => Object.keys(cohorts).find(group => report[group]?.graded > 0) || "closest";

  async function refreshManifest() {
    let current = {};
    try {
      const response = await window.redfoxAuthorizedFetch(base + "index.json", {cache: "no-store"}, {required: false});
      if (response.ok) current = await response.json();
    } catch (_) {
      // The normal board is independent of the optional research endpoint.
    }
    const fresh = Date.now() - Date.parse(current.generated) < 40 * 60000;
    manifestState = current.enabled === false ? "disabled" : current.enabled && fresh ? "ready" : "failed";
    const ready = manifestState === "ready";
    manifest = ready ? current.sides || {} : {};
    availability = ready ? current.availability || {} : {};
    pending = ready ? current.pending || {} : {};
    payloads = ready ? current.payloads || {} : {};
    frozenGames = new Set(ready ? current.frozen_games : []);
    const hash = JSON.stringify([manifestState, Object.entries(manifest).sort(), Object.entries(pending).sort(),
      Object.entries(availability).sort(), Object.entries(payloads).sort(), [...frozenGames].sort()]);
    if (hash !== manifestHash && !dialog()?.open) {
      manifestHash = hash;
      if (document.body.classList.contains("live-recent-active")) window.renderLiveRecent();
      else window.renderBoard?.(false);
    }
  }

  function status(row, index) {
    const key = gameKey(row);
    if (manifestState === "failed") return "Historical comparison updates are temporarily unavailable.";
    if (manifestState !== "ready") return "";
    const started = row.sport !== "ufc" && Date.parse(row.kickoff_iso) <= Date.now();
    if (started && !frozenGames.has(key)) return "Verified pregame comparison for this market is unavailable.";
    const state = availability[key]?.[row.market_display]?.[index] || "source_limited";
    return {
      processing: "Historical comparisons for this market are waiting to be processed.",
      backfill: "Historical captures for this decision window are still being indexed.",
      source_limited: started ? "Verified pregame comparison for this market is unavailable."
        : "Source coverage is insufficient to verify a comparable market setup.",
      unverified: "Similar setups exist, but none has both a verified outcome and entry price.",
      insufficient: "No comparable historical games met this setup’s criteria."
    }[state] || "";
  }

  function available(row) {
    const key = gameKey(row);
    return Boolean(payloads[key] && manifest[key]?.[row.market_display]?.some(Boolean));
  }

  function button(row, index, name) {
    const key = gameKey(row);
    if (row.sport !== "ufc" && Date.parse(row.kickoff_iso) <= Date.now() && !frozenGames.has(key)) return "";
    if (!payloads[key] || !manifest[key]?.[row.market_display]?.[index]) {
      return pending[key]?.includes(row.market_display)
        ? '<span class="historic-placeholder" title="Historical comparisons are processing" role="img" aria-label="Historical comparisons are processing">'+clockIcon+'</span>'
        : "";
    }
    return `<button type="button" class="historic-launch" title="Historical Comparisons"
      aria-haspopup="dialog" aria-label="Historical Comparisons for ${escape(name)}"
      data-historic-game="${escape(key)}" data-historic-payload="${escape(payloads[key])}" data-historic-market="${escape(row.market_display)}"
      data-historic-lane="${index}" data-historic-observation="${escape(row.source_latest_at)}"
      onclick="event.stopPropagation();window.redfoxHistory.open(this)"><span class="historic-launch-mark">${icon}</span></button>`;
  }

  function liveButton(row) {
    if (!available(row) || !frozenGames.has(gameKey(row))) return "";
    const name = row.flagged_side;
    const side = window.marketSideIdentity(row, row);
    const lanes = row.market_display === "TOTAL" ? ["OVER", "UNDER"] :
      String(row.game).split(/ @ |\s+vs\.?\s+/i).map(team => team.trim().toLowerCase());
    const index = lanes.indexOf(side);
    return index < 0 || !manifest[gameKey(row)]?.[row.market_display]?.[index] ? "" : button(row, index, name);
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
      const file = button.dataset.historicPayload;
      if (!/^snapshot--[a-f0-9]{24}$/.test(file)) throw new Error();
      const response = await authorized(base + file + ".json");
      if (!response.ok) throw new Error();
      const payload = await response.json();
      if (!modal.open || token !== request) return;
      const market = button.dataset.historicMarket;
      const index = Number(button.dataset.historicLane);
      if (payload.sport+"--"+payload.event !== button.dataset.historicGame ||
          !payload.markets?.[market]?.sides?.[index]?.comparison) throw new Error();
      active = {payload, market, team: market === "TOTAL" ? 0 : index, total: market === "TOTAL" ? index : 0,
        boardObservation: button.dataset.historicObservation, boardMarket: market,
        cohort: defaultCohort(payload.markets[market].sides[index].comparison)};
      render();
    } catch (_) {
      if (modal.open && token === request) modal.querySelector("[data-historic-body]").innerHTML =
        '<p class="historic-empty" role="status">The published comparison could not be loaded. Market Journey is available from the board.</p>';
    }
  }


  const tone = n => n > 0 ? "positive" : n < 0 ? "negative" : "";
  const opening = n => n > 0 && n < 100 ? pct(n) : "—";
  const quote = (m,s,open) => line(m,open?s.open_line:s.line,open?s.open_price:s.price);
  const path = q => ["retraced","recovered"].includes(q.trajectory) ? " · "+q.trajectory : "";

  function quotePath(m,q) {
    return [quote(m,q,true), ...(q.path_turns||[]).map(turn=>quote(m,turn)), quote(m,q)].join(" → ");
  }

  function movement(m,q) {
    const n=q.line_toward, p=q.price_toward_implied;
    const dir=x=>x>0?"toward":"against", amount=x=>(Math.abs(x)*100).toFixed(1)+" implied pts";
    if(m==="MONEYLINE") return p==null?"Price movement unavailable":Math.abs(p)<.0005?"Moneyline held":"Moneyline "+dir(p)+" · "+amount(p);
    let label=n==null?"Line movement unavailable":Math.abs(n)<.001?"Line held":m==="TOTAL"?
      "Total "+((q.side==="under"?-n:n)>0?"up ":"down ")+Math.abs(n)+" pts":Math.abs(n)+" pts "+dir(n);
    return p!=null&&Math.abs(p)>=.0005?label+" · Juice "+dir(p)+" "+amount(p):label;
  }

  function peers(side) {
    const entries=Object.entries(side.game_context||{});
    return entries.length?entries:side.cross_market?[[side.cross_market.market,side.cross_market]]:[];
  }

  function compactPeer(side, market) {
    const entries=peers(side), peer=entries.find(([m])=>m===(market==="MONEYLINE"?"SPREAD":"MONEYLINE"))||entries[0];
    if(!peer) return "Cross-market quote unavailable";
    const [m,q]=peer;
    const role=market==="TOTAL"&&side.context_reference?" · "+side.context_reference:"";
    const desc=q.sustained_toward===true?"Support held":movement(m,q).replace(/^(Moneyline|Total) /,"");
    return labels[m]+role+" "+(q.captured_at?quotePath(m,q)+" · ":"")+desc;
  }

  function peerDetails(side) {
    const entries=peers(side);
    return entries.length?'<ul class="historic-peer-list">'+entries.map(([m,s])=>{
      const role=m==="TOTAL"?" · Over":side.context_reference?" · "+side.context_reference:"";
      const prices=s.captured_at?" · "+quotePath(m,s)+" · "+pct(s.tickets)+" tickets / "+pct(s.money)+" money · "+when(s.captured_at):"";
      return `<li><strong>${escape(labels[m]+role)}</strong> · ${escape(movement(m,s)+path(s)+prices)}</li>`;
    }).join("")+"</ul>":'<p class="historic-note">No aligned secondary quote for this capture.</p>';
  }

  function tile(label,value, sub="", color="", extra="") {
    return `<div class="historic-metric ${extra}"><span>${label}</span><strong class="${color}">${escape(value)}</strong>${sub?"<small>"+escape(sub)+"</small>":""}</div>`;
  }

  function performance(result, report) {
    const notice=!result?.graded?(report.reason?"Source coverage":report.pending?"Backfill pending":
      result?.matched?"Unverified results":!result&&active.cohort==="cross"?"Unavailable":"Insufficient sample")
      :result.status==="limited"?"Limited sample":"";
    const reason=report.reason||(!result?"Cross-sport context is unavailable for this market.":
      !result.graded?(report.pending?"Historical captures are still being indexed for this decision window.":
        result.matched?"Matched quotes need a verified outcome and entry price to count.":"No comparable games in this group."):
        active.cohort!=="closest"&&!report.closest?.graded?
          "No priced, graded closest matches. Showing "+cohorts[active.cohort].toLowerCase()+"." : "");
    return `<section class="historic-performance" aria-label="Historical performance">
      <div class="historic-performance-head"><label><span>Comparison group</span><select data-historic-cohort aria-label="Comparison group">
        ${Object.entries(cohorts).map(([key,name])=>`<option value="${key}"${active.cohort===key?" selected":""}>${name}</option>`).join("")}
      </select></label>${notice?`<span class="historic-status">${notice}</span>`:""}</div>
      <div class="historic-metrics">${tile("Record",result?.graded?record(result):"—",(result?.graded||0)+" graded")}
        ${tile("Win rate",pct(result?.win_rate))}
        ${tile("ROI",result?.roi==null?"—":signed(result.roi)+"%",result?.units==null?"":signed(result.units)+" units",tone(result?.roi))}
        ${tile("Avg. entry odds",odds(result?.odds),"","","historic-odds")}</div>
      ${reason?`<p class="historic-note">${escape(reason)}</p>`:""}</section>`;
  }

  function setupPanel(setup,side,market,frozen,stale) {
    const opposite=setup.sides.find(s=>s.index!==side.index);
    return `<section class="historic-setup"><h3>${frozen?"Frozen pregame setup":stale?"Last verified market setup":"Current market setup"}</h3>
      <dl><div><dt>Tickets</dt><dd>${pct(side.tickets)}</dd></div><div><dt>Money</dt><dd>${pct(side.money)}</dd></div>
      <div><dt>${market==="MONEYLINE"?"Price":"Line &amp; odds"}</dt><dd>${escape(quotePath(market,side))}</dd>
      <small>${escape(movement(market,side))}</small></div></dl>
      <p class="historic-setup-foot">${opposite?.tickets!=null&&opposite?.money!=null?
        `<span><strong>${escape(opposite.name)}</strong> ${pct(opposite.tickets)} / ${pct(opposite.money)}</span>`:""}
        <span>${escape(compactPeer(side,market))}</span></p></section>`;
  }

  function factors(result) {
    const matched=result?.matches||[], priorities=[/retrac|pressure|held|toward the team|opposite directions|confirmation/i,/ticket share|money share/i,/move size|move direction/i,/entry price|entry line/i];
    const keys=[...new Set([...priorities.map(test=>matched.find(f=>test.test(f))).filter(Boolean),...matched])].slice(0,4);
    return `<details class="historic-factors"${matchMedia("(max-width: 700px)").matches?"":" open"}><summary>Key matching factors</summary>
      ${keys.length?"<ul>"+keys.map(f=>`<li><span aria-hidden="true">✓</span>${escape(f)}</li>`).join("")+"</ul>":
        '<p class="historic-note">No consistent factors for this group.</p>'}</details>`;
  }

  function exampleRow(item,market,history=false) {
    const grade={win:market==="SPREAD"?"Covered":"Won",loss:market==="SPREAD"?"Did not cover":"Lost",push:"Push"}[item.grade]||item.grade;
    return `<span class="historic-example-date">${escape(when(item.date,false))}</span>
      <span class="historic-example-team"><strong>${escape(item.game)}</strong><small>Graded ${escape(item.selection||item.side.side)}</small></span>
      <span class="historic-example-quote">${escape(history?quotePath(market,item.side):quote(market,item.side))}
        <small>${pct(item.side.tickets)} tickets / ${pct(item.side.money)} money${history?" · "+escape(movement(market,item.side)+path(item.side)):""}</small>
        ${history?`<small class="historic-example-peer">${escape(compactPeer(item.side,market))}</small>`:""}</span>
      <span class="historic-result ${item.grade==="win"?"positive":item.grade==="loss"?"negative":""}">${escape(grade)}</span>`;
  }

  function examples(result,market) {
    const rows=result?.examples||[], list=result?.match_list||[];
    if(!rows.length)return "";
    return `<section class="historic-examples"><h3>${active.cohort==="closest"?"Closest verified examples":"Verified examples"}</h3>
      ${rows.map(item=>`<details class="historic-example"><summary>${exampleRow(item,market,true)}</summary>
        <div class="historic-example-detail"><p>${escape(item.score||"Verified settlement")} · Captured ${escape(when(item.entry))}
        ${item.decision?" · Checkpoint "+escape(when(item.decision)):""}</p>
        <p>Open to observed ${escape(quotePath(market,item.side))} · ${escape(movement(market,item.side)+path(item.side))}</p>
        ${peerDetails(item.side)}
        ${item.matched.length?`<p><strong>Matched</strong> ${escape(item.matched.join(", "))}.</p>`:""}
        ${item.differences.length?`<p><strong>Varied</strong> ${escape(item.differences.join(", "))}.</p>`:""}</div></details>`).join("")}
      ${list.length>rows.length?`<details class="historic-all"><summary>${list.length===result.graded?
        "View all "+list.length+" graded matches":"View first "+list.length+" of "+result.graded+" graded matches"}</summary>
        <ul>${list.map(([date,game,selection,line,price,tickets,money,grade])=>
          `<li>${exampleRow({date,game,selection,side:{line,price,tickets,money},grade},market)}</li>`).join("")}</ul></details>`:""}</section>`;
  }

  function control(label,value,group,on) {
    return `<button type="button" data-historic-action="${group}" data-value="${escape(value)}" aria-pressed="${on}">${escape(label)}</button>`;
  }

  function support(setup,side,result,market,payload,markets) {
    const source=side.context||{};
    const audits={ticket_only:"Recorded ticket-only breach",money_only:"Recorded money-only breach",both:"Recorded dual split breach"};
    return `<details class="historic-support"><summary>Show details</summary><div class="historic-support-body">
      <h3>Market evidence</h3>
      <p>${escape(side.name)} · ${escape(movement(market,side)+path(side))} · ${escape(minuteText(setup.lead))} to scheduled start.
        ${side.observations>1?escape(side.observations+" paired observations."):""}</p>
      <p>Opening splits ${opening(side.opening_tickets)} / ${opening(side.opening_money)} → ${pct(side.tickets)} / ${pct(side.money)}.
        ${source.data_badge?"Data quality "+escape(source.data_badge)+".":""}</p>
      <p>${[source.reaction,source.path,source.evidence_role,audits[side.split_audit?.split_breach],source.context_chips].filter(Boolean).map(escape).join(" · ")}</p>
      ${peerDetails(side)}
      <h3>Matching and coverage</h3>
      <p>${active.cohort==="broader"?"Broader matches relax cross-market agreement and price and split tolerances. This group includes closest matches.":
        active.cohort==="cross"?"Cross-sport context compares normalized moneyline prices and movement within football or basketball.":
        "Closest matches prioritize the selected market’s direction, split positioning, entry quote and decision timing."}</p>
      ${payload.frozen && !payload.context_version ? `<p>This snapshot keeps its original frozen comparison. Available supporting market evidence varies by capture.</p>` : `<p>Secondary quotes must be at or before the selected capture and within 20 minutes. Missing context is unscored. Totals use the opening favorite for team-market context.</p>`}
      ${side.comparison.since?`<p>Coverage for this market and checkpoint since ${escape(side.comparison.since)}.</p>`:""}
      ${result?`<p>${result.matched} matched · ${result.graded} priced and graded · ${result.unresolved} unresolved · ${result.missing_price} missing prices · ${result.void} void.
        ${result.line!=null?"Average entry line "+escape(result.line)+".":""}</p>
        ${result.matches?.length?`<p><strong>Matched</strong> ${escape(result.matches.join(", "))}.</p>`:""}
        ${result.differences?.length?`<p><strong>Varied</strong> ${escape(result.differences.join(", "))}.</p>`:""}`:""}
      <p>Observed ${escape(when(setup.observed))} · ${payload.frozen?"Frozen before start":"Results checked through "+escape(when(setup.verified_by||payload.generated))}.</p>
      <section class="historic-by-market"><h3>Similar results by market</h3><div>${markets.map(([key,label])=>{
        const item=payload.markets[key].sides[key==="TOTAL"?active.total:active.team], group=defaultCohort(item.comparison), m=item.comparison[group];
        return `<button type="button" data-historic-action="market" data-value="${key}" aria-pressed="${key===market}">
          <strong>${escape(label+(key==="TOTAL"?" · "+item.name:""))}</strong>
          <span>${m?.graded?escape(record(m)+" · "+m.graded+" graded · "+signed(m.roi)+"% ROI"+(group==="closest"?"":" · "+cohorts[group])):"No graded matches"}</span></button>`;
      }).join("")}</div></section>
      <p class="historic-note">Win rate excludes pushes. ROI and units assume one unit risked per priced, graded match. Pushes return the stake. Voids, missing prices and unverified outcomes are excluded.</p>
    </div></details>`;
  }

  function render() {
    const {payload,market,team,total,cohort}=active, modal=dialog(), setup=payload.markets[market], index=market==="TOTAL"?total:team;
    const side=setup.sides[index], result=side.comparison[cohort], body=modal.querySelector("[data-historic-body]");
    const focus=modal.contains(document.activeElement)?document.activeElement:null;
    modal.querySelector("[data-historic-context]").textContent=`${sports[payload.sport]} · ${payload.game}`;
    modal.querySelector("[data-historic-favorite]").innerHTML=side.favorite?'<img src="/assets/red-fox-favorite.png" alt="Red Fox Favorite">':"";
    const delayedFeed=!payload.frozen&&Date.now()-Date.parse(setup.observed)>35*60000;
    const stale=delayedFeed||(!payload.frozen&&market===active.boardMarket&&Date.parse(active.boardObservation)>Date.parse(setup.observed));
    const markets=Object.entries(labels).filter(([key])=>payload.markets[key]);
    body.innerHTML=`<div class="historic-controls">
      <div><span>${market==="TOTAL"?"Outcome":payload.sport==="ufc"?"Fighter":"Team"}</span><div class="historic-options">${setup.sides.map((s,i)=>control(s.name,i,"side",i===index)).join("")}</div></div>
      <div><span>Market</span><div class="historic-options">${markets.map(([key,label])=>control(label,key,"market",key===market)).join("")}</div></div></div>
      ${performance(result,side.comparison)}
      ${stale?`<p class="historic-notice">${delayedFeed?"Board feed delayed. These comparisons use the last verified pregame capture shown on the board.":"Latest processed capture · a newer board observation may be pending."}</p>`:""}
      <div class="historic-setup-grid">${setupPanel(setup,side,market,payload.frozen,stale)}${factors(result)}</div>
      ${examples(result,market)}
      <div class="historic-footer"><p>Includes Favorites and non-Favorites · ${payload.frozen?"Frozen":"Observed"} ${escape(when(setup.observed))}</p>
        ${support(setup,side,result,market,payload,markets)}</div>`;
    if(focus?.hasAttribute("data-historic-cohort"))body.querySelector("[data-historic-cohort]").focus({preventScroll:true});
    else if(focus?.dataset.historicAction) {
      [...body.querySelectorAll("[data-historic-action]")].find(el=>el.dataset.historicAction===focus.dataset.historicAction&&el.dataset.value===focus.dataset.value)?.focus({preventScroll:true});
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
      } else active[active.market === "TOTAL" ? "total" : "team"] = Number(value);
      const side=active.payload.markets[active.market].sides[active.market==="TOTAL"?active.total:active.team];
      active.cohort=defaultCohort(side.comparison);
      render();
    });
    modal.addEventListener("change", event => {
      if (!active || !event.target.matches("[data-historic-cohort]")) return;
      active.cohort = event.target.value;
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

  window.redfoxHistory = {available, button, liveButton, open, status};
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})();
