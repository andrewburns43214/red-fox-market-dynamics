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
        cohort: defaultCohort(payload.markets[market].sides[index].comparison), ledgers: new Map(),
        subset: "all", page: 0};
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
    return `<section class="historic-examples"><h3>Archived comparison examples</h3>
      <p class="historic-note">This publication retained ${list.length} of ${result.graded} graded rows and detailed evidence for ${rows.length}. Its original record is preserved.</p>
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

  const implied = value => value == null || Math.abs(value) < 100 ? null :
    value < 0 ? -value/(100-value) : 100/(100+value);
  const moveState = (market, quote) => {
    const value=quote?.[market==="MONEYLINE"?"price_toward_implied":"line_toward"];
    const floor=market==="MONEYLINE"?.005:.25;
    return value==null?"Unknown":Math.abs(value)<floor?"Held":value>0?"Toward":"Against";
  };
  const roleName = value => value==null?"Unknown":value < 0?"Favorite":value>0?"Underdog":"Pick’em";
  function roles(m, side, other, peers) {
    if(m==="TOTAL") return [roleName(peers?.SPREAD?.open_line),roleName(peers?.SPREAD?.line)];
    if(m==="SPREAD") return [roleName(side.open_line),roleName(side.line)];
    return ["open_price","price"].map(field=>{
      const a=implied(side[field]), b=implied(other?.[field]);
      return a==null||b==null?"Unknown":roleName(b-a);
    });
  }
  function crossState(m, side, peers) {
    const a=moveState("SPREAD",m==="SPREAD"?side:peers?.SPREAD);
    const b=moveState("MONEYLINE",m==="MONEYLINE"?side:peers?.MONEYLINE);
    return [a,b].includes("Unknown")?"Unavailable":[a,b].includes("Held")?"Neutral":a===b?"Agreement":"Disagreement";
  }
  function totalState(m, side, peers) {
    const x=m==="TOTAL"?side:peers?.TOTAL, move=moveState("TOTAL",x);
    if(move==="Unknown" || move==="Held") return move;
    return (move==="Toward")===(x.side!=="under")?"Rising":"Falling";
  }
  function keyTransitions(sport,m,q) {
    if(m!=="SPREAD" || q.open_line==null || q.line==null || q.open_line===q.line) return [];
    const keys=["nfl","ncaaf"].includes(sport)?[3,7,10,14]:["nhl","mlb"].includes(sport)?[1.5]:[];
    return keys.flatMap(k=>[k,-k].flatMap(n=> q.open_line===n?["Off "+signed(n)]:
      q.line===n?["Onto "+signed(n)]:Math.min(q.open_line,q.line)<n&&n<Math.max(q.open_line,q.line)?["Through "+signed(n)]:[]));
  }
  function profile(sport,m,side,other,peers) {
    const role=roles(m,side,other,peers), direction=moveState(m,side);
    const pressure=side.money==null||side.tickets==null?"Unknown":
      [side.tickets,side.money].every(v=>v===0)||[side.tickets,side.money].every(v=>v===100)?"Possible cap":
      Math.max(side.tickets, side.money)<=45?"Lower public support":Math.min(side.tickets,side.money)>=55?"Higher public support":"Mixed public support";
    const juice=side.price_toward_implied, entry=implied(side.price), opener=implied(side.open_price);
    const secondary=[m==="MONEYLINE"?peers?.SPREAD?.line_toward:peers?.MONEYLINE?.price_toward_implied,
      m==="TOTAL"?peers?.SPREAD?.line_toward:peers?.TOTAL?.line_toward];
    return {role:role.includes("Unknown")?null:role.join(" → "), direction, entry,
      movement:m==="MONEYLINE"?juice:side.line_toward, juice, opener:m==="MONEYLINE"?opener:side.open_line,
      pressure, cross:crossState(m,side,peers), total:totalState(m,side,peers),
      secondary,
      key:keyTransitions(sport,m,side).join(" / "),
      crossover:role.includes("Unknown")||role[0]===role[1]?"":role.join(" → ")};
  }
  const near = (a,b,band) => a!=null&&b!=null&&Math.abs(a-b)<=band+1e-9;
  function breakdownBands(sport,m) {
    const scale=m==="MONEYLINE"?.018:["nfl","ncaaf","nhl","mlb"].includes(sport)?.7:1.5;
    return [m==="MONEYLINE"?.05:m==="SPREAD"?2.5*scale:["nhl","mlb"].includes(sport)?1:4,scale];
  }
  function comparisons(r,q,m,prior,current,sport) {
    const [openerBand,scale]=breakdownBands(sport,m);
    const sameRole=current.role!=null&&prior.role===current.role;
    const sameMove=current.direction!=="Unknown"&&prior.direction===current.direction;
    const sameOpener=near(prior.opener,current.opener,openerBand)&&near(prior.movement,current.movement,scale);
    const sameSplits=near(r.tickets,q.tickets,10)&&near(r.money,q.money,13)&&
      ![prior.pressure,current.pressure].includes("Possible cap");
    const sameJuice=near(prior.juice,current.juice,.015)&&
      moveState("MONEYLINE",{price_toward_implied:prior.juice})===moveState("MONEYLINE",{price_toward_implied:current.juice});
    const sameCross=prior.cross===current.cross&&prior.cross!=="Unavailable";
    const sameTotal=prior.total===current.total&&prior.total!=="Unknown";
    const peerBands=[m==="MONEYLINE"?1.5:.018,["nhl","mlb"].includes(sport)?.75:2];
    const samePeers=sameCross&&sameTotal&&peerBands.every((band,i)=>near(prior.secondary[i],current.secondary[i],band));
    return {all:true, role:sameRole, direction:sameMove, role_move:sameRole&&sameMove,
      opener:sameOpener, juice:sameJuice, split:sameSplits,
      pressure:!["Unknown","Possible cap"].includes(current.pressure)&&prior.pressure===current.pressure,
      cross_agree:prior.cross==="Agreement", cross_disagree:prior.cross==="Disagreement",
      cross_neutral:prior.cross==="Neutral", cross_unknown:prior.cross==="Unavailable",
      total_rising:prior.total==="Rising", total_falling:prior.total==="Falling",
      total_held:prior.total==="Held", total_unknown:prior.total==="Unknown",
      key:!!current.key&&prior.key===current.key, crossover:!!current.crossover&&prior.crossover===current.crossover,
      context:sameRole&&sameMove&&samePeers,
      combined:sameRole&&sameMove&&sameOpener&&sameSplits&&sameJuice&&
        (sport==="ufc"||samePeers)};
  }
  const subsetLabels = {all:"Entire group",role:"Same opening and observed role",direction:"Same primary direction",
    role_move:"Same role and direction",opener:"Comparable opener and movement size",juice:"Comparable juice or implied move",
    split:"Comparable ticket and money splits",pressure:"Same public pressure regime",
    cross_agree:"Spread and moneyline agree",cross_disagree:"Spread and moneyline disagree",
    cross_neutral:"Spread or moneyline is neutral",cross_unknown:"Spread or moneyline unavailable",
    total_rising:"Total rising",total_falling:"Total falling",total_held:"Total unchanged",total_unknown:"Total unavailable",
    key:"Same key-number transition",crossover:"Same role crossover",context:"Same role and direction with similar peer moves",
    combined:"Combined multidimensional similarity"};

  function summarizeRows(rows) {
    const counts={win:0,loss:0,push:0};
    let decimals=0,units=0;
    for(const {grade,side} of rows) {
      const decimal=1/implied(side.price);
      counts[grade]++;
      decimals+=decimal;
      units+=grade==="win"?decimal-1:grade==="loss"?-1:0;
    }
    const n=rows.length, avg=decimals/n, decisive=counts.win+counts.loss,p=counts.win/decisive;
    const den=1+3.8416/decisive, center=(p+3.8416/(2*decisive))/den;
    const width=1.96*Math.sqrt(p*(1-p)/decisive+3.8416/(4*decisive*decisive))/den;
    return {...counts,n,units,roi:n?100*units/n:null,average:n?Math.round(avg>=2?(avg-1)*100:-100/(avg-1)):null,
      breakEven:n?100/avg:null, winRate:decisive?100*p:null,
      low:decisive?100*Math.max(0,center-width):null, high:decisive?100*Math.min(1,center+width):null};
  }

  function marketsEvidence(m,side,peers,context,reference,stamp) {
    return `<div class="historic-market-evidence">${Object.keys(labels).map(name=>{
      const q=name===m?side:peers?.[name], primary=name===m;
      const role=primary?"Primary market":name==="TOTAL"?"Supporting · Over reference":
        m==="TOTAL"&&reference?"Supporting · "+reference:"Supporting market";
      const splitCap=q&&q.tickets===q.money&&(q.tickets===0||q.tickets===100);
      const capLabel=primary&&/split (risk|cap)/i.test([context?.data_badge,context?.context_chips].join(" "))
        ?" · source flagged a split cap":" · split cap unverified";
      return `<div class="historic-market-proof ${primary?"primary":""}">
        <small>${escape(role)}</small><strong>${labels[name]}</strong>
        ${q?`<b>${escape(quote(name,q,true)+" → "+quote(name,q))}</b><span>${escape(movement(name,q)+path(q))}</span>
          <span>${escape(pct(q.tickets)+" tickets / "+pct(q.money)+" money"+(splitCap?capLabel:""))}</span>
          <span>${escape(when(q.captured_at||stamp))}${primary&&context?.data_badge?" · "+escape(context.data_badge):""}</span>
          ${q.path_turns?.length?`<details><summary>Intermediate quotes</summary><ul>${q.path_turns.map(turn=>
            `<li>${escape(when(turn.captured_at)+" · "+quote(name,turn))}</li>`).join("")}</ul></details>`:""}
          ${q.trajectory==="retraced"&&q.path_retrace!=null?`<span>Gave back ${escape(
            (name==="MONEYLINE"?q.path_retrace*100:q.path_retrace).toFixed(1)
            +(name==="MONEYLINE"?" implied points":" line points"))}</span>`:""}
          ${q.path_direction_changes>0?`<span>${q.path_direction_changes} observed direction changes</span>`:""}`:
          '<span>Aligned pregame evidence unavailable</span>'}</div>`;
    }).join("")}</div>`;
  }

  function differences(trial, current) {
    const notes=[];
    if(trial.role&&current.role&&trial.role!==current.role)notes.push("Different role");
    if(trial.direction!=="Unknown"&&trial.direction!==current.direction)notes.push("Different direction");
    if(trial.cross!=="Unavailable"&&current.cross!=="Unavailable"&&trial.cross!==current.cross)notes.push("Different peer relationship");
    if(trial.total!=="Unknown"&&current.total!=="Unknown"&&trial.total!==current.total)notes.push("Different total direction");
    if(trial.entry!=null&&current.entry!=null&&!near(trial.entry,current.entry,.06))notes.push("Different price");
    if(trial.key!==current.key)notes.push("Different key-number path");
    if(trial.pressure==="Possible cap")notes.push("Extreme splits · cap unverified");
    return notes;
  }

  function ledgerExamples(doc,result,side,opposite,setup,m) {
    const sport=active.payload.sport, records=doc.groups[active.cohort]||[];
    const current=profile(sport,m,side,opposite,side.game_context);
    const rows=records.map(i=>{
      const row=doc.rows[i], trial=profile(row.sport,m,row.side,row.opposite,row.peers);
      return {row,trial,flags:comparisons(row.side,side,m,trial,current,sport)};
    });
    const available=Object.entries(subsetLabels).filter(([key])=>{
      if(sport==="ufc"&&(key.startsWith("cross_")||key.startsWith("total_")||key==="context"))return false;
      if(key==="key"&&!current.key||key==="crossover"&&!current.crossover||key==="role"&&!current.role)return false;
      return true;
    });
    const chosen=rows.filter(x=>x.flags[active.subset]);
    const summary=summarizeRows(chosen.map(x=>x.row));
    const start=active.page*10;
    const units=summary.units.toFixed(2), percent=x=>x==null?"—":x.toFixed(1)+"%";
    return `<section class="historic-examples historic-ledger">
      <div class="historic-ledger-head"><h3>Graded comparison games</h3><label>Show
        <select data-historic-subset>${available.map(([key,label])=>`<option value="${key}"${key===active.subset?" selected":""}>
          ${escape(label)} · ${rows.filter(x=>x.flags[key]).length}</option>`).join("")}</select></label></div>
      ${active.subset!=="all"?`<div class="historic-metrics historic-subset-row">
        ${tile("Record",record(summary),summary.n+" of "+result.graded+" graded")}
        ${tile("Win rate",percent(summary.winRate))}${tile("ROI",percent(summary.roi),signed(+units)+" units",tone(summary.roi))}
        ${tile("Avg. entry odds",odds(summary.average))}${tile("Price-equivalent break-even",percent(summary.breakEven))}</div>`:""}
      <p class="historic-note">${active.subset==="all"&&summary.n?"Price-equivalent break-even "+percent(summary.breakEven)+". ":""}
        ${summary.n<30?"Limited sample. ":""}${summary.low!=null?
        "Descriptive win-rate interval (95%) "+percent(summary.low)+"–"+percent(summary.high)+". ":""}
        Subsets overlap within this group. Break-even uses the average decimal payout. Outcome dependence may widen uncertainty.</p>
      ${chosen.slice(start,start+10).map(({row,trial})=>{
        const marks=differences(trial,current), clock=row.actual_start||row.date;
        const mins=(Date.parse(clock)-Date.parse(row.entry))/60000;
        const display={...row,side:{...row.side,game_context:row.peers,context_reference:row.reference}};
        return `<details class="historic-example"><summary>${exampleRow(display,m,true)}
          <span class="historic-row-profile">${escape(trial.role||"Role unavailable")} · ${escape(trial.direction)}
          ${marks.length?`<b>· ${escape(marks.slice(0,3).join(" · "))}</b>`:""}</span></summary>
          <div class="historic-example-detail">
          <p><strong>${escape(row.selection)}</strong> · ${escape(minuteText(mins))} before ${row.actual_start?"verified start":"scheduled start"}</p>
          <p>${escape(row.score||"Verified result · final score unavailable")} · Graded at ${escape(quote(m,row.side))} · ${escape(when(row.entry))}</p>
          ${marketsEvidence(m,row.side,row.peers,row.context,row.reference,row.entry)}
          <div class="historic-why"><div><strong>Why included</strong>
          <p>Qualified for ${escape(cohorts[active.cohort])} at distance ${row.distance}. ${escape(row.matched.join(", "))}.</p></div>
          <div><strong>Compared with the current setup</strong><p>${escape([...marks,...row.differences].join(" · ")||"No large recorded difference in scored factors.")}</p>
          <p>${escape("Current "+(current.role||"role unavailable")+" / "+current.direction+" · "+pct(side.tickets)+" / "+pct(side.money)+
            " · Example "+(trial.role||"role unavailable")+" / "+trial.direction+" · "+pct(row.side.tickets)+" / "+pct(row.side.money))}</p></div></div>
          </div></details>`;
      }).join("")||'<p class="historic-note">No graded games meet this breakdown.</p>'}
      <div class="historic-ledger-pages"><span aria-live="polite">${chosen.length?
        "Showing "+(start+1)+"–"+Math.min(start+10,chosen.length)+" of "+chosen.length:"0 graded games"}</span>
        ${chosen.length>10?`<button type="button" data-historic-page="-1"${start?"":" disabled"}>Previous</button>
        <button type="button" data-historic-page="1"${start+10<chosen.length?"":" disabled"}>Next</button>`:""}
      </div></section>`;
  }

  function ledgerPanel(name,result,side,opposite,setup,m) {
    if(!name || !result?.graded) return examples(result,m);
    const state=active, stored=state.ledgers.get(name);
    if(stored?.version===1 && stored.rows && stored.game===state.payload.sport+"--"+state.payload.event &&
       stored.market===m && stored.side===side.side && stored.observed===setup.observed &&
       stored.groups?.[state.cohort]?.length===result.graded) return ledgerExamples(stored,result,side,opposite,setup,m);
    if(stored==="error" || stored?.rows) return '<p class="historic-note">The complete ledger could not be verified. Reopen this comparison to retry.</p>';
    if(!stored) {
      state.ledgers.set(name,"loading");
      Promise.resolve().then(async()=>{
        try {
          if(!/^evidence--[a-f0-9]{32}$/.test(name)) throw new Error();
          const response=await authorized(base+name+".json");
          if(!response.ok) throw new Error();
          state.ledgers.set(name,await response.json());
        } catch (_) { state.ledgers.set(name,"error"); }
        if(active===state) render();
      });
    }
    return '<p class="historic-note" role="status">Loading the complete graded ledger…</p>';
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
      <section class="historic-by-market"><h3>Separate comparison groups by market</h3><div>${markets.map(([key,label])=>{
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
    const other=setup.sides[1-index], current=profile(payload.sport,market,side,other,side.game_context);
    const [openBand,moveBand]=breakdownBands(payload.sport,market), scale=market==="MONEYLINE"?100:1;
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
      <section class="historic-reference"><h3>${payload.frozen?"Frozen pregame setup":stale?"Last verified setup":"Current market setup"}</h3>
        <p>${escape([current.role,current.direction,current.cross==="Unavailable"?"Peer relationship unavailable":"Spread / moneyline "+current.cross.toLowerCase(),
        minuteText(setup.lead)+" to scheduled start"].filter(Boolean).join(" · "))}</p>
        ${marketsEvidence(market,side,side.game_context,side.context,side.context_reference,setup.observed)}
        <details class="historic-filter-method"><summary>How the breakdowns work</summary>
          <p>These descriptive filters use only members of the selected group. Ticket and money gaps are limited to 10 and 13 points. Comparable juice uses a 1.5 implied-point gap with the same direction. Lower and higher public support mean both splits are at most 45% or at least 55%.</p>
          <p>Role uses the signed handicap or the stronger paired moneyline. For a primary total, role refers to the aligned opening favorite. The opener and movement gaps are at most ${openBand*scale} and ${moveBand*scale} ${market==="MONEYLINE"?"implied points":"line points"}. Neutral and unavailable peer evidence have separate breakdowns.</p>
          <p>Team and total evidence uses the same signed directions and similar move sizes. The peer gaps are 1.8 implied points for moneyline, 1.5 points for team spreads with moneyline selected, and ${["nhl","mlb"].includes(payload.sport)?"0.75":"2"} line points for the other supporting market. Material entry-price differences exceed six implied points.</p>
        </details>
      </section>
      ${ledgerPanel(side.comparison.ledger,result,side,other,setup,market)}
      <div class="historic-footer"><p>Includes Favorites and non-Favorites · ${payload.frozen?"Frozen":"Observed"} ${escape(when(setup.observed))}</p>
        ${support(setup,side,result,market,payload,markets)}</div>`;
    if(focus?.hasAttribute("data-historic-cohort"))body.querySelector("[data-historic-cohort]").focus({preventScroll:true});
    else if(focus?.hasAttribute("data-historic-subset"))body.querySelector("[data-historic-subset]")?.focus({preventScroll:true});
    else if(focus?.hasAttribute("data-historic-page")) {
      (body.querySelector(`[data-historic-page="${focus.dataset.historicPage}"]:not(:disabled)`)||
        body.querySelector("[data-historic-page]:not(:disabled)"))?.focus({preventScroll:true});
    }
    else if(focus?.dataset.historicAction) {
      [...body.querySelectorAll("[data-historic-action]")].find(el=>el.dataset.historicAction===focus.dataset.historicAction&&el.dataset.value===focus.dataset.value)?.focus({preventScroll:true});
    }
  }

  function init() {
    const modal = dialog();
    if (!modal) return;
    modal.addEventListener("click", event => {
      const pagination=event.target.closest("[data-historic-page]");
      if(pagination && active) {
        active.page+=Number(pagination.dataset.historicPage);
        render();
        return;
      }
      const picked = event.target.closest("[data-historic-action]");
      if (!picked || !active) return;
      const {historicAction, value} = picked.dataset;
      if (historicAction === "market") {
        active.market = value;
      } else active[active.market === "TOTAL" ? "total" : "team"] = Number(value);
      const side=active.payload.markets[active.market].sides[active.market==="TOTAL"?active.total:active.team];
      active.cohort=defaultCohort(side.comparison);
      active.page=0;
      active.subset="all";
      render();
    });
    modal.addEventListener("change", event => {
      if(!active)return;
      if(event.target.matches("[data-historic-cohort]")) {
        active.cohort=event.target.value;
        active.subset="all";
      } else if(event.target.matches("[data-historic-subset]")) active.subset=event.target.value;
      else return;
      active.page=0;
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
