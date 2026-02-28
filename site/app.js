const DASH_URL = "/data/dashboard.csv";
const SNAP_URL = "/data/snapshots.csv";

/* normalize header whitespace + BOM */
function normalize(row){
  const obj = {};
  Object.keys(row).forEach(k => {
    const clean = k.replace(/\r|\n|\ufeff/g,"").trim();
    obj[clean] = row[k];
  });
  return obj;
}

/* REMOVE INVISIBLE CHARACTERS FROM VALUES */
function cleanVal(v){
  return String(v ?? "")
    .replace(/\r|\n|\ufeff/g,"")
    .trim();
}

/* percent formatting */
function pct(v){
  if(v===undefined || v===null || v==="") return "-";
  return `${v}%`;
}

/* EST time formatting */
function formatTime(ts){
  if(!ts) return "-";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return String(ts);
  return d.toLocaleString("en-US",{
    month:'2-digit',
    day:'2-digit',
    hour:'2-digit',
    minute:'2-digit',
    hour12:true,
    timeZone:'America/New_York'
  });
}

/* MARKET LINE PARSER */
function splitLine(raw){
  if(!raw) return [null,null];
  const parts = String(raw).split("@");
  if(parts.length !== 2) return [String(raw).trim(), null];
  return [parts[0].trim(), parts[1].trim()];
}

function parseMarketLine(raw, market){
  if(!raw) return "-";

  const [selection, odds] = splitLine(raw);
  if(!selection) return "-";

  if(market==="MONEYLINE") return odds || "-";

  if(market==="TOTAL"){
    const m = selection.match(/(\d+(\.\d+)?)/);
    return m ? m[1] : "-";
  }

  if(market==="SPREAD"){
    const m = selection.match(/[+-]\d+(\.\d+)?/);
    return m ? m[0] : "-";
  }

  return "-";
}

/* DECISION LABEL */
function decisionInfo(score){
  if(score>=72) return ["STRONG BET","strong"];
  if(score>=68) return ["BET","bet"];
  if(score>=60) return ["LEAN","lean"];
  return ["NO BET","nobet"];
}

/* ---------- SNAPSHOT INDEX ---------- */
function parseTs(ts){
  const d = new Date(ts);
  const t = d.getTime();
  return isNaN(t) ? 0 : t;
}

function normTeam(s){
  return cleanVal(s).toLowerCase().replace(/\s+/g," ");
}

function snapSelection(sideStr){
  return normTeam(String(sideStr||"").split("@")[0]);
}

function buildSnapIndex(snaps){
  const byGame = new Map();
  for(const raw of snaps){
    const gid = cleanVal(raw.game_id);
    if(!gid) continue;
    const arr = byGame.get(gid) || [];
    arr.push(raw);
    byGame.set(gid, arr);
  }
  for(const arr of byGame.values()){
    arr.sort((a,b)=> parseTs(b.timestamp) - parseTs(a.timestamp));
  }
  return byGame;
}

function findSnapshotForRow(snapIndex, row){
  const gid = cleanVal(row.game_id);
  const arr = snapIndex.get(gid);
  if(!arr) return null;

  const pick = normTeam(row.favored_side);
  if(pick){
    const exact = arr.find(x => snapSelection(x.side) === pick);
    if(exact) return exact;
  }
  return arr[0];
}

/* MAIN LOAD */
async function loadBoard(){

  const [dashTxt, snapTxt] = await Promise.all([
    fetch(DASH_URL + "?t=" + Date.now()).then(r=>r.text()),
    fetch(SNAP_URL + "?t=" + Date.now()).then(r=>r.text())
  ]);

  const dash = Papa.parse(dashTxt,{header:true, skipEmptyLines:true}).data.map(normalize);
  const snaps = Papa.parse(snapTxt,{header:true, skipEmptyLines:true}).data.map(normalize);

  const snapIndex = buildSnapIndex(snaps);

  const top = dash
    .map(r => ({...r, _score: Number(r.game_confidence)}))
    .filter(r => Number.isFinite(r._score))
    .sort((a,b)=> b._score - a._score)
    .slice(0,10);

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";

  top.forEach((s,i)=>{
    const snap = findSnapshotForRow(snapIndex, s);

    const score = Math.round(Number(s.game_confidence||0));
    const edge  = Math.round(Number(s.net_edge||0));
    const [decisionText, decisionClass] = decisionInfo(score);

    const openLine = snap ? parseMarketLine(snap.open_line, s.market_display) : "-";
    const curLine  = snap ? parseMarketLine(snap.current_line, s.market_display) : "-";

    /* FINAL PICK FIX */
    let pick = cleanVal(s.favored_side);
    if(!pick && snap && snap.side){
      pick = cleanVal(snap.side.split("@")[0]);
    }
    if(!pick) pick = "-";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i+1}</td>
      <td>${s.game || "-"}</td>
      <td>${formatTime(s._game_time)}</td>
      <td>${s.market_display || "-"}</td>
      <td>${snap ? pct(snap.bets_pct) : "-"}</td>
      <td>${snap ? pct(snap.money_pct) : "-"}</td>
      <td>${openLine}</td>
      <td>${curLine}</td>
      <td>${edge}</td>
      <td>${score}</td>
      <td><span class="decision ${decisionClass}">${decisionText}</span></td>
      <td class="pick">${pick}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("status").textContent =
    `LIVE • ${top.length} signals • ${new Date().toLocaleTimeString()}`;
}

setInterval(loadBoard, 5000);
loadBoard();
