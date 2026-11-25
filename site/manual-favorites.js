/* Only the server capability enables these controls. No account data is put on the board. */
const manualFavoriteRoute='/api/owner/favorites';
const manualFavoritePending=new Map();
let manualFavoriteEnabled=false;
let manualFavoriteView=0;

async function manualFavoriteRequest(query='',init={}){
  const session=(await window.redfoxSupabase.auth.getSession()).data.session;
  if(!session?.access_token) return null;
  const headers={...init.headers,Authorization:'Bearer '+session.access_token};
  return fetch(manualFavoriteRoute+query,{...init,headers,cache:'no-store'});
}
document.addEventListener('DOMContentLoaded',async ()=>{
  try {
    const response=await manualFavoriteRequest();
    manualFavoriteEnabled=Boolean(response?.ok&&(await response.json()).enabled===true);
    if(manualFavoriteEnabled&&document.body.classList.contains('game-detail-open')) renderGameDetail();
  } catch (_error) {}
});
window.redfoxSupabase?.auth.onAuthStateChange((event)=>{
  if(event==='SIGNED_OUT'){
    manualFavoriteEnabled=false;
    document.getElementById('manual-favorite-dialog')?.close();
    manualFavoritePending.clear();
    document.querySelectorAll('[data-manual-favorite-control]').forEach(button=>button.remove());
  }
});

function manualFavoriteButton(row){
  if(!manualFavoriteEnabled||!row?.game_id) return '';
  const sport=String(row.sport);
  if(sport!=='ufc'&&!(Date.parse(row.kickoff_iso)>Date.now())) return '';
  return '<button type="button" class="manual-favorite-button" data-manual-favorite-control data-sport="'+
    escHtml(sport)+'" data-event="'+escHtml(row.game_id)+'" onclick="openManualFavorite(this)">Manual Controls</button>';
}
function openFavoriteBadge(badge){
  const row=allDash.find(r=>String(r.game_id)===badge.dataset.favoriteEvent&&r.sport===badge.dataset.favoriteSport&&r.market_display===badge.dataset.favoriteMarket);
  openSignalGuide('Red Fox Favorite',row);
}
function manualFavoriteProjection(rows){
  if(!manualFavoritePending.size) return rows;
  for(const row of rows){
    const key=row.sport+'|'+row.game_id;
    const pending=manualFavoritePending.get(key);
    if(!pending) continue;
    if(Date.parse(_boardFreshness?.board_published_at)>=Date.parse(pending.as_of)) continue;
    if(Date.now()-Date.parse(pending.as_of)>15*60*1000) continue;
    const choice=pending.choice;
    const selected=choice?.market===row.market_display;
    if(pending.action==='restore'&&row.favorite_origin!=='manual') continue;
    row.red_fox_favorite=selected?'true':'false';
    row.favorite_side=selected?choice.side:'';
    row.favorite_origin=selected?'manual':'';
    row.manual_entry_price=selected?choice.price:'';
  }
  for(const [key,pending] of manualFavoritePending){
    if(Date.parse(_boardFreshness?.board_published_at)>=Date.parse(pending.as_of)||Date.now()-Date.parse(pending.as_of)>15*60*1000){
      manualFavoritePending.delete(key);
    }
  }
  return rows;
}

async function openManualFavorite(trigger){
  if(!manualFavoriteEnabled) return;
  const view=++manualFavoriteView;
  const sport=trigger.dataset.sport, event=trigger.dataset.event;
  const dialog=document.getElementById('manual-favorite-dialog');
  const form=dialog.querySelector('form');
  const message=dialog.querySelector('[role=status]');
  const title=dialog.querySelector('h2');
  const save=form.querySelector('button[type=submit]');
  const actions=form.querySelector('[data-manual-actions]');
  const quotes=form.querySelector('[data-manual-quotes]');
  const query='?'+new URLSearchParams({sport,event});
  let model;
  let saving=false;
  const current=()=>dialog.open&&view===manualFavoriteView;
  const selected=name=>form.querySelector('input[name="'+name+'"]:checked')?.value||'';
  const clear=group=>group.querySelector('.manual-favorite-options').replaceChildren();
  function addOption(group,name,value,label,disabled=false){
    const option=document.createElement('label');
    const input=document.createElement('input');
    input.type='radio';
    input.name=name;
    input.value=value;
    input.disabled=disabled;
    option.append(input,document.createTextNode(label));
    group.querySelector('.manual-favorite-options').append(option);
  }
  function setQuoteState(){
    const action=selected('action_choice');
    quotes.hidden=!['add','change'].includes(action);
    save.disabled=saving||!action||(!quotes.hidden&&!selected('quote'));
  }
  async function update(){
    save.disabled=true;
    actions.disabled=true;
    quotes.disabled=true;
    const response=await manualFavoriteRequest(query);
    if(!current()) return false;
    if(!response?.ok) throw new Error('Favorite controls are unavailable.');
    const context=await response.json();
    if(!current()) return false;
    if(!context.game) throw new Error('This game is no longer available on the pregame board.');
    model=context;
    title.textContent=model.game||'Manual Controls';
    clear(actions);
    clear(quotes);
    if(model.choice){
      addOption(actions,'action_choice','change','Change Favorite',!model.quotes.length);
      addOption(actions,'action_choice','remove','Remove Favorite');
    } else {
      addOption(actions,'action_choice','add','Add Favorite',!model.quotes.length);
    }
    if(model.override) addOption(actions,'action_choice','restore','Restore automatic qualification');
    for(const quote of model.quotes){
      const market=quote.market==='TOTAL'?'Total':quote.market==='SPREAD'?'Spread':'Moneyline';
      addOption(quotes,'quote',quote.token,market+' · '+quote.label+' ('+quote.price+')');
    }
    actions.disabled=false;
    quotes.disabled=false;
    setQuoteState();
    message.textContent=model.quotes.length?'':'Adding or changing a Favorite requires a fresh sportsbook quote.';
    return true;
  }
  title.textContent='Manual Controls';
  message.textContent='Checking current quotes…';
  form.reset();
  clear(actions);
  clear(quotes);
  if(allDash.some(row=>row.sport===sport&&String(row.game_id)===event&&isRedFoxFavorite(row))){
    addOption(actions,'action_choice','change','Change Favorite');
    addOption(actions,'action_choice','remove','Remove Favorite');
  } else {
    addOption(actions,'action_choice','add','Add Favorite');
  }
  actions.disabled=true;
  quotes.hidden=true;
  save.disabled=true;
  form.onchange=setQuoteState;
  form.onsubmit=async eventSubmit=>{
    eventSubmit.preventDefault();
    if(save.disabled) return;
    const action=selected('action_choice'), quote=selected('quote');
    saving=true;
    actions.disabled=true;
    quotes.disabled=true;
    setQuoteState();
    message.textContent='Verifying and saving…';
    try{
      const response=await manualFavoriteRequest('',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({sport,event,revision:model.revision,action,quote,reason:form.elements.reason.value}),
      });
      if(!response) throw new Error('Sign in again to save a Favorite.');
      const result=await response.json();
      if(!response.ok){
        if(current()&&(!result.refresh||await update())) message.textContent=result.error||'The decision could not be saved.';
        return;
      }
      manualFavoritePending.set(sport+'|'+event,result);
      manualFavoriteProjection(allDash);
      renderBoard(false);
      await loadAll({forceRender:true});
      if(!current()) return;
      const eventRows=allDash.filter(r=>r.sport===sport&&String(r.game_id)===event);
      const row=eventRows.find(r=>result.choice?r.market_display===result.choice.market:isRedFoxFavorite(r))||eventRows[0];
      if(document.getElementById('signal-guide-overlay').classList.contains('open')){
        openSignalGuide('Red Fox Favorite',row);
      }
      clear(actions);
      clear(quotes);
      const confirmation=result.action==='restore'?'Automatic qualification restored. The next board refresh will show its current decision.':'Saved.';
      message.textContent=confirmation;
      if(await update()) message.textContent=confirmation;
    } catch (error){
      if(current()) message.textContent=error.message||'The decision could not be saved.';
    } finally {
      if(current()){
        saving=false;
        actions.disabled=false;
        quotes.disabled=false;
        setQuoteState();
      }
    }
  };
  dialog.showModal();
  try{await update();}
  catch(error){if(current()) message.textContent=error.message;}
}
