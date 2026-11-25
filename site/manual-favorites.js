/* Only the server capability enables these controls. No account data is put on the board. */
const manualFavoriteRoute='/api/owner/favorites';
const manualFavoritePending=new Map();
let manualFavoriteEnabled=false;

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
  const sport=trigger.dataset.sport, event=trigger.dataset.event;
  const dialog=document.getElementById('manual-favorite-dialog');
  const form=dialog.querySelector('form');
  const message=dialog.querySelector('[role=status]');
  const title=dialog.querySelector('h2');
  const save=form.querySelector('button[type=submit]');
  const action=form.elements.action_choice, selectedQuote=form.elements.quote;
  const quoteLabel=selectedQuote.closest('label');
  const query='?'+new URLSearchParams({sport,event});
  let model;
  function setQuoteState(){
    quoteLabel.hidden=!['add','change'].includes(action.value);
    selectedQuote.required=!quoteLabel.hidden;
    save.disabled=!action.value||(!quoteLabel.hidden&&!selectedQuote.value);
  }
  async function update(){
    save.disabled=true;
    const response=await manualFavoriteRequest(query);
    if(!response?.ok) throw new Error('Favorite controls are unavailable.');
    model=await response.json();
    title.textContent=model.game||'Manual Controls';
    action.replaceChildren(new Option('Choose action',''));
    if(model.choice){
      if(model.quotes.length) action.add(new Option('Change Favorite','change'));
      action.add(new Option('Remove Favorite','remove'));
    } else if(model.quotes.length) action.add(new Option('Add Favorite','add'));
    if(model.override) action.add(new Option('Restore automatic qualification','restore'));
    selectedQuote.replaceChildren(new Option('Choose market and side',''));
    for(const quote of model.quotes){
      const market=quote.market==='TOTAL'?'Total':quote.market==='SPREAD'?'Spread':'Moneyline';
      selectedQuote.add(new Option(market+' · '+quote.label+' ('+quote.price+')',quote.token));
    }
    setQuoteState();
    if(action.options.length===1) message.textContent='No fresh eligible prices or actions are available.';
  }
  title.textContent='Manual Controls';
  message.textContent='Checking current quotes…';
  form.reset();
  action.replaceChildren();
  selectedQuote.replaceChildren();
  quoteLabel.hidden=true;
  form.onchange=setQuoteState;
  form.onsubmit=async eventSubmit=>{
    eventSubmit.preventDefault();
    if(save.disabled) return;
    save.disabled=true;
    message.textContent='Verifying and saving…';
    try{
      const response=await manualFavoriteRequest('',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({sport,event,revision:model.revision,action:action.value,quote:selectedQuote.value,reason:form.elements.reason.value}),
      });
      if(!response) throw new Error('Sign in again to save a Favorite.');
      const result=await response.json();
      if(!response.ok){
        if(result.refresh) await update();
        message.textContent=result.error||'The decision could not be saved.';
        return;
      }
      manualFavoritePending.set(sport+'|'+event,result);
      manualFavoriteProjection(allDash);
      renderBoard(false);
      await loadAll({forceRender:true});
      const eventRows=allDash.filter(r=>r.sport===sport&&String(r.game_id)===event);
      const row=eventRows.find(r=>result.choice?r.market_display===result.choice.market:isRedFoxFavorite(r))||eventRows[0];
      if(document.getElementById('signal-guide-overlay').classList.contains('open')){
        openSignalGuide('Red Fox Favorite',row);
      }
      message.textContent=result.action==='restore'?'Automatic qualification restored. The next board refresh will show its current decision.':'Saved.';
      action.replaceChildren();
      selectedQuote.replaceChildren();
      setQuoteState();
    } catch (error){
      message.textContent=error.message||'The decision could not be saved.';
    } finally {
      setQuoteState();
    }
  };
  dialog.showModal();
  try{message.textContent=''; await update();}
  catch(error){message.textContent=error.message;}
}
