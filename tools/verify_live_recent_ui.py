"""Browser-level verification for Live & Recent controls and lazy rendering."""

from __future__ import annotations

import argparse
import json

from playwright.sync_api import sync_playwright


def verify(url: str, executable: str, mobile: bool) -> dict:
    viewport = {"width": 390, "height": 844} if mobile else {"width": 1440, "height": 1000}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page(viewport=viewport)
        detail_requests: list[str] = []
        page.on("request", lambda request: detail_requests.append(request.url) if "anomaly_event_details" in request.url else None)
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_function("typeof window.loadLiveRecent === 'function'", timeout=30_000)
        page.evaluate("setTab(document.querySelector('.tab[data-tab=\"live-recent\"]'))")
        page.wait_for_selector(".live-game-card", timeout=120_000)

        result = page.evaluate(
            """
            (()=>{
              if(_liveRecentObserver)_liveRecentObserver.disconnect();
              const all=groupLiveRecentRows(allLiveRecent), states=games=>games.map(game=>liveState(game));
              const sports=[...new Set(allLiveRecent.map(row=>sportLabel(row.sport)).filter(Boolean))].sort();
              const sportButtons=[...document.querySelectorAll('#live-sport-filters .schip')].map(node=>node.textContent.trim());
              liveRecentSport='ALL';liveRecentStatus='ALL';liveRecentSort='STATUS_TIME';
              const defaultGames=sortLiveRecentGames(filteredLiveRecentGames());
              const defaultStates=states(defaultGames), firstRecent=defaultStates.indexOf('post');
              const defaultOrdered=(firstRecent<0||defaultStates.slice(0,firstRecent).every(state=>state==='in'))&&defaultStates.slice(firstRecent<0?defaultStates.length:firstRecent).every(state=>state!=='in');
              liveRecentStatus='LIVE';const liveGames=sortLiveRecentGames(filteredLiveRecentGames());
              liveRecentStatus='RECENT';const recentGames=sortLiveRecentGames(filteredLiveRecentGames());
              liveRecentStatus='ALL';liveRecentSort='GAME_TIME';const byTime=sortLiveRecentGames(filteredLiveRecentGames()).map(game=>game.kickoff);
              liveRecentSort='RANK';const byRank=sortLiveRecentGames(filteredLiveRecentGames()).map(game=>game.bestRank);
              const priorSaved=window.isSavedGame;window.isSavedGame=rows=>String(rows[0]?.game_id)==='__saved__';
              const savedRows=[...allLiveRecent,{...allLiveRecent[0],game_id:'__saved__',game:'Saved Test'}];liveRecentSort='SAVED';
              const savedFirst=sortLiveRecentGames(groupLiveRecentRows(savedRows))[0].first.game_id==='__saved__';window.isSavedGame=priorSaved;
              liveRecentSort='STATUS_TIME';liveRecentVisible=18;renderLiveRecent();if(_liveRecentObserver)_liveRecentObserver.disconnect();
              const initialCards=document.querySelectorAll('.live-game-card').length;while(document.querySelector('#live-recent-more'))showMoreLiveRecent();
              const renderedKeys=[...document.querySelectorAll('.live-game-card')].map(node=>node.dataset.liveGameKey);
              return {games:all.length,sports,sportButtons,defaultOrdered,liveOnly:liveGames.every(game=>liveState(game)==='in'),recentOnly:recentGames.every(game=>liveState(game)==='post'),gameTimeOrdered:byTime.every((value,index)=>!index||byTime[index-1]<=value),rankOrdered:byRank.every((value,index)=>!index||byRank[index-1]<=value),savedFirst,initialCards,allCards:renderedKeys.length,noDuplicates:new Set(renderedKeys).size===renderedKeys.length,bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth};
            })()
            """
        )
        expected_buttons = ["All", *result["sports"]]
        assert result["sportButtons"] == expected_buttons
        assert result["defaultOrdered"] and result["liveOnly"] and result["recentOnly"]
        assert result["gameTimeOrdered"] and result["rankOrdered"] and result["savedFirst"]
        assert result["initialCards"] == min(18, result["games"])
        assert result["allCards"] == result["games"] and result["noDuplicates"]
        assert not result["bodyOverflow"]

        page.locator(".live-market-side-row").first.click()
        page.wait_for_selector("#drill-overlay.open", timeout=30_000)
        result["drilldownDetailRequests"] = len(detail_requests)
        assert result["drilldownDetailRequests"] == 1
        result["viewport"] = viewport
        browser.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--executable", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    parser.add_argument("--mobile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.executable, args.mobile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
