"""Measure the customer-visible Live & Recent render path in a real browser."""

from __future__ import annotations

import argparse
import json
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--executable", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--screenshot")
    args = parser.parse_args()

    viewport = {"width": 390, "height": 844} if args.mobile else {"width": 1440, "height": 1000}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=args.executable)
        page = browser.new_page(viewport=viewport)
        requests: list[str] = []
        responses: dict[str, int] = {}
        page.on("request", lambda request: requests.append(request.url))
        page.on("response", lambda response: responses.__setitem__(response.url, response.request.sizes()["responseBodySize"]))
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_function("typeof window.loadLiveRecent === 'function'", timeout=30_000)
        page.evaluate(
            """
            window.__liveMeasure={clicked:performance.now(),controls:null,firstCard:null,peakInitialCards:0};
            const root=document.getElementById('tab-live-recent');
            const observer=new MutationObserver(()=>{
              const now=performance.now(), cards=root.querySelectorAll('.live-game-card').length;
              if(!window.__liveMeasure.controls && root.querySelector('.live-recent-controls')) window.__liveMeasure.controls=now;
              if(!window.__liveMeasure.firstCard && cards) window.__liveMeasure.firstCard=now;
              window.__liveMeasure.peakInitialCards=Math.max(window.__liveMeasure.peakInitialCards,cards);
            });
            observer.observe(root,{childList:true,subtree:true});
            """
        )
        before = time.perf_counter()
        page.evaluate("setTab(document.querySelector('.tab[data-tab=\"live-recent\"]'))")
        page.evaluate("if(document.querySelector('.live-recent-controls'))window.__liveMeasure.controls=performance.now()")
        page.wait_for_selector(".live-game-card", timeout=120_000)
        first = time.perf_counter()
        page.wait_for_timeout(250)
        measure = page.evaluate("window.__liveMeasure")
        initial_metrics = page.evaluate(
            """
            ({
              initialCards:document.querySelectorAll('.live-game-card').length,
              initialDomElements:document.getElementById('tab-live-recent').querySelectorAll('*').length,
              gameCount:Number(document.getElementById('b-live')?.textContent||0),
              controls:Boolean(document.querySelector('.live-recent-controls')),
            })
            """
        )
        if args.screenshot:
            page.screenshot(path=args.screenshot, full_page=True)
        full_before = time.perf_counter()
        if page.evaluate("typeof showMoreLiveRecent === 'function'"):
            page.evaluate("while(document.querySelector('#live-recent-more'))showMoreLiveRecent()")
        full_metrics = page.evaluate(
            "({cards:document.querySelectorAll('.live-game-card').length,liveDomElements:document.getElementById('tab-live-recent').querySelectorAll('*').length})"
        )
        full_after = time.perf_counter()
        live_requests = [url for url in requests if "live_recent" in url or "anomaly_event_details" in url or "anomaly_events.csv" in url]
        live_bytes = max([size for url, size in responses.items() if "live_recent" in url and size > 0] or [0])
        output = {
            **initial_metrics,
            **full_metrics,
            "clickToFirstCardMs": round((first - before) * 1000, 1),
            "clickToFullRenderMs": round((full_after - before) * 1000, 1),
            "remainingFullRenderMs": round((full_after - full_before) * 1000, 1),
            "observerFirstCardMs": round((measure["firstCard"] - measure["clicked"]), 1) if measure["firstCard"] else None,
            "controlsMs": round((measure["controls"] - measure["clicked"]), 1) if measure["controls"] else None,
            "livePayloadBytes": live_bytes,
            "livePathRequestCount": len(live_requests),
            "detailRequestCount": sum("anomaly_event_details" in url for url in live_requests),
            "viewport": viewport,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        browser.close()


if __name__ == "__main__":
    main()
