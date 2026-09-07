from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
HOMEPAGE = ROOT / "site" / "index.html"


def _source() -> str:
    return HOMEPAGE.read_text(encoding="utf-8")


def test_homepage_preserves_normal_mobile_viewport_and_avoids_page_scaling():
    source = _source()
    soup = BeautifulSoup(source, "html.parser")
    viewport = soup.find("meta", attrs={"name": "viewport"})

    assert viewport is not None
    assert viewport.get("content") == "width=device-width, initial-scale=1"
    assert "maximum-scale" not in viewport.get("content", "")
    assert "user-scalable" not in viewport.get("content", "")
    assert "transform:scale" not in source.replace(" ", "").lower()
    assert "zoom:" not in source.replace(" ", "").lower()


def test_install_promotion_has_separate_desktop_artwork_and_mobile_image_link():
    source = _source()
    soup = BeautifulSoup(source, "html.parser")

    desktop = soup.select_one(".mobile-install-desktop")
    mobile = soup.select_one(".mobile-install-card")
    artwork = desktop.select_one("img.mobile-install-artwork")
    assert desktop is not None and mobile is not None and artwork is not None
    assert artwork.get("src", "").startswith("/mobile-install-banner.png")
    assert mobile.name == "a"
    assert mobile.get("id") == "mobile-app-cta"
    assert mobile.get("href") == "/app"
    assert mobile.get("aria-label") == "Install Red Fox on your phone"
    mobile_artwork = mobile.select_one("img.mobile-install-card-image")
    assert mobile_artwork is not None
    assert mobile_artwork.get("src", "").startswith("/mobile-install-promo.png")
    assert mobile.select_one(".mobile-banner-qr-image") is None
    assert "@media (max-width: 768px)" in source
    assert ".mobile-install-desktop { display:none; }" in source
    compact = source.replace(" ", "").replace("\n", "")
    assert ".mobile-install-card-image{display:block;width:100%;max-width:100%;height:auto;}" in compact


def test_free_trial_and_pricing_copy_are_explicit_and_annual_is_only_featured_plan():
    soup = BeautifulSoup(_source(), "html.parser")

    assert soup.select_one("#hero-primary-action").get_text(strip=True) == "Try Free for 24 Hours"
    assert soup.select_one("#hero-access-note").get_text(" ", strip=True) == (
        "No credit card required · Full board access · 24 hours from activation"
    )
    assert soup.select_one("#pricing h2").get_text(strip=True) == "Continue after your free 24 hours."
    assert soup.select_one("#pricing .pricing-sub").get_text(strip=True) == (
        "Choose a day when you need it or keep the full board open all season."
    )
    featured = soup.select("#pricing .price-card.featured")
    assert len(featured) == 1
    assert "Annual" in featured[0].get_text(" ", strip=True)
    assert len(soup.select("#pricing .best-value")) == 1


def test_homepage_account_states_remain_presentation_over_existing_entitlements():
    source = _source()

    assert "_supabase.rpc('has_active_access')" in source
    assert "_supabase.rpc('claim_free_day')" in source
    for state in ("trial_available", "active_trial", "expired_trial", "paid", "admin"):
        assert state in source
    for label in (
        "Start My Free 24 Hours",
        "Go to Board",
        "View Paid Options",
        "Free trial · ",
    ):
        assert label in source


def test_mobile_screenshots_and_pricing_keep_intrinsic_widths():
    source = _source().replace(" ", "").lower()

    assert ".preview-screenshot,.journey-screenshot{width:100%;max-width:100%;height:auto;object-fit:contain;}" in source
    assert ".pricing-grid{grid-template-columns:minmax(0,1fr);gap:16px;width:100%;}" in source
    assert "grid-template-columns:repeat(3,minmax(0,1fr));" in source
    assert "object-fit:cover" not in source


def test_homepage_favorite_copy_update_preserves_existing_structure_and_assets():
    source = _source()
    soup = BeautifulSoup(source, "html.parser")

    assert soup.select_one(".hero-badge").get_text(strip=True) == (
        "Bloomberg-style market intelligence for sports bettors"
    )
    assert soup.select_one(".hero h1").get_text(" ", strip=True) == (
        "Make More Informed Wagering Decisions."
    )
    assert soup.select_one(".hero-sub").get_text(" ", strip=True) == (
        "Red Fox continuously evaluates market positioning, line and price movement, pressure, resistance, "
        "reversals, and timing to understand how the book is responding. Start with Red Fox Favorites, then drill "
        "into the Market Reads and evidence behind every signal."
    )
    assert soup.select_one(".hero-philosophy").get_text(" ", strip=True) == (
        "We study how the book responds to betting pressure — not just where the bets are."
    )
    assert soup.select_one(".hero-philosophy > strong") is not None
    assert soup.select_one(".hero-positioning") is None
    assert [item.get_text(" ", strip=True) for item in soup.select(".feat-card h3")] == [
        "Follow the Market Journey", "Understand the Market Read", "Start With Red Fox Favorites"
    ]
    feature_copy = [item.get_text(" ", strip=True) for item in soup.select(".feat-card p")]
    assert feature_copy == [
        "See where a line started, how price and splits changed along the way, and where the market stands now. "
        "Replay the journey whenever you need more context.",
        "Red Fox interprets how the book responds to positioning, movement, pressure, resistance, reversals, and "
        "timing — then turns that behavior into a structured Market Read.",
        "Quickly find the selective markets that meet Red Fox’s preferred wager criteria, then open the full "
        "Market Read and evidence before making your decision.",
    ]
    assert [item.get_text(" ", strip=True) for item in soup.select(".not-grid .not-item")][:2] == [
        "× Blind picks", "✓ Evidence-backed Favorites"
    ]
    assert soup.select_one(".journey-teaser h3").get_text(strip=True) == (
        "See the evidence behind every Market Read."
    )
    assert soup.select_one(".journey-teaser p").get_text(" ", strip=True) == (
        "Trace the line or price from first observation to current state and review the market evidence behind the read."
    )
    preview = soup.select_one(".preview")
    assert [heading.get_text(" ", strip=True) for heading in preview.select("h2")] == [
        "Favorites first. Full market intelligence underneath.", "See the Market Story Unfold"
    ]
    assert [item.get_text(" ", strip=True) for item in preview.select(":scope > .container > .preview-sub")] == [
        "Red Fox Favorites rise to the top, followed by the strongest remaining Market Reads.",
        "Follow the journey from open to current, then review the exact path, split changes, and timestamped evidence.",
    ]
    assert [image.get("src") for image in soup.select(".preview-screenshot,.journey-screenshot")] == [
        "home-board.png?v=20260904", "home-journey.png?v=20260904"
    ]
    assert [item.get_text(" ", strip=True) for item in soup.select(".price-card:nth-of-type(1) .price-features li")] == [
        "Full board access", "Red Fox Favorites", "All active markets", "Market journey and timeline", "Real-time updates"
    ]
    assert [item.get_text(" ", strip=True) for item in soup.select(".price-card:nth-of-type(2) .price-features li")] == [
        "Unlimited daily access", "Red Fox Favorites", "All markets and timeline history", "Live Market Read context", "Cancel anytime"
    ]
    assert [item.get_text(" ", strip=True) for item in soup.select(".price-card:nth-of-type(3) .price-features li")] == [
        "Everything in Monthly Unlimited", "Red Fox Favorites all season", "Full season of market journeys", "Cancel anytime"
    ]
    page_text = soup.get_text(" ", strip=True)
    assert "Market-driven" in page_text
    assert "Price-sensitive" not in page_text
    assert "DraftKings" not in page_text


def test_homepage_book_response_copy_is_single_source_across_responsive_views():
    source = _source()
    assert source.count("We study how the book responds to betting pressure — not just where the bets are.") == 1
    assert source.count('class="hero-sub"') == 1
    assert source.count('class="hero-philosophy"') == 1
    assert ".hero-philosophy {" in source
    assert ".hero-philosophy { font-size:14px; margin-bottom:32px; }" in source
    assert ".hero-ctas {" in source


def test_desktop_pricing_cards_use_equal_height_flow_without_changing_mobile_cards():
    source = _source().replace(" ", "").replace("\n", "")

    desktop = source[source.index("@media(min-width:769px)"):source.index("@media(min-width:1201px)")]
    assert ".price-card{display:flex;flex-direction:column;}" in desktop
    assert ".price-features{flex:11auto;}" in desktop
    assert ".price-actions{flex:0093px;}" in desktop
    assert source.count('class="price-actions"') == 3
    mobile = source[source.index("@media(max-width:768px)"):]
    assert ".price-card{display:flex" not in mobile
    assert ".price-actions{" not in mobile
