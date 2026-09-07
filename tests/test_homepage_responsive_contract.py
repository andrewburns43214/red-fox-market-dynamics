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
        "Red Fox continuously evaluates market positioning, line and price movement, pressure and resistance, "
        "reversals, and timing to surface meaningful market behavior. Start with Red Fox Favorites, follow the "
        "strongest Market Reads, and drill into the evidence behind every signal."
    )
    assert soup.select_one(".hero-positioning").get_text(" ", strip=True) == (
        "No blind picks. Red Fox shows you what the market is doing, why it matters, and which setups meet its "
        "most selective criteria."
    )
    feature_copy = [item.get_text(" ", strip=True) for item in soup.select(".feat-card p")]
    assert feature_copy == [
        "See where a line started, how price and splits changed along the way, and where the market stands now. "
        "Replay the journey whenever you need more context.",
        "Red Fox combines positioning, movement, pressure, resistance, reversals, and timing into a structured "
        "Market Read with the supporting evidence behind it.",
        "Quickly find the selective markets that meet Red Fox’s preferred wager criteria, then open the full "
        "Market Read and evidence before making your decision.",
    ]
    assert [item.get_text(" ", strip=True) for item in soup.select(".not-grid .not-item")][:2] == [
        "× Blind picks", "✓ Evidence-backed Favorites"
    ]
    assert soup.select_one(".journey-teaser h3").get_text(strip=True) == (
        "See the evidence behind every Market Read."
    )
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
    assert "DraftKings" not in soup.get_text(" ", strip=True)
