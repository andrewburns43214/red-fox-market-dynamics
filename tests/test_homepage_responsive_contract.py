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


def test_install_promotion_has_separate_desktop_artwork_and_mobile_html_card():
    source = _source()
    soup = BeautifulSoup(source, "html.parser")

    desktop = soup.select_one(".mobile-install-desktop")
    mobile = soup.select_one(".mobile-install-card")
    artwork = desktop.select_one("img.mobile-install-artwork")
    assert desktop is not None and mobile is not None and artwork is not None
    assert artwork.get("src", "").startswith("/mobile-install-banner.png")
    assert mobile.select_one("#mobile-app-cta").get_text(strip=True) == "Install Red Fox"
    assert mobile.select_one(".mobile-banner-qr-image") is None
    assert "@media (max-width: 768px)" in source
    assert ".mobile-install-desktop { display:none; }" in source


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
