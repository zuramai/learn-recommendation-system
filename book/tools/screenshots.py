"""Capture screenshots of the running marketplace for the book (chapter 13 tour).

Needs all three services running (recsys serve, cargo run, npm run dev) and the
app at http://localhost:5173. Run from book/:

    uv run python tools/screenshots.py                  # every screenshot except the fallback banner
    uv run python tools/screenshots.py --only fallback  # run while `recsys serve` is STOPPED
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = "http://localhost:5173"
OUT = Path(__file__).resolve().parent.parent / "assets" / "screenshots"
PASSWORD = "password"


def shot(page: Page, name: str, *, selector: str | None = None, full_page: bool = False) -> None:
    path = OUT / name
    page.wait_for_timeout(400)  # let transitions settle
    if selector:
        page.locator(selector).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=full_page)
    print(f"  {path.relative_to(OUT.parent.parent)}")


def login(page: Page, email: str) -> None:
    page.goto(f"{BASE}/")
    page.wait_for_load_state("networkidle")
    logout = page.get_by_role("button", name="Logout")
    if logout.count():
        logout.click()
        page.wait_for_timeout(500)
    page.goto(f"{BASE}/login")
    page.locator("input[type=email]").fill(email)
    page.locator("input[type=password]").fill(PASSWORD)
    page.get_by_role("button", name="Login").click()
    page.wait_for_url(lambda u: "/login" not in u, timeout=15000)


def wait_home(page: Page) -> None:
    page.goto(f"{BASE}/")
    page.get_by_text("Why this?").first.wait_for(timeout=20000)


def capture_all(page: Page) -> None:
    # Anonymous home + login page
    page.goto(f"{BASE}/")
    page.context.clear_cookies()
    page.goto(f"{BASE}/")
    page.get_by_text("Popular now").wait_for(timeout=20000)
    page.locator("a[href^='/products/']").first.wait_for()
    shot(page, "home-anon.png")
    page.goto(f"{BASE}/login")
    page.locator("input[type=email]").wait_for()
    shot(page, "login.png")

    # Shopper
    login(page, "user43@demo.test")
    wait_home(page)
    page.get_by_text("candidates").first.wait_for(timeout=20000)
    shot(page, "home-user.png")
    page.get_by_text("Why this?").first.click()
    page.get_by_text("Because:").first.wait_for()
    card = page.get_by_text("Because:").first.locator("xpath=ancestor::*[contains(@class,'rounded-xl')][1]")
    card.screenshot(path=str(OUT / "why-this.png"))
    print("  assets/screenshots/why-this.png")

    page.goto(f"{BASE}/search?q=coffee")
    page.locator("a[href^='/products/']").first.wait_for(timeout=15000)
    shot(page, "search.png")

    # Product page with similar items; add one in-stock item to the cart.
    page.locator("a[href^='/products/']").first.click()
    page.get_by_text("Similar items").wait_for(timeout=15000)
    page.locator("a[href^='/products/']").nth(3).wait_for(timeout=15000)
    shot(page, "product-similar.png")
    page.get_by_role("button", name="Add to cart").click()
    page.wait_for_timeout(1000)

    page.goto(f"{BASE}/cart")
    page.get_by_role("link", name="Checkout").wait_for(timeout=10000)
    shot(page, "cart.png")
    page.get_by_role("link", name="Checkout").click()
    page.locator("textarea").wait_for()
    page.locator("textarea").fill("Jl. Sudirman No. 1, Jakarta")
    shot(page, "checkout.png")
    page.get_by_role("button", name="Place order").click()
    page.wait_for_url("**/orders", timeout=15000)
    page.get_by_text("Order #").first.wait_for()
    shot(page, "orders.png")
    order_shop = page.get_by_text("Order #").first.inner_text().split("Store ")[-1].strip()

    page.goto(f"{BASE}/how-it-works")
    page.get_by_text("Your behaviour").wait_for()
    page.wait_for_timeout(3500)  # one poll cycle so the latest events are shown
    shot(page, "how-it-works-events.png", selector="section:has-text('Your behaviour')")
    shot(page, "how-it-works-model.png", selector="section:has-text('The model')")

    # Seller of the item just ordered: shop N is owned by sellerN@demo.test.
    login(page, f"seller{order_shop}@demo.test")
    page.goto(f"{BASE}/seller")
    page.get_by_text("Seller Center").first.wait_for()
    page.locator("table").first.wait_for(timeout=10000)
    shot(page, "seller-products.png")
    page.get_by_role("button", name="orders", exact=True).click()  # tab label is CSS-capitalised
    page.get_by_text("Mark shipped").first.wait_for(timeout=10000)
    shot(page, "seller-orders.png")

    # Admin retrain
    login(page, "admin@demo.test")
    page.goto(f"{BASE}/how-it-works")
    btn = page.get_by_role("button", name="Retrain now")
    btn.wait_for(timeout=15000)
    btn.click()
    page.get_by_role("button", name="Training…").wait_for(timeout=5000)
    page.get_by_role("button", name="Retrain now").wait_for(timeout=180000)  # back to idle = done
    page.wait_for_timeout(1000)
    shot(page, "admin-retrain.png", selector="section:has(button:has-text('Retrain now'))")


def capture_fallback(page: Page) -> None:
    login(page, "user43@demo.test")
    page.goto(f"{BASE}/")
    page.get_by_text("Recommender offline").wait_for(timeout=20000)
    page.locator("a[href^='/products/']").first.wait_for()
    shot(page, "fallback-banner.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fallback"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
        try:
            page.goto(f"{BASE}/", timeout=10000)
        except Exception:
            sys.exit(f"app not reachable at {BASE}: start recsys serve, cargo run and npm run dev")
        if args.only == "fallback":
            capture_fallback(page)
        else:
            capture_all(page)
        browser.close()


if __name__ == "__main__":
    main()
