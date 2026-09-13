"""Focused tests for crawler URL safety and discovery helpers."""

import unittest
from unittest.mock import patch

from crawler import (
    FetchedPage,
    classify_page_type,
    crawl_website,
    filter_sitemap_urls,
    is_disallowed_link,
    is_safe_public_url,
    is_same_domain,
    normalize_url,
)


class FakeSession:
    def close(self) -> None:
        return None


class FakeBackend:
    @staticmethod
    def Session() -> FakeSession:
        return FakeSession()


def make_fetched_page(
    url: str,
    page_type: str,
    content: str,
    *,
    script_navigation_detected: bool = False,
) -> FetchedPage:
    return FetchedPage(
        page={
            "url": url,
            "title": "Test page",
            "type": page_type,
            "content": content,
            "description": "",
            "image_alts": [],
            "diagnostics": {
                "visible_text_chars": len(content),
                "anchor_count": 0,
                "iframe_count": 0,
                "script_count": 1 if script_navigation_detected else 0,
                "script_navigation_detected": script_navigation_detected,
                "app_shell_detected": False,
            },
        },
        soup=object(),
    )


class CrawlerHelperTests(unittest.TestCase):
    def test_normalize_url(self) -> None:
        self.assertEqual(normalize_url("example.com"), "https://example.com/")
        self.assertEqual(
            normalize_url("../products/#catalog", "https://example.com/company/about/"),
            "https://example.com/company/products/",
        )
        self.assertEqual(
            normalize_url("HTTPS://WWW.EXAMPLE.COM:443/about#team"),
            "https://www.example.com/about",
        )

    def test_same_domain_allows_www_but_not_subdomains(self) -> None:
        self.assertTrue(
            is_same_domain("https://example.com", "https://www.example.com/products")
        )
        self.assertFalse(
            is_same_domain("https://example.com", "https://shop.example.com/products")
        )
        self.assertFalse(
            is_same_domain("https://example.com", "https://example.org/products")
        )

    def test_localhost_and_private_addresses_are_blocked(self) -> None:
        blocked_urls = (
            "http://localhost",
            "http://127.0.0.1",
            "http://10.0.0.5",
            "http://172.16.0.5",
            "http://192.168.1.10",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]",
        )
        for url in blocked_urls:
            with self.subTest(url=url):
                self.assertFalse(is_safe_public_url(url, resolve_dns=False))

        self.assertTrue(is_safe_public_url("https://93.184.216.34", resolve_dns=False))
        self.assertTrue(is_safe_public_url("https://example.com", resolve_dns=False))

    def test_file_and_non_web_links_are_filtered(self) -> None:
        blocked_links = (
            "mailto:team@example.com",
            "tel:+123456789",
            "javascript:void(0)",
            "#products",
            "/downloads/catalog.pdf?download=1",
            "/assets/product-image.webp",
            "/media/company-video.mp4#play",
        )
        for link in blocked_links:
            with self.subTest(link=link):
                self.assertTrue(is_disallowed_link(link))

        self.assertFalse(is_disallowed_link("/products/industrial-software"))

    def test_page_type_classification(self) -> None:
        cases = {
            "https://example.com/products": "product",
            "https://example.com/solutions/industry": "solution",
            "https://example.com/services": "service",
            "https://example.com/business": "business",
            "https://example.com/company": "about",
        }
        for url, expected_type in cases.items():
            link_text = "Company profile" if expected_type == "about" else ""
            with self.subTest(url=url):
                self.assertEqual(classify_page_type(url, link_text), expected_type)

        self.assertEqual(
            classify_page_type("https://example.com/corporate", "公司介绍"),
            "about",
        )

    def test_homepage_with_little_content_returns_limited_coverage(self) -> None:
        homepage = make_fetched_page(
            "https://example.com/",
            "homepage",
            "Short company landing page",
            script_navigation_detected=True,
        )

        with (
            patch("crawler._load_http_backend", return_value=(FakeBackend, False)),
            patch("crawler._fetch_page", return_value=(homepage, None)),
            patch("crawler._discover_links", return_value=[]),
            patch("crawler._discover_fallback_candidates", return_value=[]),
        ):
            result = crawl_website("https://example.com")

        self.assertEqual(result["status"], "partial")
        warning_types = {warning["type"] for warning in result["warnings"]}
        self.assertIn("limited_coverage", warning_types)
        self.assertIn("possible_js_rendering_required", warning_types)

    def test_multiple_business_pages_return_success(self) -> None:
        homepage = make_fetched_page(
            "https://example.com/",
            "homepage",
            "Company homepage content " * 20,
        )
        product_page = make_fetched_page(
            "https://example.com/products/",
            "product",
            "Industrial product information " * 20,
        )
        about_page = make_fetched_page(
            "https://example.com/about/",
            "about",
            "Company background information " * 20,
        )
        homepage_candidates = [
            {
                "url": "https://example.com/products/",
                "type": "product",
                "score": 6,
                "source": "anchor",
            },
            {
                "url": "https://example.com/about/",
                "type": "about",
                "score": 3,
                "source": "anchor",
            },
        ]

        with (
            patch("crawler._load_http_backend", return_value=(FakeBackend, False)),
            patch(
                "crawler._fetch_page",
                side_effect=[
                    (homepage, None),
                    (product_page, None),
                    (about_page, None),
                ],
            ),
            patch(
                "crawler._discover_links",
                side_effect=[homepage_candidates, [], []],
            ),
        ):
            result = crawl_website("https://example.com")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(result["warnings"], [])

    def test_sitemap_candidates_still_require_url_safety(self) -> None:
        unsafe_candidates = filter_sitemap_urls(
            ["http://127.0.0.1/products/"],
            "http://127.0.0.1/",
        )
        self.assertEqual(unsafe_candidates, [])

        safe_candidates = filter_sitemap_urls(
            [
                "https://example.com/products/",
                "https://external.example.org/products/",
            ],
            "https://example.com/",
        )
        self.assertEqual(len(safe_candidates), 1)
        self.assertEqual(safe_candidates[0]["type"], "product")


if __name__ == "__main__":
    unittest.main()
