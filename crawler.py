"""Safe, bounded website crawling for company research.

The crawler keeps the useful discovery heuristics from the original script while
returning structured page and error records. It does not call an LLM.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

DEFAULT_MAX_PAGES = 10
DEFAULT_TIMEOUT_SECONDS = 12
DEFAULT_RETRIES = 2
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2_000_000
MAX_CONTENT_CHARS = 6_000
MAX_DISCOVERY_ATTEMPTS_FACTOR = 3
LIMITED_CONTENT_CHARS = 250
MIN_ANCHOR_CANDIDATES_BEFORE_FALLBACK = 2
MAX_SITEMAP_FILES = 4

COMMON_BUSINESS_PATHS = (
    "/products/",
    "/product/",
    "/product.html",
    "/products.html",
    "/pro/",
    "/product-center/",
    "/product_center/",
    "/cp/",
    "/chanpin/",
    "/solutions/",
    "/services/",
    "/business/",
    "/about/",
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

PAGE_KEYWORDS = {
    "product": (
        "products",
        "product",
        "product center",
        "产品中心",
        "产品",
        "设备",
        "机型",
        "产品系列",
    ),
    "solution": ("solutions", "solution", "解决方案", "行业方案"),
    "service": ("services", "service", "服务", "服务项目"),
    "business": ("business", "业务", "主营", "业务范围"),
    "about": (
        "about us",
        "about",
        "company profile",
        "公司介绍",
        "企业介绍",
        "关于我们",
        "关于",
        "简介",
        "概况",
    ),
}

PAGE_TYPE_SCORES = {
    "product": 5,
    "solution": 4,
    "service": 4,
    "business": 4,
    "about": 2,
}

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
    "instance-data",
}

BLOCKED_FILE_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".svg",
    ".tar",
    ".wav",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}

ANTIBOT_MARKERS = (
    "attention required",
    "checking your browser",
    "enable javascript and cookies",
    "verify you are human",
    "cf-chl-",
    "cf-ray",
    "captcha",
)

RETRIABLE_STATUS_CODES = {429, 502, 503}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}

ProgressCallback = Callable[[str, str], None]


def _load_http_backend() -> tuple[Any, bool]:
    """Load curl_cffi when available, with requests as a local fallback."""
    try:
        from curl_cffi import requests as request_backend

        return request_backend, True
    except ImportError:
        try:
            import requests as request_backend

            return request_backend, False
        except ImportError as exc:
            raise CrawlFailure(
                "dependency_error",
                "Website crawling dependencies are not installed.",
                "",
                "Install curl-cffi (preferred) or requests.",
            ) from exc


class CrawlFailure(Exception):
    """A crawler error that can be safely converted to a UI-facing record."""

    def __init__(
        self,
        error_type: str,
        message: str,
        url: str,
        technical_details: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.url = url
        self.technical_details = technical_details
        self.status_code = status_code

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.error_type,
            "message": self.message,
            "url": self.url,
            "status_code": self.status_code,
            "technical_details": self.technical_details,
        }


@dataclass
class FetchedPage:
    page: dict[str, Any]
    soup: Any


def normalize_url(value: str, base_url: str | None = None) -> str:
    """Return an absolute HTTP(S) URL without a fragment."""
    raw_value = str(value or "").strip()
    if not raw_value:
        raise ValueError("URL is empty")

    if base_url:
        absolute = urljoin(base_url, raw_value)
    else:
        absolute = raw_value
        if "://" not in absolute:
            absolute = f"https://{absolute}"

    parsed = urlsplit(absolute)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed")

    hostname = parsed.hostname.lower().rstrip(".")
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc

    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _normalized_hostname(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    return hostname[4:] if hostname.startswith("www.") else hostname


def is_same_domain(base_url: str, target_url: str) -> bool:
    """Treat the www and non-www forms of one hostname as the same site."""
    try:
        return bool(_normalized_hostname(base_url)) and (
            _normalized_hostname(base_url) == _normalized_hostname(target_url)
        )
    except ValueError:
        return False


def is_disallowed_link(href: str | None) -> bool:
    """Filter non-web actions and downloadable/media files."""
    if not href or not str(href).strip():
        return True

    candidate = str(href).strip()
    lowered = candidate.lower()
    if lowered.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
        return True

    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
        return True

    path = parsed.path.lower().rstrip("/")
    return any(path.endswith(extension) for extension in BLOCKED_FILE_EXTENSIONS)


def classify_page_type(url: str, link_text: str = "") -> str:
    """Classify relevant company pages using URL and anchor-text keywords."""
    searchable = unquote(
        f"{urlsplit(url).path} {urlsplit(url).query} {link_text}"
    ).lower()
    for page_type in ("product", "solution", "service", "business", "about"):
        if any(keyword in searchable for keyword in PAGE_KEYWORDS[page_type]):
            return page_type
    return "other"


def _is_blocked_hostname(hostname: str) -> bool:
    lowered = hostname.lower().rstrip(".")
    return (
        lowered in BLOCKED_HOSTNAMES
        or lowered.endswith(".localhost")
        or lowered.endswith(".local")
        or lowered.endswith(".internal")
    )


def _is_disallowed_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    return not ip.is_global


def _ensure_safe_public_url(url: str, resolve_dns: bool = True) -> None:
    """Reject local/private destinations before a network request is made."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise CrawlFailure("unsafe_url", "The website URL is invalid.", url, str(exc)) from exc

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise CrawlFailure(
            "unsafe_url",
            "Only public HTTP and HTTPS website URLs are allowed.",
            url,
        )

    if parsed.username or parsed.password:
        raise CrawlFailure(
            "unsafe_url",
            "Website URLs containing embedded credentials are not allowed.",
            url,
        )

    hostname = parsed.hostname.lower().rstrip(".")
    if _is_blocked_hostname(hostname):
        raise CrawlFailure(
            "unsafe_url",
            "Local and internal network addresses are not allowed.",
            url,
        )

    try:
        literal_ip = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literal_ip = None

    if literal_ip is not None and not literal_ip.is_global:
        raise CrawlFailure(
            "unsafe_url",
            "Private, loopback, and link-local IP addresses are not allowed.",
            url,
        )

    if not resolve_dns:
        return

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port or (443 if parsed.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise CrawlFailure(
            "dns_error",
            "The website hostname could not be resolved.",
            url,
            str(exc),
        ) from exc

    resolved_ips = {item[4][0] for item in addresses if item[4]}
    if not resolved_ips:
        raise CrawlFailure(
            "dns_error",
            "The website hostname did not resolve to an address.",
            url,
        )

    if any(_is_disallowed_ip(address) for address in resolved_ips):
        raise CrawlFailure(
            "unsafe_url",
            "The website resolves to a private or non-public network address.",
            url,
            f"Blocked resolved address: {', '.join(sorted(resolved_ips))}",
        )


def is_safe_public_url(url: str, resolve_dns: bool = False) -> bool:
    """Boolean helper for validation and focused unit tests."""
    try:
        normalized = normalize_url(url)
        _ensure_safe_public_url(normalized, resolve_dns=resolve_dns)
        return True
    except (CrawlFailure, ValueError):
        return False


def _error_for_status(status_code: int, url: str, body_text: str) -> CrawlFailure:
    if _looks_like_antibot(body_text):
        return CrawlFailure(
            "anti_bot",
            "The website blocked automated access or requested a browser verification.",
            url,
            f"HTTP {status_code}",
            status_code,
        )

    messages = {
        403: ("http_403", "The website refused access to the crawler."),
        404: ("http_404", "The requested page was not found."),
        429: ("http_429", "The website rate-limited the crawler."),
        502: ("http_502", "The website returned a temporary gateway error."),
        503: ("http_503", "The website is temporarily unavailable."),
    }
    error_type, message = messages.get(
        status_code,
        ("http_error", f"The website returned HTTP status {status_code}."),
    )
    return CrawlFailure(error_type, message, url, f"HTTP {status_code}", status_code)


def _looks_like_antibot(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in ANTIBOT_MARKERS):
        return True
    return "cloudflare" in lowered and (
        "just a moment" in lowered or "access denied" in lowered
    )


def _classify_request_exception(exc: Exception, url: str) -> CrawlFailure:
    class_name = exc.__class__.__name__.lower()
    details = str(exc)[:500]
    lowered = details.lower()

    if "ssl" in class_name or "certificate" in lowered or "ssl" in lowered:
        return CrawlFailure(
            "ssl_error",
            "The website's SSL certificate could not be verified.",
            url,
            details,
        )
    if "timeout" in class_name or "timed out" in lowered:
        return CrawlFailure(
            "timeout",
            "The website took too long to respond.",
            url,
            details,
        )
    if any(
        marker in lowered
        for marker in ("could not resolve", "name resolution", "getaddrinfo", "no such host")
    ):
        return CrawlFailure(
            "dns_error",
            "The website hostname could not be resolved.",
            url,
            details,
        )
    return CrawlFailure(
        "connection_error",
        "The website could not be reached.",
        url,
        details,
    )


def _request_once(
    session: Any,
    url: str,
    timeout: int,
    using_curl_cffi: bool,
) -> Any:
    request_options = {
        "headers": REQUEST_HEADERS,
        "timeout": timeout,
        "verify": True,
        "allow_redirects": False,
    }
    if using_curl_cffi:
        request_options["impersonate"] = "chrome120"
    return session.get(url, **request_options)


def _request_with_safe_redirects(
    session: Any,
    url: str,
    allowed_domain_url: str,
    timeout: int,
    using_curl_cffi: bool,
) -> tuple[Any, str]:
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        _ensure_safe_public_url(current_url, resolve_dns=True)
        response = _request_once(session, current_url, timeout, using_curl_cffi)

        if response.status_code not in REDIRECT_STATUS_CODES:
            return response, current_url

        location = response.headers.get("Location")
        if not location:
            raise CrawlFailure(
                "redirect_error",
                "The website returned an invalid redirect.",
                current_url,
                f"HTTP {response.status_code} without a Location header",
                response.status_code,
            )
        if redirect_count >= MAX_REDIRECTS:
            raise CrawlFailure(
                "redirect_error",
                "The website redirected too many times.",
                current_url,
            )

        next_url = normalize_url(location, current_url)
        _ensure_safe_public_url(next_url, resolve_dns=True)
        if not is_same_domain(allowed_domain_url, next_url):
            raise CrawlFailure(
                "unsafe_redirect",
                "The website redirected outside the original company domain.",
                next_url,
            )
        current_url = next_url

    raise CrawlFailure("redirect_error", "The website redirected too many times.", current_url)


def _extract_page(response: Any, final_url: str, page_type: str) -> FetchedPage:
    from bs4 import BeautifulSoup

    content = response.content or b""
    if len(content) > MAX_RESPONSE_BYTES:
        raise CrawlFailure(
            "response_too_large",
            "The webpage is too large to process safely.",
            final_url,
            f"Response size: {len(content)} bytes",
            response.status_code,
        )

    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type and "html" not in content_type and "xhtml" not in content_type:
        raise CrawlFailure(
            "unsupported_content",
            "The URL did not return an HTML webpage.",
            final_url,
            f"Content-Type: {content_type}",
            response.status_code,
        )

    raw_html = response.text or ""
    if _looks_like_antibot(raw_html[:20_000]):
        raise CrawlFailure(
            "anti_bot",
            "The website blocked automated access or requested a browser verification.",
            final_url,
            f"HTTP {response.status_code}",
            response.status_code,
        )

    soup = BeautifulSoup(content, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    description_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    description = (
        description_tag.get("content", "").strip() if description_tag else ""
    )

    image_alts: list[str] = []
    seen_alts: set[str] = set()
    for image in soup.find_all("img"):
        alt_text = re.sub(r"\s+", " ", image.get("alt", "")).strip()
        if len(alt_text) > 1 and alt_text not in seen_alts:
            image_alts.append(alt_text)
            seen_alts.add(alt_text)

    text_soup = BeautifulSoup(content, "html.parser")
    for tag in text_soup(["script", "style", "noscript", "svg", "footer", "nav", "header"]):
        tag.decompose()

    text_content = re.sub(r"\s+", " ", text_soup.get_text(" ", strip=True)).strip()
    anchor_count = len(soup.find_all("a"))
    iframe_count = len(soup.find_all("iframe"))
    script_count = len(soup.find_all("script"))
    script_text = " ".join(script.get_text(" ", strip=True) for script in soup.find_all("script"))
    script_navigation_detected = bool(
        re.search(
            r"(?:window\.)?location(?:\.href)?\s*=|window\.open\s*\(",
            script_text,
            re.I,
        )
    )
    has_app_shell = bool(
        soup.find(id=re.compile(r"^(app|root|__next)$", re.I))
        or soup.find(attrs={"data-reactroot": True})
    )

    if len(text_content) < 40:
        if has_app_shell or script_count >= 5:
            raise CrawlFailure(
                "js_only",
                "The webpage appears to require JavaScript rendering.",
                final_url,
                f"Visible text length: {len(text_content)}; scripts: {script_count}",
                response.status_code,
            )
        raise CrawlFailure(
            "empty_page",
            "The webpage did not contain enough readable text.",
            final_url,
            f"Visible text length: {len(text_content)}",
            response.status_code,
        )

    resolved_type = page_type if page_type == "homepage" else classify_page_type(final_url)
    if resolved_type == "other" and page_type != "other":
        resolved_type = page_type

    page = {
        "url": final_url,
        "title": title or urlsplit(final_url).hostname or final_url,
        "type": resolved_type,
        "content": text_content[:MAX_CONTENT_CHARS],
        "description": description,
        "image_alts": image_alts[:30],
        "diagnostics": {
            "visible_text_chars": len(text_content),
            "anchor_count": anchor_count,
            "iframe_count": iframe_count,
            "script_count": script_count,
            "script_navigation_detected": script_navigation_detected,
            "app_shell_detected": has_app_shell,
        },
    }
    return FetchedPage(page=page, soup=soup)


def _fetch_page(
    session: Any,
    url: str,
    page_type: str,
    allowed_domain_url: str,
    timeout: int,
    retries: int,
    using_curl_cffi: bool,
) -> tuple[FetchedPage | None, dict[str, Any] | None]:
    last_error: CrawlFailure | None = None

    for attempt in range(retries + 1):
        try:
            response, final_url = _request_with_safe_redirects(
                session,
                url,
                allowed_domain_url,
                timeout,
                using_curl_cffi,
            )

            if response.status_code >= 400:
                body_preview = (response.text or "")[:20_000]
                failure = _error_for_status(response.status_code, final_url, body_preview)
                if response.status_code in RETRIABLE_STATUS_CODES and attempt < retries:
                    last_error = failure
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None, failure.as_dict()

            return _extract_page(response, final_url, page_type), None
        except CrawlFailure as exc:
            last_error = exc
            if exc.error_type in {"timeout", "connection_error"} and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None, exc.as_dict()
        except Exception as exc:  # Network backends expose slightly different classes.
            last_error = _classify_request_exception(exc, url)
            if last_error.error_type in {"timeout", "connection_error"} and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None, last_error.as_dict()

    fallback = last_error or CrawlFailure(
        "connection_error",
        "The website could not be reached.",
        url,
    )
    return None, fallback.as_dict()


def _link_score(url: str, link_text: str, page_type: str) -> int:
    if page_type in PAGE_TYPE_SCORES:
        score = PAGE_TYPE_SCORES[page_type]
        if link_text.strip():
            score += 1
        return score

    lowered_text = link_text.lower().strip()
    lowered_path = urlsplit(url).path.lower()
    if re.fullmatch(r"[a-z]{1,3}\s*\d{1,4}", lowered_text):
        return 3
    if any(token in lowered_path for token in ("/pro/", "/cp/", "productshow", "showproduct")):
        return 3
    if any(token in link_text for token in ("系列", "型号", "机型")):
        return 3
    return 0


def _discover_links(
    soup: Any,
    page_url: str,
    site_url: str,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    for anchor in soup.find_all("a"):
        link_text = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        href = anchor.get("href")

        if is_disallowed_link(href):
            continue

        try:
            target_url = normalize_url(str(href), page_url)
        except ValueError:
            continue

        if not is_same_domain(site_url, target_url):
            continue

        page_type = classify_page_type(target_url, link_text)
        score = _link_score(target_url, link_text, page_type)
        if score <= 0:
            continue

        key = target_url.rstrip("/")
        candidate = {
            "url": target_url,
            "type": page_type if page_type != "other" else "product",
            "score": score,
            "source": "anchor",
        }
        if key not in candidates or score > candidates[key]["score"]:
            candidates[key] = candidate

    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)


def _site_origin(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))


def _fetch_text_resource(
    session: Any,
    url: str,
    site_url: str,
    timeout: int,
    using_curl_cffi: bool,
) -> str | None:
    """Fetch an optional robots or sitemap resource without surfacing 404 noise."""
    try:
        response, final_url = _request_with_safe_redirects(
            session,
            url,
            site_url,
            timeout,
            using_curl_cffi,
        )
    except Exception:
        return None

    if response.status_code != 200:
        return None
    if not is_same_domain(site_url, final_url):
        return None

    content = response.content or b""
    if len(content) > MAX_RESPONSE_BYTES:
        return None
    return response.text or ""


def _robots_sitemap_urls(robots_text: str, site_url: str) -> list[str]:
    sitemap_urls: list[str] = []
    for line in robots_text.splitlines():
        match = re.match(r"^\s*sitemap\s*:\s*(\S+)", line, re.I)
        if not match:
            continue
        try:
            sitemap_url = normalize_url(match.group(1), site_url)
        except ValueError:
            continue
        if is_same_domain(site_url, sitemap_url) and is_safe_public_url(
            sitemap_url,
            resolve_dns=False,
        ):
            sitemap_urls.append(sitemap_url)
    return sitemap_urls


def _sitemap_locations(sitemap_text: str, sitemap_url: str) -> list[str]:
    locations: list[str] = []
    for raw_location in re.findall(
        r"<loc\b[^>]*>([\s\S]*?)</loc>",
        sitemap_text,
        re.I,
    ):
        decoded = html.unescape(re.sub(r"\s+", "", raw_location)).strip()
        if not decoded:
            continue
        try:
            locations.append(normalize_url(decoded, sitemap_url))
        except ValueError:
            continue
    return locations


def filter_sitemap_urls(urls: list[str], site_url: str) -> list[dict[str, Any]]:
    """Keep only safe, same-domain, business-relevant sitemap page URLs."""
    candidates: dict[str, dict[str, Any]] = {}
    for raw_url in urls:
        try:
            target_url = normalize_url(raw_url, site_url)
        except ValueError:
            continue
        if is_disallowed_link(target_url):
            continue
        if not is_same_domain(site_url, target_url):
            continue
        if not is_safe_public_url(target_url, resolve_dns=False):
            continue

        page_type = classify_page_type(target_url)
        if page_type == "other":
            continue
        key = target_url.rstrip("/")
        candidates[key] = {
            "url": target_url,
            "type": page_type,
            "score": PAGE_TYPE_SCORES[page_type] + 1,
            "source": "sitemap",
        }
    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)


def _discover_fallback_candidates(
    session: Any,
    site_url: str,
    timeout: int,
    using_curl_cffi: bool,
) -> list[dict[str, Any]]:
    """Use bounded sitemap, robots, and common-path discovery as a fallback."""
    origin = _site_origin(site_url)
    robots_url = normalize_url("/robots.txt", origin)
    robots_text = _fetch_text_resource(
        session,
        robots_url,
        site_url,
        timeout,
        using_curl_cffi,
    )

    sitemap_queue = [normalize_url("/sitemap.xml", origin)]
    if robots_text:
        sitemap_queue.extend(_robots_sitemap_urls(robots_text, site_url))

    sitemap_queue = list(dict.fromkeys(sitemap_queue))[:MAX_SITEMAP_FILES]
    visited_sitemaps: set[str] = set()
    sitemap_page_urls: list[str] = []

    while sitemap_queue and len(visited_sitemaps) < MAX_SITEMAP_FILES:
        sitemap_url = sitemap_queue.pop(0)
        sitemap_key = sitemap_url.rstrip("/")
        if sitemap_key in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_key)

        sitemap_text = _fetch_text_resource(
            session,
            sitemap_url,
            site_url,
            timeout,
            using_curl_cffi,
        )
        if not sitemap_text:
            continue

        for location in _sitemap_locations(sitemap_text, sitemap_url):
            if location.lower().split("?", 1)[0].endswith(".xml"):
                if (
                    len(visited_sitemaps) + len(sitemap_queue) < MAX_SITEMAP_FILES
                    and is_same_domain(site_url, location)
                    and is_safe_public_url(location, resolve_dns=False)
                ):
                    sitemap_queue.append(location)
                continue
            sitemap_page_urls.append(location)

    candidates = filter_sitemap_urls(sitemap_page_urls, site_url)
    existing = {candidate["url"].rstrip("/") for candidate in candidates}

    for path in COMMON_BUSINESS_PATHS:
        target_url = normalize_url(path, origin)
        key = target_url.rstrip("/")
        if key in existing or not is_safe_public_url(target_url, resolve_dns=False):
            continue
        page_type = classify_page_type(target_url)
        candidates.append(
            {
                "url": target_url,
                "type": page_type if page_type != "other" else "product",
                "score": max(PAGE_TYPE_SCORES.get(page_type, 2) - 1, 1),
                "source": "common_path",
            }
        )
        existing.add(key)

    return candidates


def _coverage_warnings(result: dict[str, Any]) -> list[dict[str, Any]]:
    homepage = result.get("homepage")
    pages = result.get("pages", [])
    if not homepage:
        return []

    total_content_chars = len(homepage.get("content", "")) + sum(
        len(page.get("content", "")) for page in pages
    )
    relevant_pages = [
        page
        for page in pages
        if page.get("type") in {"product", "solution", "service", "business", "about"}
    ]

    reasons: list[str] = []
    if not pages:
        reasons.append("only the homepage was crawled")
    if total_content_chars < LIMITED_CONTENT_CHARS:
        reasons.append(f"only {total_content_chars} readable characters were extracted")
    if not relevant_pages:
        reasons.append("no business-related subpages were discovered")

    if not reasons:
        return []

    warnings = [
        {
            "type": "limited_coverage",
            "message": (
                "Only limited website content could be discovered. "
                "Business information may be incomplete."
            ),
            "details": "; ".join(reasons),
        }
    ]

    diagnostics = homepage.get("diagnostics", {})
    if diagnostics.get("script_navigation_detected") or diagnostics.get(
        "app_shell_detected"
    ):
        warnings.append(
            {
                "type": "possible_js_rendering_required",
                "message": (
                    "The homepage may rely on JavaScript for navigation or content."
                ),
                "details": (
                    f"Scripts: {diagnostics.get('script_count', 0)}; "
                    f"anchors: {diagnostics.get('anchor_count', 0)}"
                ),
            }
        )

    return warnings


def _content_signature(content: str) -> str:
    """Create a lightweight signature to avoid counting duplicate fallback pages."""
    return re.sub(r"\s+", "", content).lower()[:2_000]


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, message)
    except Exception:
        return


def crawl_website(
    url: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Crawl a company homepage and a bounded set of relevant same-domain pages."""
    result: dict[str, Any] = {
        "status": "failed",
        "homepage": None,
        "pages": [],
        "errors": [],
        "warnings": [],
    }

    try:
        start_url = normalize_url(url)
    except ValueError as exc:
        result["errors"].append(
            CrawlFailure("unsafe_url", "Enter a valid HTTP or HTTPS website URL.", str(url), str(exc)).as_dict()
        )
        return result

    try:
        _ensure_safe_public_url(start_url, resolve_dns=False)
    except CrawlFailure as exc:
        result["errors"].append(exc.as_dict())
        return result

    max_pages = max(1, min(int(max_pages), DEFAULT_MAX_PAGES))
    timeout = max(3, int(timeout))
    retries = max(0, min(int(retries), 3))

    try:
        http_backend, using_curl_cffi = _load_http_backend()
    except CrawlFailure as exc:
        exc.url = start_url
        result["errors"].append(exc.as_dict())
        return result

    session = http_backend.Session()
    _emit_progress(progress_callback, "fetch_homepage", "Step 1/4: Fetching homepage")
    homepage_fetch, homepage_error = _fetch_page(
        session,
        start_url,
        "homepage",
        start_url,
        timeout,
        retries,
        using_curl_cffi,
    )

    if homepage_error:
        result["errors"].append(homepage_error)
        try:
            session.close()
        except Exception:
            pass
        return result

    assert homepage_fetch is not None
    homepage = homepage_fetch.page
    site_url = homepage["url"]
    result["homepage"] = homepage
    result["status"] = "success"

    _emit_progress(
        progress_callback,
        "discover_pages",
        "Step 2/4: Discovering relevant pages",
    )

    anchor_candidates = _discover_links(homepage_fetch.soup, site_url, site_url)
    queue = [{**candidate, "depth": 1} for candidate in anchor_candidates]

    if len(anchor_candidates) < MIN_ANCHOR_CANDIDATES_BEFORE_FALLBACK:
        fallback_candidates = _discover_fallback_candidates(
            session,
            site_url,
            timeout,
            using_curl_cffi,
        )
        queued_keys = {candidate["url"].rstrip("/") for candidate in queue}
        for candidate in fallback_candidates:
            if candidate["url"].rstrip("/") not in queued_keys:
                queue.append({**candidate, "depth": 1})
                queued_keys.add(candidate["url"].rstrip("/"))
    seen_urls = {site_url.rstrip("/")}
    seen_content = {_content_signature(homepage.get("content", ""))}
    queued_urls = {candidate["url"].rstrip("/") for candidate in queue}
    attempts = 0
    max_attempts = max_pages * MAX_DISCOVERY_ATTEMPTS_FACTOR

    while queue and (1 + len(result["pages"])) < max_pages and attempts < max_attempts:
        queue.sort(key=lambda item: item["score"], reverse=True)
        candidate = queue.pop(0)
        candidate_key = candidate["url"].rstrip("/")
        queued_urls.discard(candidate_key)
        if candidate_key in seen_urls:
            continue

        seen_urls.add(candidate_key)
        attempts += 1
        fetched, error = _fetch_page(
            session,
            candidate["url"],
            candidate["type"],
            site_url,
            timeout,
            retries,
            using_curl_cffi,
        )

        if error:
            if candidate.get("source") != "common_path":
                result["errors"].append(error)
            continue

        assert fetched is not None
        content_signature = _content_signature(fetched.page.get("content", ""))
        if content_signature and content_signature in seen_content:
            continue
        seen_content.add(content_signature)
        result["pages"].append(fetched.page)

        if candidate["depth"] >= 2:
            continue

        for discovered in _discover_links(fetched.soup, fetched.page["url"], site_url):
            discovered_key = discovered["url"].rstrip("/")
            if discovered_key in seen_urls or discovered_key in queued_urls:
                continue
            queue.append({**discovered, "depth": candidate["depth"] + 1})
            queued_urls.add(discovered_key)

    result["warnings"].extend(_coverage_warnings(result))
    if result["errors"] or result["warnings"]:
        result["status"] = "partial"

    try:
        session.close()
    except Exception:
        pass

    return result


__all__ = [
    "classify_page_type",
    "crawl_website",
    "filter_sitemap_urls",
    "is_disallowed_link",
    "is_safe_public_url",
    "is_same_domain",
    "normalize_url",
]
