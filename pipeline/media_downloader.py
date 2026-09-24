import urllib.request
import urllib.parse
import re
import html
from pathlib import Path
from typing import Optional, Tuple
from .config import REFERENCES_DIR

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

class MediaDownloader:
    """Resolves and downloads reference images from Pinterest and Instagram."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or REFERENCES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_cached_path(self, asset_id: str) -> Optional[Path]:
        """Checks if a valid downloaded reference already exists."""
        for ext in [".jpg", ".png", ".webp", ".jpeg"]:
            p = self.output_dir / f"{asset_id}{ext}"
            if p.exists() and p.stat().st_size > 1024:
                return p
        return None

    def resolve_pinterest_image_url(self, pin_url: str) -> Optional[str]:
        """Follows pin.it redirect and parses the pin page for highest quality image URL."""
        try:
            req = urllib.request.Request(pin_url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                content = resp.read().decode('utf-8', errors='ignore')

                # Check og:image meta tag
                m = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', content)
                if not m:
                    m = re.search(r'<meta\s+name=["\']og:image["\']\s+content=["\']([^"\']+)["\']', content)
                if m:
                    img_url = m.group(1)
                    # Try upgrading to 736x or originals if low-res
                    img_url = re.sub(r'/(?:236x|474x|564x)/', '/736x/', img_url)
                    return img_url

                # Check regex matches for i.pinimg.com
                pinimgs = re.findall(r'https://i\.pinimg\.com/(?:originals|736x|564x|474x)/[^\s"\'<>()]+', content)
                if pinimgs:
                    for img in pinimgs:
                        if '/originals/' in img or '/736x/' in img:
                            return img
                    return pinimgs[0]

                # Fallback any pinimg
                pinimgs_any = re.findall(r'https://i\.pinimg\.com/[^\s"\'<>()]+', content)
                if pinimgs_any:
                    return pinimgs_any[0]

        except Exception as e:
            print(f"[MediaDownloader] Error resolving Pinterest URL '{pin_url}': {e}")
        return None

    def resolve_instagram_image_url(self, ig_url: str) -> Optional[str]:
        """Extracts high quality image URL from Instagram post embeds."""
        try:
            # Extract post shortcode: /p/CODE/ or /reel/CODE/
            match = re.search(r'/(?:p|reel)/([A-Za-z0-9_-]+)', ig_url)
            if match:
                shortcode = match.group(1)
                embed_url = f"https://www.instagram.com/p/{shortcode}/embed/captioned/"
                req = urllib.request.Request(embed_url, headers=DEFAULT_HEADERS)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    content = resp.read().decode('utf-8', errors='ignore')

                    # Look for EmbeddedMediaImage
                    img_m = re.search(r'<img\s+class="EmbeddedMediaImage"[^>]+src="([^"]+)"', content)
                    if img_m:
                        return html.unescape(img_m.group(1))

                    # Look for any cdninstagram image URL
                    cdn_matches = re.findall(r'https://[^\s"\'<>]+\.cdninstagram\.com/[^\s"\'<>]+', content)
                    if cdn_matches:
                        return html.unescape(cdn_matches[0])

            # Fallback to direct page parse
            req = urllib.request.Request(ig_url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                content = resp.read().decode('utf-8', errors='ignore')
                m = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', content)
                if m:
                    return html.unescape(m.group(1))
        except Exception as e:
            print(f"[MediaDownloader] Error resolving Instagram URL '{ig_url}': {e}")
        return None

    def download_image(self, image_url: str, target_path: Path) -> bool:
        """Downloads image bytes and saves to target path."""
        try:
            req = urllib.request.Request(image_url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) > 500:
                    with open(target_path, 'wb') as f:
                        f.write(data)
                    return True
        except Exception as e:
            print(f"[MediaDownloader] Download error from '{image_url}': {e}")
        return False

    def resolve_direct_url(self, url: Optional[str]) -> Optional[str]:
        """Resolves Pinterest shortened URLs or Instagram embeds to direct high-res image CDN links."""
        if not url or not str(url).strip().startswith("http"):
            return None
        url = str(url).strip()
        if "pin.it" in url or "pinterest.com" in url:
            return self.resolve_pinterest_image_url(url)
        elif "instagram.com" in url:
            return self.resolve_instagram_image_url(url)
        elif any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            return url
        return url

    def fetch_image_bytes(self, image_url: str) -> Optional[bytes]:
        """Downloads raw image bytes in-memory for cloud / serverless streaming."""
        try:
            req = urllib.request.Request(image_url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) > 500:
                    return data
        except Exception as e:
            print(f"[MediaDownloader] Error fetching image bytes from '{image_url}': {e}")
        return None

    def ingest_reference(self, asset_id: str, url: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
        """Resolves and downloads reference asset. Returns (local_file_path, resolved_direct_url)."""
        if not url or not str(url).strip().startswith("http"):
            return None, None

        url = str(url).strip()
        resolved_img_url = self.resolve_direct_url(url)

        # Check existing cache
        cached = self.get_cached_path(asset_id)
        if cached:
            return cached, resolved_img_url

        target_file = self.output_dir / f"{asset_id}.jpg"

        if resolved_img_url:
            success = self.download_image(resolved_img_url, target_file)
            if success:
                return target_file, resolved_img_url

        return None, resolved_img_url
