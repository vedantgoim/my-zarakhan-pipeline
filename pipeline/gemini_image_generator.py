import os
import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image

from .config import (
    GEMINI_API_KEY,
    GEMINI_IMAGE_MODEL,
    OUTPUTS_DIR,
    AVATAR_REFERENCE_PATH
)

class GeminiImageGenerator:
    """Generates images directly via Google Gemini Image API (gemini-3.1-flash-lite-image)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_IMAGE_MODEL
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiImageGenerator] Failed to initialize genai client: {e}")

    def is_available(self) -> bool:
        """Checks if Gemini client is initialized."""
        return self.client is not None

    def generate_image(
        self,
        asset_id: str,
        prompt: str,
        reference_image_path: Optional[Path] = None,
        reference_image_url: Optional[str] = None
    ) -> Tuple[bool, Optional[Path], str]:
        """
        Submits prompt to gemini-3.1-flash-lite-image and saves the resulting image asset.
        Returns: (success: bool, saved_path: Optional[Path], message: str)
        """
        if not self.client:
            return False, None, "Gemini API key is not configured."

        contents = []

        # If base avatar reference exists, include it for visual consistency
        if AVATAR_REFERENCE_PATH.exists():
            try:
                avatar_img = Image.open(AVATAR_REFERENCE_PATH)
                contents.append("Character Reference Sheet (Zara Khan):")
                contents.append(avatar_img)
            except Exception:
                pass

        # If reference image provided, include it
        if reference_image_path and reference_image_path.exists():
            try:
                ref_img = Image.open(reference_image_path)
                contents.append("Visual Reference (Composition, Lighting, Style):")
                contents.append(ref_img)
            except Exception:
                pass
        elif reference_image_url and reference_image_url.startswith("http"):
            try:
                import urllib.request
                req = urllib.request.Request(reference_image_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                    if len(data) > 500:
                        contents.append("Visual Reference (Composition, Lighting, Style):")
                        contents.append(Image.open(io.BytesIO(data)))
            except Exception:
                pass

        # Instruction prompt
        generation_prompt = f"""
Generate a high-resolution, photorealistic photograph based on the following creative prompt:
{prompt}
Ensure accurate Indian facial aesthetics consistent with the character reference, delicate authentic fabric embroidery, and warm natural lighting.
"""
        contents.append(generation_prompt.strip())

        try:
            print(f"[GeminiImageGenerator] Calling {self.model_name} for asset {asset_id}...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )

            # Search for inline image bytes in response parts
            for candidate in getattr(response, "candidates", []):
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []):
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        img_bytes = inline.data
                        target_file = self.output_dir / f"{asset_id}.png"
                        with open(target_file, "wb") as f:
                            f.write(img_bytes)
                        print(f"[GeminiImageGenerator] Image successfully generated: {target_file}")
                        return True, target_file, "Success"

            return False, None, "No image binary returned in Gemini API response."

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                user_friendly_err = (
                    f"Quota exceeded for {self.model_name}. "
                    "Image generation on Gemini requires Pay-As-You-Go billing enabled "
                    "in your Google AI Studio project (free tier has limit 0 for image models)."
                )
                print(f"[GeminiImageGenerator] [BILLING REQUIRED] {user_friendly_err}")
                return False, None, user_friendly_err

            print(f"[GeminiImageGenerator] API call failed: {e}")
            return False, None, f"Gemini Image API error: {err_msg}"
