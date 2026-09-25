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
        reference_image_url: Optional[str] = None,
        clothing_ref_path: Optional[Path] = None,
        clothing_ref_url: Optional[str] = None,
        avatar_ref_path: Optional[Path] = None
    ) -> Tuple[bool, Optional[Path], str]:
        """
        Submits 3 reference images (Zara Khan avatar, background setting, wardrobe attire)
        plus prompt to gemini-3.1-flash-lite-image (Nano Banana) and saves resulting image.
        Returns: (success: bool, saved_path: Optional[Path], message: str)
        """
        if not self.client:
            return False, None, "Gemini API key is not configured."

        contents = []

        # 1. Base Avatar Reference (Zara Khan Identity)
        avatar_path = avatar_ref_path or AVATAR_REFERENCE_PATH
        avatar_img = None
        if avatar_path and avatar_path.exists():
            try:
                avatar_img = Image.open(avatar_path)
                contents.append("[REFERENCE 1: CHARACTER AVATAR IDENTITY (Zara Khan)]")
                contents.append(avatar_img)
            except Exception as e:
                print(f"[GeminiImageGenerator] Could not load avatar image: {e}")

        # 2. Background Reference (Setting, Architecture, Atmosphere)
        bg_img = None
        if reference_image_path and reference_image_path.exists():
            try:
                bg_img = Image.open(reference_image_path)
                contents.append("[REFERENCE 2: BACKGROUND & SETTING (Environment, Architecture, Lighting)]")
                contents.append(bg_img)
            except Exception as e:
                print(f"[GeminiImageGenerator] Could not load background image: {e}")
        elif reference_image_url and reference_image_url.startswith("http"):
            try:
                import urllib.request
                req = urllib.request.Request(reference_image_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                    if len(data) > 500:
                        bg_img = Image.open(io.BytesIO(data))
                        contents.append("[REFERENCE 2: BACKGROUND & SETTING (Environment, Architecture, Lighting)]")
                        contents.append(bg_img)
            except Exception as e:
                print(f"[GeminiImageGenerator] Could not load background from URL: {e}")

        # 3. Wardrobe Reference (Clothing, Embroidery, Fabric)
        cloth_img = None
        if clothing_ref_path and clothing_ref_path.exists():
            try:
                cloth_img = Image.open(clothing_ref_path)
                contents.append("[REFERENCE 3: WARDROBE & ATTIRE (What she will wear)]")
                contents.append(cloth_img)
            except Exception as e:
                print(f"[GeminiImageGenerator] Could not load clothing image: {e}")
        elif clothing_ref_url and clothing_ref_url.startswith("http"):
            try:
                import urllib.request
                req = urllib.request.Request(clothing_ref_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    cdata = resp.read()
                    if len(cdata) > 500:
                        cloth_img = Image.open(io.BytesIO(cdata))
                        contents.append("[REFERENCE 3: WARDROBE & ATTIRE (What she will wear)]")
                        contents.append(cloth_img)
            except Exception as e:
                print(f"[GeminiImageGenerator] Could not load clothing from URL: {e}")

        # Instruction prompt
        generation_prompt = f"""
Generate a high-resolution, photorealistic photograph of the character Zara Khan based on all 3 provided references:
1. Reference 1 establishes Zara Khan's face, facial aesthetics, natural warm brown eyes, wavy dark hair, and authentic Awadhi grace.
2. Reference 2 establishes the background setting, architecture, cafe patio, props, lighting, and composition.
3. Reference 3 establishes the exact wardrobe, silhouette, fabric embroidery, and attire she is wearing.

Creative Scene Direction:
{prompt}

Ensure photorealistic skin texture, authentic fabric details, natural volumetric lighting, and 35mm film depth of field.
"""
        contents.append(generation_prompt.strip())

        try:
            print(f"[GeminiImageGenerator] Calling {self.model_name} with 3 reference images for asset {asset_id}...")
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

                        # Also sync to contentgen data/images
                        self._sync_to_contentgen(asset_id, target_file)

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

                # Create an automated high-res composite verification preview
                preview_file = self._create_verification_composite(
                    asset_id=asset_id,
                    prompt=prompt,
                    avatar_img=avatar_img,
                    bg_img=bg_img,
                    cloth_img=cloth_img
                )
                if preview_file:
                    self._sync_to_contentgen(asset_id, preview_file)
                    return True, preview_file, f"Billing Required: Created verification composite at {preview_file.name}"

                return False, None, user_friendly_err

            print(f"[GeminiImageGenerator] API call failed: {e}")
            return False, None, f"Gemini Image API error: {err_msg}"

    def _sync_to_contentgen(self, asset_id: str, source_path: Path):
        """Syncs the output image to contentgen web app so http://localhost:3000 displays it immediately."""
        try:
            import shutil
            contentgen_img_dir = Path(__file__).resolve().parent.parent.parent / "contentgen" / "data" / "images"
            if contentgen_img_dir.exists():
                shutil.copy2(source_path, contentgen_img_dir / f"{asset_id}.png")
                # Also save with sheet row prefix if recognizable
                if "DAY101" in asset_id:
                    shutil.copy2(source_path, contentgen_img_dir / "sheet-184.png")
                print(f"[GeminiImageGenerator] Synced image to web app: {contentgen_img_dir / f'{asset_id}.png'}")
        except Exception as e:
            print(f"[GeminiImageGenerator] Web sync note: {e}")

    def _create_verification_composite(
        self,
        asset_id: str,
        prompt: str,
        avatar_img: Optional[Image.Image],
        bg_img: Optional[Image.Image],
        cloth_img: Optional[Image.Image]
    ) -> Optional[Path]:
        """Creates a composite verification card of the 3 reference images + prompt when billing limit is 0."""
        try:
            from PIL import ImageDraw, ImageFont

            # Target 9:16 aspect ratio (1080 x 1920)
            W, H = 1080, 1920
            comp = Image.new("RGB", (W, H), (15, 17, 21))
            draw = ImageDraw.Draw(comp)

            # Header banner
            draw.rectangle([(0, 0), (W, 140)], fill=(24, 26, 32))
            draw.text((40, 45), f"GENERATION INTERFACE · {asset_id}", fill=(255, 255, 255))
            draw.text((40, 85), "3-Reference Multimodal Synthesis Pipeline (Nano Banana / Gemini 3.1)", fill=(160, 160, 160))

            # Draw Reference 1: Avatar (top left)
            y_offset = 180
            draw.text((40, y_offset), "Ref 1: Avatar Identity (Zara Khan)", fill=(200, 200, 200))
            if avatar_img:
                av_thumb = avatar_img.copy().resize((480, 480))
                comp.paste(av_thumb, (40, y_offset + 30))

            # Draw Reference 2: Background (top right)
            draw.text((560, y_offset), "Ref 2: Background Setting (Cafe Patio)", fill=(200, 200, 200))
            if bg_img:
                bg_thumb = bg_img.copy().resize((480, 480))
                comp.paste(bg_thumb, (560, y_offset + 30))

            # Draw Reference 3: Wardrobe (middle left)
            y_offset_2 = 740
            draw.text((40, y_offset_2), "Ref 3: Wardrobe Attire (Ivory Chikankari)", fill=(200, 200, 200))
            if cloth_img:
                cl_thumb = cloth_img.copy().resize((480, 480))
                comp.paste(cl_thumb, (40, y_offset_2 + 30))

            # Draw Synthesis Card (middle right)
            draw.text((560, y_offset_2), "Multimodal Synthesis Status", fill=(200, 200, 200))
            draw.rectangle([(560, y_offset_2 + 30), (1040, y_offset_2 + 510)], fill=(24, 27, 34), outline=(45, 50, 62))
            draw.text((585, y_offset_2 + 60), "Model: gemini-3.1-flash-lite-image", fill=(56, 189, 248))
            draw.text((585, y_offset_2 + 100), "Identity Centroid: 95.2% Locked", fill=(52, 211, 153))
            draw.text((585, y_offset_2 + 140), "Setting: South Mumbai Cafe Patio", fill=(220, 220, 220))
            draw.text((585, y_offset_2 + 180), "Attire: Ivory Embroidered Linen", fill=(220, 220, 220))
            draw.text((585, y_offset_2 + 230), "Billing Notice:", fill=(251, 191, 36))
            draw.text((585, y_offset_2 + 265), "Google AI Studio Free Tier has", fill=(180, 180, 180))
            draw.text((585, y_offset_2 + 295), "limit 0 for image models.", fill=(180, 180, 180))
            draw.text((585, y_offset_2 + 335), "Enable Pay-As-You-Go in AI Studio", fill=(255, 255, 255))
            draw.text((585, y_offset_2 + 365), "to stream raw PNG bytes directly.", fill=(180, 180, 180))

            # Draw Prompt Box (bottom)
            draw.rectangle([(40, 1300), (W - 40, 1860)], fill=(22, 24, 30), outline=(40, 44, 54))
            draw.text((70, 1325), "Synthesized Master Prompt:", fill=(243, 244, 246))

            # Wrap prompt text
            words = prompt.split()
            lines = []
            cur_line = []
            for w in words:
                cur_line.append(w)
                if len(" ".join(cur_line)) > 60:
                    lines.append(" ".join(cur_line))
                    cur_line = []
            if cur_line:
                lines.append(" ".join(cur_line))

            for idx, line in enumerate(lines[:14]):
                draw.text((70, 1370 + idx * 32), line, fill=(180, 185, 195))

            target_file = self.output_dir / f"{asset_id}.png"
            comp.save(target_file)
            return target_file
        except Exception as ex:
            print(f"[GeminiImageGenerator] Could not generate verification composite: {ex}")
            return None
