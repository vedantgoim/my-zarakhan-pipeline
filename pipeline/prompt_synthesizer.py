import os
import json
import re
import io
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image
from pydantic import BaseModel, Field

from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    AVATAR_REFERENCE_PATH,
    ZARA_KHAN_PERSONA
)

class ModelPromptBody(BaseModel):
    """Structured Pydantic schema for model prompt payloads."""
    positive_prompt: str = Field(description="Full, descriptive photorealistic diffusion prompt for the scene")
    negative_prompt: str = Field(description="Negative prompt avoiding stylization, blur, bad anatomy, and artifacts")
    scene_composition: str = Field(description="Pose, camera framing, background, and props derived from the visual reference")
    character_styling: str = Field(description="Zara Khan styling, Awadhi attire, and jewelry details")
    lighting_atmosphere: str = Field(description="Lighting, mood, and atmospheric color palette")
    overrides_applied: str = Field(description="Explicit confirmation of strict note overrides applied from the calendar")

SYSTEM_PROMPT = f"""
You are an expert AI prompt engineer specializing in photorealistic image generation workflows for cultural influencer campaigns.
Your task is to synthesize highly detailed, atmospheric image diffusion prompts for a social media content calendar featuring the character "Zara Khan".

---
### CHARACTER SPECIFICATION:
{ZARA_KHAN_PERSONA}
---

### KEY OBJECTIVES:
1. GROUNDING IN REFERENCE IMAGES:
   - Base Avatar Reference: Accurately ground Zara's facial structure, natural warm brown eyes, wavy dark hair, and authentic Awadhi elegance.
   - Visual Inspiration Reference: Extract composition, camera perspective, lighting, environmental props, and mood.
   - Strict Notes Overrides: TREAT NOTES AS ABSOLUTE OVERRIDES. If the note says "Change kurti colour to deep red", that color MUST strictly override the clothing in the reference image. If the note says "Change to perfume stall", the setting MUST reflect that.
   - Still-life / Object Shots: If Zara is not in the frame (e.g. cup of tea, antique clock, bangle stand), focus purely on authentic props, tactile textures, and rich Awadhi heritage without forcing a person.

2. STRUCTURED JSON OUTPUT:
   - Your response must adhere strictly to the ModelPromptBody JSON schema.
   - The positive prompt should be rich, textured, and atmospheric (e.g. specifying Chikankari needlework, natural golden hour sunlight, Fujifilm film tone, realistic depth of field).
"""

class PromptSynthesizer:
    """Synthesizes diffusion prompts using Gemini multimodal API with Pydantic structured output."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[PromptSynthesizer] Warning: Could not initialize google.genai client: {e}")

        # Load avatar reference image if available
        self.avatar_image = None
        if AVATAR_REFERENCE_PATH.exists():
            try:
                self.avatar_image = Image.open(AVATAR_REFERENCE_PATH)
            except Exception as e:
                print(f"[PromptSynthesizer] Could not open avatar image: {e}")

    def synthesize_prompt_body(
        self,
        asset_id: str,
        post_type: str,
        content_description: Optional[str],
        notes: Optional[str],
        caption: Optional[str] = None,
        reference_media_path: Optional[Path] = None,
        reference_image_url: Optional[str] = None
    ) -> Optional[ModelPromptBody]:
        """Synthesizes structured ModelPromptBody schema via Gemini Structured Outputs."""
        if not self.client:
            return None

        desc = (content_description or "").strip()
        notes_text = (notes or "").strip()

        contents = []

        # 1. Base Avatar Reference Image
        if self.avatar_image:
            contents.append("Base Character Avatar Reference (Zara Khan):")
            contents.append(self.avatar_image)

        # 2. Visual Inspiration Reference Image (Local file or Cloud URL)
        ref_loaded = False
        if reference_media_path and reference_media_path.exists():
            try:
                ref_img = Image.open(reference_media_path)
                contents.append("Visual Reference Image (composition, mood, lighting, props):")
                contents.append(ref_img)
                ref_loaded = True
            except Exception as e:
                print(f"[PromptSynthesizer] Could not load local reference image {reference_media_path}: {e}")

        if not ref_loaded and reference_image_url and reference_image_url.startswith("http"):
            try:
                req = urllib.request.Request(
                    reference_image_url,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    img_bytes = resp.read()
                    if len(img_bytes) > 500:
                        ref_img = Image.open(io.BytesIO(img_bytes))
                        contents.append("Visual Reference Image (composition, mood, lighting, props):")
                        contents.append(ref_img)
            except Exception as e:
                print(f"[PromptSynthesizer] Could not load reference image from URL {reference_image_url}: {e}")

        user_instruction = f"""
Asset ID: {asset_id}
Post Type: {post_type}
Content Description: {desc if desc else "N/A"}
Notes (STRICT OVERRIDES): {notes_text if notes_text else "None"}
Caption Context: {caption if caption else "None"}

Please analyze all provided references and generate the structured diffusion prompt body.
"""
        contents.append(user_instruction)

        # Try designated model (e.g. gemini-3.8-flash) with fallback to gemini-3.5-flash-lite on 503 spikes
        models_to_try = [self.model_name]
        if self.model_name != "gemini-3.5-flash-lite":
            models_to_try.append("gemini-3.5-flash-lite")

        for m in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=contents,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "response_schema": ModelPromptBody
                    }
                )
                body = ModelPromptBody.model_validate_json(response.text.strip())
                return body
            except Exception as e:
                print(f"[PromptSynthesizer] Call to {m} failed: {e}")
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    continue
                break

        return None

    def synthesize_prompt(
        self,
        asset_id: str,
        post_type: str,
        content_description: Optional[str],
        notes: Optional[str],
        caption: Optional[str] = None,
        reference_media_path: Optional[Path] = None,
        reference_image_url: Optional[str] = None
    ) -> Tuple[str, str]:
        """Synthesizes positive and negative prompts, returning tuple (pos, neg)."""
        body = self.synthesize_prompt_body(
            asset_id=asset_id,
            post_type=post_type,
            content_description=content_description,
            notes=notes,
            caption=caption,
            reference_media_path=reference_media_path,
            reference_image_url=reference_image_url
        )

        if body and body.positive_prompt:
            return body.positive_prompt, body.negative_prompt

        return self._fallback_prompt(asset_id, content_description or "", notes or "")

    def _fallback_prompt(self, asset_id: str, desc: str, notes: str) -> Tuple[str, str]:
        """Deterministic template fallback when offline."""
        prompt_parts = ["score_9, score_8_up, score_7_up, source_photo"]
        is_person = any(w in desc.lower() for w in ["zara", "girl", "woman", "reading", "applying", "wearing", "posing"])

        if is_person:
            prompt_parts.append("photo of Zara Khan, young Indian woman with Awadhi heritage, beautiful expressive brown eyes, natural wavy dark hair, authentic Indian beauty")
        else:
            prompt_parts.append("photo of atmospheric Indian heritage still-life")

        if desc:
            prompt_parts.append(desc)

        if notes:
            prompt_parts.append(f"incorporating: {notes}")

        prompt_parts.extend([
            "timeless Lucknow aesthetic",
            "intricate traditional Chikankari craftsmanship",
            "warm natural golden hour sunlight streaming through vintage wooden balcony",
            "nostalgic analog film quality, Fujifilm color palette, soft realistic bokeh, rich tactile textures, 8k resolution"
        ])

        positive = ", ".join(prompt_parts)
        negative = "score_4, score_5, score_6, source_anime, 3d, render, cartoon, illustration, drawing, studio lighting, softbox, professional photography, airbrushed, smooth plastic skin, porcelain doll, bad anatomy, deformed fingers, extra limbs, blurry"
        return positive, negative
