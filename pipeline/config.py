import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Calendar paths
SOURCE_CALENDAR_PATH = ROOT_DIR / "Vinayak’s Copy of Zara Khan.xlsx"
OPTIMIZED_CALENDAR_PATH = ROOT_DIR / "Zara_Khan_Optimized_Calendar.xlsx"

# Directories
REFERENCES_DIR = ROOT_DIR / "references"
OUTPUTS_DIR = ROOT_DIR / "outputs"
AVATAR_REFERENCE_PATH = ROOT_DIR / "ZARAKHANREFERENCE" / "ZARA_KHAN.png"

REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ComfyUI Configuration (Local rendering fallback)
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_OUTPUT_DIR = Path("E:/Comfy-Desktop/ComfyUI-Shared/output")
COMFYUI_INPUT_DIR = Path("E:/Comfy-Desktop/ComfyUI-Shared/input")

# Cloud & Google Sheets Configuration
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_SHEETS_TAB_NAME = os.getenv("GOOGLE_SHEETS_TAB_NAME", "Zara Khan Calendar")

# CyberPony Model Weights & Settings
CYBERPONY_CHECKPOINT = "cyberrealisticPony_v180Coreshift.safetensors"
CYBERPONY_YOLO_MODEL = "bbox/face_yolov8m.pt"
CYBERPONY_STEPS = 30
CYBERPONY_CFG = 5.5
CYBERPONY_SAMPLER = "dpmpp_2m_sde_gpu"
CYBERPONY_SCHEDULER = "karras"
CYBERPONY_FACE_STEPS = 20
CYBERPONY_FACE_CFG = 5.0
CYBERPONY_FACE_DENOISE = 0.35

# Flux Klein 9B Model Weights & Settings (Local ComfyUI)
FLUX_KLEIN_UNET = "flux-2-klein-9b_int8_convrot.safetensors"
FLUX_KLEIN_CLIP = "qwen_3_8b_fp4mixed.safetensors"
FLUX_KLEIN_VAE = "FLUX.2-Klein-Base-9B-VAE.safetensors"
FLUX_KLEIN_STEPS = 20
FLUX_KLEIN_GUIDANCE = 3.5
FLUX_KLEIN_SAMPLER = "euler"

# Dynamic Aspect Ratios for Instagram
ASPECT_RATIOS = {
    "Story": {"width": 832, "height": 1472, "desc": "9:16 Portrait"},
    "Post": {"width": 896, "height": 1152, "desc": "4:5 Portrait"},
    "Carousel": {"width": 896, "height": 1152, "desc": "4:5 Portrait"},
    "Square": {"width": 1024, "height": 1024, "desc": "1:1 Square"}
}

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")

# Character Persona Definition
ZARA_KHAN_PERSONA = """
Character Identity: Zara Khan
Heritage & Background: Awadhi / Lucknowi heritage, timeless Indian cultural roots, classic Lucknow elegance, Tehzeeb and Adaab.
Aesthetic & Style:
- Classical Indian aesthetic, traditional Lucknowi Chikankari embroidery, delicate fabrics (georgette, mulmul, cotton silk, chanderi).
- Elegant jewelry: delicate silver rings, small vintage jhumkas, subtle nose pin or delicate payal.
- Hair & Features: Long, dark, natural wavy hair, expressive warm brown eyes, graceful dignified expressions.
- Atmosphere & Lighting: Vintage indoor courtyards, heritage archways, balconies overlooking old city, soft warm natural sunlight, golden hour ambiance, gentle shadows.
- Avoid modern artificiality, synthetic plastic textures, or generic Western aesthetics unless expressly requested.
"""
