# Zara Khan Multimodal Content & Generation Pipeline

Production-grade autonomous ingestion, multimodal prompt synthesis, and visual generation pipeline for the virtual lifestyle creator **Zara Khan**. 

The system ingests editorial calendar rows from Google Sheets, resolves visual reference media, synthesizes physically grounded diffusion prompts using Google Gemini Vision, and renders photorealistic outputs via local ComfyUI Flux 2 Klein 9B pipelines. It is paired with an obsidian-styled Next.js creative studio dashboard (`contentgen`) for creative inspection, manual styling overrides, and publication review.

---

## System Architecture

```
                       [Google Sheets Editorial Calendar]
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
             [Cloud Ingestion]                  [Next.js Web Studio]
         (pipeline/cloud_ingestion)             (contentgen - Port 3000)
                     │                                   │
       ┌─────────────┼─────────────┐                     │
       ▼             ▼             ▼                     │
 [Avatar Ref]  [Setting Ref]  [Wardrobe Ref]             │
(ZARA_KHAN.png) (Pinterest)   (Fashion Look)             │
       └─────────────┬─────────────┘                     │
                     ▼                                   │
        [Gemini Vision Synthesizer]                      │
       (Multimodal Prompt Compiler)                      │
                     │                                   │
                     ▼                                   │
         [Master Prompt Compiled] ◄──────────────────────┘
                     │                     (Overrides & Reviews)
                     ▼
         [ComfyUI Diffusion Engine]
           • Flux 2 Klein 9B
           • Dual CLIP-L + T5XXL
           • FaceDetailer Portrait Pass
                     │
                     ▼
             [Verified Outputs]
          (outputs/DAY*.png · 9:16)
```

---

## Key Features

1. **3-Reference Conditioning Engine**
   - Eliminates identity drift and background hallucination by conditioning simultaneously on Avatar (`ZARA_KHAN.png`), Spatial Setting (Pinterest/Drive), and Wardrobe Reference.
2. **Gemini Vision Multimodal Synthesis**
   - Automatically inspects visual textures, draping, and architectural lighting across all reference images to produce structured, high-coherence prompts.
3. **Dual ComfyUI Workflow Serialization**
   - Generates both headless API workflows (`workflow_api.json`) for automated Python dispatch and visual LiteGraph layouts (`workflow.json`) for interactive inspection in ComfyUI Web GUI.
4. **Creator Studio Dashboard (`contentgen`)**
   - Modern, clutter-free Next.js interface with 3-reference visual galleries, 1-click prompt copying, live creative overrides (wardrobe, camera, lighting, environment, negative guardrails), and high-resolution output review.
5. **Bidirectional Sheet Synchronization**
   - Keeps Google Sheets and local Excel workbooks (`Vinayak's Copy of Zara Khan.xlsx`, `Zara_Khan_Optimized_Calendar.xlsx`) synchronized across status flags (`queued` -> `prompted` -> `done`).

---

## Project Structure

```
AUTOMATION1/
├── pipeline/                           # Core Python pipeline modules
│   ├── calendar_manager.py             # Calendar indexing & status management
│   ├── cloud_ingestion_runner.py       # Sheet ingestion runner
│   ├── comfyui_client.py               # WebSocket client for local ComfyUI
│   ├── config.py                       # Configuration & environment loader
│   ├── gemini_image_generator.py       # Gemini API client
│   ├── google_sheets_manager.py        # Google Sheets v4 API manager
│   ├── media_downloader.py             # Pinterest & direct URL resolver
│   ├── pipeline_runner.py              # End-to-end execution orchestrator
│   ├── prompt_synthesizer.py           # Gemini Vision 3-ref prompt compiler
│   └── workflow_builder.py             # ComfyUI workflow generator
├── workflows/                          # ComfyUI workflow definitions
│   ├── flux_2_klein_9b_multimodal.json     # Visual LiteGraph canvas
│   └── flux_2_klein_9b_multimodal_api.json # Headless API prompt format
├── outputs/                            # Rendered production assets
│   ├── DAY101_POST_01.png              # Row 184 output (Heritage Cafe)
│   ├── DAY102_POST_01.png              # Row 185 output (Art Gallery)
│   └── DAY103_POST_01.png              # Row 186 output (Mahogany Library)
├── references/                         # Downloaded & cached reference media
├── ZARAKHANREFERENCE/                  # Character anchor assets
│   └── ZARA_KHAN.png                   # Primary 4-view facial likeness anchor
├── main.py                             # Unified CLI entry point
├── PROGRESS.md                         # In-depth engineering report
└── README.md                           # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for Next.js studio)
- **ComfyUI** running locally on port 8188 with:
  - Flux 2 Klein 9B checkpoint (`flux-2-klein-9b.safetensors` or compatible)
  - `t5xxl_fp16.safetensors` and `clip_l.safetensors`
  - ComfyUI-Impact-Pack (FaceDetailer)
- **Google Cloud Service Account** with Google Sheets and Drive API scopes.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vedantgoim/my-zarakhan-pipeline.git
   cd my-zarakhan-pipeline
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-3.8-flash
   GOOGLE_SHEET_ID=your_spreadsheet_id
   GOOGLE_SERVICE_ACCOUNT_JSON=service_account.json
   GOOGLE_SHEETS_TAB_NAME=Sheet1
   COMFYUI_URL=http://127.0.0.1:8188
   ```

4. **Launch the Next.js Studio (Optional):**
   ```bash
   cd ../contentgen
   npm install
   npm run dev
   # Studio available at http://localhost:3000
   ```

---

## Usage

### Ingesting & Prompting Calendar Rows

```bash
# Ingest and prompt all pending rows from Google Sheets
python main.py --cloud --pending-only

# Ingest and prompt a specific calendar Day
python main.py --cloud --day "Day 101"

# Synthesize prompt for a single row without calling ComfyUI
python main.py --asset-id DAY101_POST_01 --dry-run
```

### Rendering via ComfyUI

```bash
# Execute local ComfyUI render for a specific asset
python main.py --asset-id DAY101_POST_01

# Inspect pipeline calendar status
python main.py --status
```

---

## Verified Production Test Runs

| Row / Day | Concept & Setting | Wardrobe Styling | Output Resolution | Output Asset |
| :--- | :--- | :--- | :--- | :--- |
| **Row 184** (Day 101) | Heritage Cafe Veranda Patio, South Mumbai | Ivory linen shirt, gold necklaces, black coffee | 896 × 1152 PNG | `outputs/DAY101_POST_01.png` |
| **Row 185** (Day 102) | Modern Minimalist Art Gallery, South Mumbai | Royal purple Banarasi silk saree, gold zari | 896 × 1152 PNG | `outputs/DAY102_POST_01.png` |
| **Row 186** (Day 103) | Mahogany Wood Heritage Library Lounge | Crimson & gold Kanjeevaram saree, temple jewelry | 896 × 1152 PNG | `outputs/DAY103_POST_01.png` |

---

## Security & Integrity

- **Path Traversal Protection:** Image endpoints use strict basename sanitization and directory whitelist bounds checking.
- **Zero Known CVEs:** Automated security scanning verified 0 vulnerabilities in dependency audit.
- **Strict Validation:** Calendar row inputs and sheet parameters are type-checked and sanitized before API calls.

---

## License

Internal proprietary pipeline for the Zara Khan virtual creator project. All rights reserved.
