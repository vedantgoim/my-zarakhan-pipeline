# Zara Khan Content Pipeline

Automated ingestion and prompt synthesis pipeline for the Zara Khan content calendar. Ingests assets from Google Sheets, resolves Pinterest and Instagram shortlinks to direct image CDN URLs, synthesizes diffusion prompts using Gemini 3.8 Flash, and writes the results back to the sheet for generation in Higgsfield or local workflows.

---

## Workflow

1. Ingests calendar rows from Google Sheets (or local Excel fallback).
2. Resolves reference shortlinks (`pin.it`, Instagram embeds) into direct CDN image URLs (`i.pinimg.com/...`).
3. Uses Gemini 3.8 Flash with structured Pydantic outputs (`ModelPromptBody`) to synthesize prompts grounded in the character avatar (`ZARA_KHAN.png`) and the resolved visual reference.
4. Enforces calendar notes as strict overrides (e.g. wardrobe color, setting adjustments).
5. Updates the sheet with direct reference URLs, positive prompts, negative prompts, and status `READY_FOR_HIGGSFIELD`.

---

## Setup

### 1. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file based on `.env.example`:
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
GOOGLE_SHEET_ID=your_sheet_id
GOOGLE_SERVICE_ACCOUNT_JSON=service_account.json
GOOGLE_SHEETS_TAB_NAME=Sheet1
```

### 3. Google Sheets Access

1. Enable the Google Sheets API and Google Drive API in Google Cloud Console.
2. Create a Service Account, generate a JSON key, and save it locally.
3. Share your Google Sheet with the service account email as **Editor**.
4. To populate the sheet with the 72-row calendar data, run:
```bash
python main.py --sync-sheets
```

---

## Usage

### Cloud Ingestion (Google Sheets)

Process pending items and prepare prompts for Higgsfield:
```bash
# Process all pending image assets
python main.py --cloud --pending-only

# Process a specific day
python main.py --cloud --day "Day 3"

# Process a single asset
python main.py --cloud --asset-id DAY01_STORY_01
```

### Local Operations

```bash
# View calendar summary
python main.py --status

# List scheduled assets
python main.py --list --day "Day 1"

# Dry run (synthesizes prompts and generates local placeholder)
python main.py --asset-id DAY01_STORY_01 --dry-run

# Local ComfyUI render (CyberPony SDXL + FaceDetailer)
python main.py --asset-id DAY01_STORY_01
```

---

## GitHub Actions Automation

The repository includes a scheduled workflow (`.github/workflows/ingestion_pipeline.yml`) configured to run daily at `06:00 UTC`.

Required GitHub Secrets (under repository Settings > Secrets and variables > Actions):
- `GEMINI_API_KEY`: Google Gemini API key.
- `GOOGLE_SHEET_ID`: Spreadsheet ID from the sheet URL.
- `GOOGLE_SERVICE_ACCOUNT_JSON`: Complete content of the service account JSON key file.

Manual runs can also be triggered directly from the Actions tab using the **Run workflow** button.
