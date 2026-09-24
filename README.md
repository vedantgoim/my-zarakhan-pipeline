# Zara Khan Content Ingestion & Automation Pipeline

An automated cloud-ready ingestion, prompt synthesis, and image generation pipeline for the **Zara Khan** cultural social media campaign.

This repository ingests a 30-day content calendar from **Google Sheets**, resolves visual references (Pinterest / Instagram) into **direct high-resolution CDN links**, synthesizes photorealistic prompts tailored to the Awadhi/Lucknowi aesthetic using **Google Gemini 3.5 Flash Lite**, and prepares assets for generation in **Higgsfield** or directly via **Gemini 3.1 Flash Lite Image API**.

---

## 🏗 Architecture & Workflow

```mermaid
flowchart LR
    A["Google Sheet\n(Cloud Calendar)"] -->|Daily Scheduled Cron\n(GitHub Actions)| B["Cloud Ingestion Runner"]
    B -->|Resolve Shortlinks| C["High-Res CDN URLs\n(Pinterest / Instagram)"]
    C -->|Visual Media + Overrides| D["Gemini 3.5 Flash Lite\n(Multimodal API)"]
    D -->|Synthesized Prompts| E["Update Google Sheet"]
    E -->|Direct Links & Prompts| F["Higgsfield / Gemini Image API\n(Creative Team)"]
```

1. **Scheduled Trigger**: GitHub Actions runs daily on a cron schedule (`06:00 UTC` / `11:30 AM IST`) or on-demand via `workflow_dispatch`.
2. **Reference Resolution**: Resolves shortened links (`pin.it`, Instagram embeds) into permanent, direct high-resolution image links (`i.pinimg.com/...`).
3. **Multimodal Prompt Synthesis**: Feeds the avatar reference, visual inspiration, scene description, and user note overrides into Gemini to produce culturally authentic, rich diffusion prompts.
4. **Cloud Sync**: Writes direct reference links, positive prompts, negative prompts, and `READY_FOR_HIGGSFIELD` status back to the Google Sheet.
5. **Image Generation Options**:
   - **Higgsfield**: Creative team uses the prompts and direct reference URLs directly in Higgsfield.
   - **Gemini Image API (`gemini-3.1-flash-lite-image`)**: Generate images directly from the pipeline via `--gemini-generate`.
   - **Local ComfyUI**: Local fallback using CyberPony SDXL + FaceDetailer.

---

## 🔑 Where & How to Get Your Credentials

### 1. `GEMINI_API_KEY`
1. Go to **Google AI Studio**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account and click **"Create API key"**.
3. Copy the generated key.
> **Note for `gemini-3.1-flash-lite-image` Image Generation**: Free tier API keys have image generation quota set to 0 by Google. To generate images directly using the Gemini Image API, link a Google Cloud billing account (Pay-As-You-Go) to your Google AI Studio project in [AI Studio Settings / Billing](https://aistudio.google.com/). Text and multimodal prompt synthesis work on the standard tier.

### 2. `GOOGLE_SHEET_ID`
1. Go to [https://sheets.new](https://sheets.new) and create a new Google Sheet (or open your existing calendar sheet).
2. Look at the URL in your browser's address bar:
   ```text
   https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
   ```
3. Copy the ID string between `/d/` and `/edit` (in the example above: `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms`).

### 3. `GOOGLE_SERVICE_ACCOUNT_JSON`
1. Go to the **Google Cloud Console**: [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Create a project (or select your existing Gemini project).
3. Enable the two required Google APIs:
   - **Google Sheets API**: [Enable Sheets API](https://console.cloud.google.com/apis/library/sheets.googleapis.com)
   - **Google Drive API**: [Enable Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com)
4. Go to **IAM & Admin > Service Accounts**: [https://console.cloud.google.com/iam-admin/serviceaccounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
5. Click **"+ CREATE SERVICE ACCOUNT"**:
   - Service account name: `zara-khan-pipeline`
   - Click **Create and Continue**, then click **Done**.
6. Generate the JSON key:
   - Click on the newly created service account email.
   - Go to the **Keys** tab > click **Add Key** > **Create new key** > choose **JSON** > click **Create**.
   - A `.json` file will automatically download to your computer.
   - Open this file in Notepad — the entire JSON text (from `{` to `}`) is your `GOOGLE_SERVICE_ACCOUNT_JSON`.
7. **Share your Google Sheet with the Service Account**:
   - Inside the downloaded JSON file, copy the `"client_email"` (e.g. `zara-khan-pipeline@your-project.iam.gserviceaccount.com`).
   - Open your Google Sheet, click the green **Share** button in the top-right.
   - Paste the service account email, grant it **Editor** permissions, and click **Send / Share**.

---

## ☁️ How to Host on GitHub Actions

1. In your GitHub repository, go to **Settings** > **Secrets and variables** > **Actions** > **New repository secret**.
2. Add the three secrets:

| Secret Name | Value |
| :--- | :--- |
| `GEMINI_API_KEY` | Your Google Gemini API Key |
| `GOOGLE_SHEET_ID` | Your Google Spreadsheet ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The entire contents of your downloaded Service Account `.json` file |

3. To seed your 72-row calendar into the Google Sheet initially, run locally:
```bash
python main.py --sync-sheets
```
4. GitHub Actions will now automatically run daily at **06:00 UTC** (or you can trigger it manually under the **Actions** tab by selecting **"Zara Khan Cloud Ingestion Pipeline"** > **Run workflow**).

---

## ⚙️ CLI Usage

```bash
# Cloud Ingestion Mode (Google Sheets + Direct CDN links for Higgsfield)
python main.py --cloud --pending-only

# Generate images directly via Gemini 3.1 Flash Lite Image API
python main.py --gemini-generate --limit 1

# Local ComfyUI CyberPony SDXL Generation
python main.py --asset-id DAY02_STORY_01

# Inspect Calendar Status
python main.py --status
```
