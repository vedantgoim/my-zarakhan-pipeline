import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import ROOT_DIR, GOOGLE_SHEET_ID
from .google_sheets_manager import GoogleSheetsManager
from .calendar_manager import CalendarManager
from .media_downloader import MediaDownloader
from .prompt_synthesizer import PromptSynthesizer

class CloudIngestionRunner:
    """
    Automated ingestion pipeline:
    - Ingests content calendar from Google Sheets (or local Excel fallback).
    - Resolves reference media URLs into direct high-res CDN links for Higgsfield.
    - Synthesizes cultural and photorealistic prompts using Gemini multimodal API.
    - Marks assets as READY_FOR_HIGGSFIELD in cloud spreadsheet.
    """

    def __init__(
        self,
        use_google_sheets: bool = True,
        google_sheets_manager: Optional[GoogleSheetsManager] = None,
        calendar_manager: Optional[CalendarManager] = None,
        media_downloader: Optional[MediaDownloader] = None,
        prompt_synthesizer: Optional[PromptSynthesizer] = None
    ):
        self.use_google_sheets = use_google_sheets
        self.gs_manager = google_sheets_manager or GoogleSheetsManager()
        self.cal_manager = calendar_manager or CalendarManager()
        self.downloader = media_downloader or MediaDownloader()
        self.synthesizer = prompt_synthesizer or PromptSynthesizer()

        # Fallback check
        if self.use_google_sheets and not self.gs_manager.is_connected():
            print("[CloudIngestionRunner] Google Sheets credentials not active; falling back to local Excel calendar.")
            self.use_google_sheets = False

    def process_asset(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single calendar record through reference resolution and prompt synthesis."""
        asset_id = record["asset_id"]
        post_type = record["post_type"]
        asset_type = record["asset_type"]
        desc = record.get("content_description", "")
        notes = record.get("notes", "")
        caption = record.get("caption", "")
        ref_url = record.get("visual_reference_url", "")

        print(f"\n{'='*70}")
        print(f"Processing Cloud Asset: [{asset_id}] | Day: {record.get('day')} | Type: {asset_type}")
        print(f"Description: {desc}")
        if notes:
            print(f"Notes (Overrides): {notes}")

        # Step 1: Resolve Direct CDN Media URL
        direct_url = None
        if ref_url:
            print(f"-> Resolving reference media from: {ref_url}...")
            direct_url = self.downloader.resolve_direct_url(ref_url)
            if direct_url:
                print(f"   Direct CDN URL: {direct_url[:90]}...")
            else:
                print(f"   Could not resolve direct CDN URL; retaining original.")
                direct_url = ref_url
        else:
            print(f"   No visual reference URL provided.")

        # Step 2: Synthesize Diffusion Prompts (Multimodal Gemini)
        print(f"-> Synthesizing diffusion prompts via Gemini ({self.synthesizer.model_name})...")
        pos_prompt, neg_prompt = self.synthesizer.synthesize_prompt(
            asset_id=asset_id,
            post_type=post_type,
            content_description=desc,
            notes=notes,
            caption=caption,
            reference_image_url=direct_url
        )
        print(f"   Positive Prompt: {pos_prompt[:120]}...")
        print(f"   Negative Prompt: {neg_prompt[:80]}...")

        # Step 3: Update Cloud Spreadsheet (or local Excel)
        status = "READY_FOR_HIGGSFIELD"
        if self.use_google_sheets:
            print(f"-> Updating Google Sheet for {asset_id}...")
            self.gs_manager.update_asset(
                asset_id=asset_id,
                resolved_url=direct_url,
                llm_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                status=status
            )
        else:
            print(f"-> Updating local Excel for {asset_id}...")
            self.cal_manager.update_asset(
                asset_id=asset_id,
                llm_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                status=status
            )

        print(f"   [SUCCESS] Marked as {status}")
        return {
            "asset_id": asset_id,
            "status": status,
            "direct_url": direct_url,
            "prompt": pos_prompt
        }

    def run_batch(
        self,
        day: Optional[str] = None,
        asset_id: Optional[str] = None,
        limit: Optional[int] = None,
        pending_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Executes ingestion & prompting over targeted assets."""
        if self.use_google_sheets:
            print(f"\n=======================================================")
            print(f" Starting Cloud Ingestion Pipeline (Google Sheets)")
            print(f" Target Sheet ID: {self.gs_manager.sheet_id}")
            print(f"=======================================================")
            assets = self.gs_manager.get_assets(
                day=day,
                asset_id=asset_id,
                image_only=True,
                pending_only=pending_only,
                limit=limit
            )
        else:
            print(f"\n=======================================================")
            print(f" Starting Ingestion Pipeline (Local Excel Mode)")
            print(f" Target File: {self.cal_manager.calendar_path.name}")
            print(f"=======================================================")
            assets = self.cal_manager.get_assets(
                day=day,
                asset_id=asset_id,
                image_only=True,
                pending_only=pending_only,
                limit=limit
            )

        print(f" Assets to process: {len(assets)}")
        results = []
        for rec in assets:
            res = self.process_asset(rec)
            results.append(res)

        print(f"\n=======================================================")
        print(f" Cloud Ingestion Run Complete: {len(results)} assets prepared for Higgsfield")
        print(f"=======================================================\n")

        # GitHub Actions Step Summary output if running in CI
        step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if step_summary_path and os.path.exists(os.path.dirname(step_summary_path)):
            try:
                with open(step_summary_path, "a", encoding="utf-8") as f:
                    f.write(f"## 🚀 Zara Khan Cloud Ingestion Summary\n\n")
                    f.write(f"- **Target Sheet / Mode**: {'Google Sheets' if self.use_google_sheets else 'Local Excel'}\n")
                    f.write(f"- **Total Assets Prompted**: {len(results)}\n")
                    f.write(f"- **Status Set**: `READY_FOR_HIGGSFIELD`\n\n")
                    f.write("| Asset ID | Status | Direct Reference | Prompt Snippet |\n")
                    f.write("| :--- | :--- | :--- | :--- |\n")
                    for r in results:
                        u_snippet = f"[View]({r['direct_url']})" if r.get('direct_url') else "N/A"
                        f.write(f"| `{r['asset_id']}` | `{r['status']}` | {u_snippet} | {r['prompt'][:60]}... |\n")
            except Exception as e:
                print(f"[CloudIngestionRunner] Could not write GitHub Step Summary: {e}")

        return results
