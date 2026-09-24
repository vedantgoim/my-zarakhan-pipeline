import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import (
    GOOGLE_SHEET_ID,
    GOOGLE_SERVICE_ACCOUNT_JSON,
    GOOGLE_SHEETS_TAB_NAME,
    OPTIMIZED_CALENDAR_PATH
)

class GoogleSheetsManager:
    """Manages cloud calendar ingestion and synchronization via Google Sheets API."""

    EXPECTED_COLUMNS = [
        "Asset ID",
        "Day",
        "Post Type",
        "Asset Type",
        "Content Description",
        "Notes (Overrides)",
        "Caption Context",
        "Music",
        "Visual Reference URL",
        "Resolved Direct URL",
        "Final LLM Prompt",
        "Negative Prompt",
        "Generation Status",
        "Skip / Error Reason",
        "Last Updated"
    ]

    def __init__(
        self,
        sheet_id: Optional[str] = None,
        service_account_creds: Optional[str] = None,
        tab_name: Optional[str] = None
    ):
        self.sheet_id = (sheet_id or GOOGLE_SHEET_ID or os.getenv("GOOGLE_SHEET_ID", "")).strip().strip("'\"")
        self.creds_str = (
            service_account_creds
            or GOOGLE_SERVICE_ACCOUNT_JSON
            or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        )
        self.tab_name = tab_name or GOOGLE_SHEETS_TAB_NAME
        self.client = None
        self.sheet = None
        self.worksheet = None

        self._authenticate()

    def _authenticate(self):
        """Authenticates with Google Sheets API using service account credentials."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]

            raw = (self.creds_str or "").strip()
            # Strip outer quotes if any
            if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
                raw = raw[1:-1].strip()

            creds = None

            # Strategy 1: Direct JSON content (from GitHub Secrets or inline env)
            if "{" in raw and "}" in raw:
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    info = json.loads(raw[start:end])
                    creds = Credentials.from_service_account_info(info, scopes=scopes)
                    print(f"[GoogleSheetsManager] Authenticated via inline JSON for: {creds.service_account_email}")
                except Exception as json_err:
                    print(f"[GoogleSheetsManager] Inline JSON parse failed: {json_err}")

            # Strategy 2: File path (e.g. /tmp/gcp/service_account.json, GOOGLE_APPLICATION_CREDENTIALS, or local keys)
            if not creds:
                candidates = []
                if raw and len(raw) < 500:
                    candidates.append(Path(raw))
                    from .config import ROOT_DIR
                    candidates.append(ROOT_DIR / raw)

                env_app_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
                if env_app_creds:
                    candidates.append(Path(env_app_creds))

                candidates.append(Path("/tmp/gcp/service_account.json"))
                candidates.append(Path("/tmp/service_account.json"))

                from .config import ROOT_DIR
                candidates.append(ROOT_DIR / ".secrets" / "service_account.json")
                for sa_file in ROOT_DIR.glob("gen-lang-client-*.json"):
                    candidates.append(sa_file)

                # Deduplicate candidates preserving order
                unique_candidates = []
                seen = set()
                for c in candidates:
                    resolved = str(c.resolve()) if c.exists() else str(c)
                    if resolved not in seen:
                        seen.add(resolved)
                        unique_candidates.append(c)

                for p in unique_candidates:
                    if not p.exists() or not p.is_file():
                        continue
                    try:
                        file_text = p.read_text(encoding="utf-8")
                        if "{" in file_text and "}" in file_text:
                            start = file_text.find("{")
                            end = file_text.rfind("}") + 1
                            info = json.loads(file_text[start:end])
                            creds = Credentials.from_service_account_info(info, scopes=scopes)
                            print(f"[GoogleSheetsManager] Authenticated via key file '{p}' for: {creds.service_account_email}")
                            break
                        else:
                            creds = Credentials.from_service_account_file(str(p), scopes=scopes)
                            print(f"[GoogleSheetsManager] Authenticated via service_account_file '{p}' for: {creds.service_account_email}")
                            break
                    except Exception as file_err:
                        print(f"[GoogleSheetsManager] Candidate key file '{p}' failed: {file_err}")

            if not creds:
                print(f"[GoogleSheetsManager] ERROR: No valid service account credentials found (raw input length: {len(raw)} chars).")
                return

            self.client = gspread.authorize(creds)

            clean_sheet_id = (self.sheet_id or "").strip().strip("'\"")
            if not clean_sheet_id:
                print("[GoogleSheetsManager] ERROR: Google Sheet ID is missing or empty.")
                return

            try:
                self.sheet = self.client.open_by_key(clean_sheet_id)
            except Exception as sheet_err:
                print(f"[GoogleSheetsManager] Failed to open Google Sheet '{clean_sheet_id}': {sheet_err}")
                return

            # Discover available worksheets
            try:
                available_titles = [ws.title for ws in self.sheet.worksheets()]
            except Exception:
                available_titles = []

            for attempt in [self.tab_name, "Sheet1", "Zara Khan Calendar"]:
                if attempt and attempt in available_titles:
                    try:
                        self.worksheet = self.sheet.worksheet(attempt)
                        break
                    except Exception:
                        pass

            if not self.worksheet:
                self.worksheet = self.sheet.get_worksheet(0)

            if self.worksheet:
                print(f"[GoogleSheetsManager] Connected to worksheet '{self.worksheet.title}' in Google Sheet.")
            else:
                print(f"[GoogleSheetsManager] Warning: Sheet opened but no worksheet could be selected.")

        except Exception as e:
            print(f"[GoogleSheetsManager] Authentication setup failed: {e}")

    def is_connected(self) -> bool:
        """Returns True if successfully connected to the Google Sheet."""
        return self.worksheet is not None

    def get_assets(
        self,
        day: Optional[str] = None,
        asset_id: Optional[str] = None,
        image_only: bool = True,
        pending_only: bool = False,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves and normalizes assets from Google Sheets."""
        if not self.is_connected():
            return []

        try:
            records = self.worksheet.get_all_records()
        except Exception as e:
            print(f"[GoogleSheetsManager] Failed to fetch records: {e}")
            return []

        matched = []
        for idx, row in enumerate(records, start=2): # 1-indexed header is row 1
            a_id = str(row.get("Asset ID", "")).strip()
            if not a_id:
                continue

            status = str(row.get("Generation Status", "PENDING")).strip().upper()
            post_type = str(row.get("Post Type", "")).strip()
            asset_type = str(row.get("Asset Type", "")).strip()
            row_day = str(row.get("Day", "")).strip()

            is_image = "reel" not in post_type.lower() and "video" not in post_type.lower()

            if image_only and not is_image:
                continue

            if pending_only and status not in ["PENDING", "", "FAILED"]:
                continue

            if day and row_day.lower() != str(day).lower():
                continue

            if asset_id and a_id.lower() != str(asset_id).lower():
                continue

            matched.append({
                "row_index": idx,
                "asset_id": a_id,
                "day": row_day,
                "post_type": post_type,
                "asset_type": asset_type,
                "content_description": row.get("Content Description", ""),
                "notes": row.get("Notes (Overrides)", ""),
                "caption": row.get("Caption Context", ""),
                "music": row.get("Music", ""),
                "visual_reference_url": row.get("Visual Reference URL", ""),
                "resolved_direct_url": row.get("Resolved Direct URL", ""),
                "final_llm_prompt": row.get("Final LLM Prompt", ""),
                "negative_prompt": row.get("Negative Prompt", ""),
                "generation_status": status,
                "skip_reason": row.get("Skip / Error Reason", "")
            })

            if limit and len(matched) >= limit:
                break

        return matched

    def update_asset(
        self,
        asset_id: str,
        resolved_url: Optional[str] = None,
        llm_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        status: Optional[str] = None,
        skip_reason: Optional[str] = None
    ) -> bool:
        """Updates a single row in Google Sheets."""
        if not self.is_connected():
            return False

        try:
            cell = self.worksheet.find(asset_id, in_column=1)
            if not cell:
                return False

            row = cell.row
            headers = self.worksheet.row_values(1)
            header_map = {name: col for col, name in enumerate(headers, start=1)}

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            updates = []
            if resolved_url and "Resolved Direct URL" in header_map:
                updates.append({"col": header_map["Resolved Direct URL"], "val": resolved_url})
            if llm_prompt and "Final LLM Prompt" in header_map:
                updates.append({"col": header_map["Final LLM Prompt"], "val": llm_prompt})
            if negative_prompt and "Negative Prompt" in header_map:
                updates.append({"col": header_map["Negative Prompt"], "val": negative_prompt})
            if status and "Generation Status" in header_map:
                updates.append({"col": header_map["Generation Status"], "val": status})
            if skip_reason is not None and "Skip / Error Reason" in header_map:
                updates.append({"col": header_map["Skip / Error Reason"], "val": skip_reason})
            if "Last Updated" in header_map:
                updates.append({"col": header_map["Last Updated"], "val": now_str})

            # Update cells
            for u in updates:
                self.worksheet.update_cell(row, u["col"], u["val"])

            return True
        except Exception as e:
            print(f"[GoogleSheetsManager] Error updating asset {asset_id}: {e}")
            return False

    def sync_from_excel(self, excel_path: Optional[Path] = None) -> int:
        """Seeds or updates Google Sheets with normalized rows from optimized Excel file."""
        if not self.is_connected():
            print("[GoogleSheetsManager] Cannot sync: not connected to Google Sheets.")
            return 0

        path = excel_path or OPTIMIZED_CALENDAR_PATH
        if not path.exists():
            print(f"[GoogleSheetsManager] Source excel not found at: {path}")
            return 0

        try:
            from .calendar_manager import CalendarManager
            cm = CalendarManager(calendar_path=path)
            assets = cm.get_assets(image_only=False)

            rows_to_insert = [self.EXPECTED_COLUMNS]
            for a in assets:
                rows_to_insert.append([
                    a.get("asset_id", ""),
                    a.get("day", ""),
                    a.get("post_type", ""),
                    a.get("asset_type", ""),
                    a.get("content_description", ""),
                    a.get("notes", ""),
                    a.get("caption", ""),
                    a.get("music", ""),
                    a.get("visual_reference_url", ""),
                    "", # Resolved Direct URL
                    a.get("final_llm_prompt", ""),
                    a.get("negative_prompt", ""),
                    a.get("generation_status", "PENDING"),
                    a.get("skip_reason", ""),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

            self.worksheet.clear()
            self.worksheet.update(values=rows_to_insert, range_name="A1")
            print(f"[GoogleSheetsManager] Successfully uploaded {len(assets)} calendar records to Google Sheet!")
            return len(assets)

        except Exception as e:
            print(f"[GoogleSheetsManager] Error syncing from Excel: {e}")
            return 0
