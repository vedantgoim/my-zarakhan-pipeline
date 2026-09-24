from pathlib import Path
from typing import Optional, Dict, Any, List
from .config import ROOT_DIR, REFERENCES_DIR, OUTPUTS_DIR
from .calendar_manager import CalendarManager
from .media_downloader import MediaDownloader
from .prompt_synthesizer import PromptSynthesizer
from .workflow_builder import WorkflowBuilder
from .comfyui_client import ComfyUIClient
from .gemini_image_generator import GeminiImageGenerator

class PipelineRunner:
    """Orchestrates end-to-end processing from calendar ingestion to final generation."""

    def __init__(
        self,
        calendar_manager: Optional[CalendarManager] = None,
        media_downloader: Optional[MediaDownloader] = None,
        prompt_synthesizer: Optional[PromptSynthesizer] = None,
        comfyui_client: Optional[ComfyUIClient] = None,
        gemini_generator: Optional[GeminiImageGenerator] = None
    ):
        self.calendar_manager = calendar_manager or CalendarManager()
        self.media_downloader = media_downloader or MediaDownloader()
        self.prompt_synthesizer = prompt_synthesizer or PromptSynthesizer()
        self.comfyui_client = comfyui_client or ComfyUIClient()
        self.gemini_generator = gemini_generator or GeminiImageGenerator()

        # Ensure base avatar asset is synced to ComfyUI input directory
        try:
            from .config import AVATAR_REFERENCE_PATH, COMFYUI_INPUT_DIR
            if AVATAR_REFERENCE_PATH.exists() and COMFYUI_INPUT_DIR.exists():
                import shutil
                shutil.copy2(AVATAR_REFERENCE_PATH, COMFYUI_INPUT_DIR / "ZARA_KHAN.png")
        except Exception as e:
            print(f"[PipelineRunner] Warning: Could not sync avatar to ComfyUI input: {e}")

    def process_asset(
        self,
        record: Dict[str, Any],
        dry_run: bool = False,
        skip_download: bool = False,
        generator: str = "comfyui"
    ) -> Dict[str, Any]:
        """Processes a single calendar record through all pipeline stages."""
        asset_id = record["asset_id"]
        post_type = record["post_type"]
        asset_type = record["asset_type"]
        desc = record["content_description"]
        notes = record["notes"]
        caption = record["caption"]
        ref_url = record["visual_reference_url"]

        print(f"\n{'='*70}")
        print(f"Processing: [{asset_id}] | Day: {record['day']} | Type: {asset_type}")
        print(f"Description: {desc}")
        if notes:
            print(f"Notes (Overrides): {notes}")

        # Stage 1: Reference Media Ingestion
        ref_media_path = None
        direct_url = None
        if not skip_download and ref_url:
            print(f"-> Resolving reference media from: {ref_url}...")
            local_path, direct_url = self.media_downloader.ingest_reference(asset_id, ref_url)
            if local_path and local_path.exists():
                ref_media_path = local_path
                print(f"   Saved reference to: {ref_media_path}")
            else:
                print(f"   Reference download skipped or not found.")
        elif skip_download:
            cached = self.media_downloader.get_cached_path(asset_id)
            if cached:
                ref_media_path = cached
                print(f"   Using existing reference cache: {ref_media_path}")

        # Update reference path in calendar
        if ref_media_path:
            rel_ref_path = f"references/{ref_media_path.name}"
            self.calendar_manager.update_asset(asset_id, ref_media_path=rel_ref_path)

        # Stage 2: Multimodal Prompt Synthesis
        print(f"-> Synthesizing diffusion prompts via Gemini ({self.prompt_synthesizer.model_name})...")
        pos_prompt, neg_prompt = self.prompt_synthesizer.synthesize_prompt(
            asset_id=asset_id,
            post_type=post_type,
            content_description=desc,
            notes=notes,
            caption=caption,
            reference_media_path=ref_media_path,
            reference_image_url=direct_url
        )
        print(f"   Positive Prompt: {pos_prompt[:120]}...")
        print(f"   Negative Prompt: {neg_prompt[:80]}...")

        # Update synthesized prompt in calendar
        self.calendar_manager.update_asset(
            asset_id=asset_id,
            llm_prompt=pos_prompt,
            negative_prompt=neg_prompt
        )

        dim = WorkflowBuilder.get_dimensions(post_type, asset_type)
        rel_out_path = f"outputs/{asset_id}.png"

        # Stage 3: Dry Run Mode
        if dry_run:
            print(f"-> [DRY RUN] Creating placeholder render for verification...")
            mock_path = self.comfyui_client.create_mock_output(
                asset_id=asset_id,
                width=dim["width"],
                height=dim["height"],
                prompt=pos_prompt
            )
            self.calendar_manager.update_asset(
                asset_id=asset_id,
                status="DRY_RUN",
                output_path=rel_out_path
            )
            print(f"   Dry run asset saved to: {mock_path}")
            return {"asset_id": asset_id, "status": "DRY_RUN", "output": rel_out_path}

        # Stage 4: Generation Mode (Gemini Image API vs ComfyUI)
        if generator.lower() == "gemini":
            print(f"-> Dispatching job to Gemini Image API ({self.gemini_generator.model_name})...")
            success, out_file, msg = self.gemini_generator.generate_image(
                asset_id=asset_id,
                prompt=pos_prompt,
                reference_image_path=ref_media_path,
                reference_image_url=direct_url
            )
        else:
            print(f"-> Checking ComfyUI instance at {self.comfyui_client.base_url}...")
            if not self.comfyui_client.is_online():
                err = f"ComfyUI offline at {self.comfyui_client.base_url}"
                print(f"   [ERROR] {err}")
                self.calendar_manager.update_asset(
                    asset_id=asset_id,
                    status="FAILED",
                    skip_reason=err
                )
                return {"asset_id": asset_id, "status": "FAILED", "error": err}

            workflow_prompt = WorkflowBuilder.build_cyberpony_workflow(
                asset_id=asset_id,
                post_type=post_type,
                asset_type=asset_type,
                positive_prompt=pos_prompt,
                negative_prompt=neg_prompt
            )
            print(f"-> Dispatching job to ComfyUI (CyberPony SDXL + FaceDetailer)...")
            success, out_file, msg = self.comfyui_client.render_workflow(
                asset_id=asset_id,
                workflow_prompt=workflow_prompt
            )

        if success and out_file:
            print(f"   [SUCCESS] Image rendered and saved to: {out_file}")
            self.calendar_manager.update_asset(
                asset_id=asset_id,
                status="GENERATED",
                output_path=rel_out_path
            )
            return {"asset_id": asset_id, "status": "GENERATED", "output": rel_out_path}
        else:
            print(f"   [FAILED] {msg}")
            self.calendar_manager.update_asset(
                asset_id=asset_id,
                status="FAILED",
                skip_reason=msg
            )
            return {"asset_id": asset_id, "status": "FAILED", "error": msg}

    def run_batch(
        self,
        day: Optional[str] = None,
        asset_id: Optional[str] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        skip_download: bool = False,
        pending_only: bool = False,
        generator: str = "comfyui"
    ) -> List[Dict[str, Any]]:
        """Runs the pipeline over matched assets."""
        assets = self.calendar_manager.get_assets(
            day=day,
            asset_id=asset_id,
            image_only=True,
            pending_only=pending_only,
            limit=limit
        )

        print(f"\n=======================================================")
        print(f" Starting Zara Khan Image Pipeline")
        print(f" Total targets to process: {len(assets)}")
        print(f" Mode: {'DRY RUN' if dry_run else 'LIVE COMFYUI'}")
        print(f"=======================================================")

        results = []
        for rec in assets:
            res = self.process_asset(rec, dry_run=dry_run, skip_download=skip_download, generator=generator)
            results.append(res)

        # Print summary
        summary = self.calendar_manager.get_summary()
        print(f"\n=======================================================")
        print(f" Pipeline Run Complete")
        print(f" Total Calendar Records: {summary['total_records']}")
        print(f" Image Assets: {summary['image_assets']} | Skipped: {summary['skipped_assets']}")
        print(f" Status -> Generated: {summary['generated']}, Dry Run: {summary['dry_run']}, Pending: {summary['pending']}, Failed: {summary['failed']}")
        print(f" Updated Spreadsheet: {self.calendar_manager.calendar_path}")
        print(f"=======================================================\n")

        return results
