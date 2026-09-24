import argparse
import sys
from pathlib import Path

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure utf-8 output in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

from pipeline.calendar_manager import CalendarManager
from pipeline.pipeline_runner import PipelineRunner
from pipeline.config import OPTIMIZED_CALENDAR_PATH

def print_summary(manager: CalendarManager):
    stats = manager.get_summary()
    print("\n" + "=" * 55)
    print(" ZARA KHAN CONTENT PIPELINE - STATUS SUMMARY")
    print("=" * 55)
    print(f" Spreadsheet: {manager.calendar_path.name}")
    print(f" Total Records:     {stats['total_records']}")
    print(f" Active Images:     {stats['image_assets']}")
    print(f" Skipped (Reels):   {stats['skipped_assets']}")
    print("-" * 55)
    print(f" PENDING:           {stats['pending']}")
    print(f" GENERATED:         {stats['generated']}")
    print(f" DRY_RUN:           {stats['dry_run']}")
    print(f" FAILED:            {stats['failed']}")
    print("=" * 55 + "\n")

def list_assets(manager: CalendarManager, day: str = None):
    records = manager.get_assets(day=day, image_only=False)
    print("\n" + "=" * 90)
    print(f"{'Asset ID':<26} | {'Day':<8} | {'Type':<18} | {'Status':<10} | {'Output'}")
    print("=" * 90)
    for rec in records:
        aid = rec["asset_id"]
        d = rec["day"]
        atype = rec["asset_type"]
        status = rec["generation_status"]
        out = rec["output_path"] or (rec["skip_reason"][:30] if rec["skip_reason"] else "-")
        print(f"{aid:<26} | {d:<8} | {atype:<18} | {status:<10} | {out}")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Zara Khan Local Image Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current calendar stats
  python main.py --status

  # List all assets for Day 3
  python main.py --list --day "Day 3"

  # Test a single asset in dry-run mode (Gemini prompt synthesis + mock render)
  python main.py --asset-id DAY01_STORY_01 --dry-run

  # Run all image assets for Day 1 in dry-run mode
  python main.py --day "Day 1" --dry-run

  # Run the first 3 pending image assets in dry-run mode
  python main.py --dry-run --limit 3

  # Run live ComfyUI generation for Day 3
  python main.py --day "Day 3"
        """
    )

    parser.add_argument("--day", type=str, help="Filter by Day (e.g. 'Day 1', 'Day 3', or '3')")
    parser.add_argument("--asset-id", type=str, help="Process a single specific Asset ID (e.g. DAY01_STORY_01)")
    parser.add_argument("--dry-run", action="store_true", help="Synthesize prompts and validate without invoking ComfyUI")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading reference media if already cached")
    parser.add_argument("--pending-only", action="store_true", help="Only process items in PENDING state")
    parser.add_argument("--cloud", action="store_true", help="Run in cloud ingestion mode for Higgsfield (Google Sheets + direct URLs)")
    parser.add_argument("--sync-sheets", action="store_true", help="Upload and sync local Excel calendar to Google Sheet")
    parser.add_argument("--gemini-generate", action="store_true", help="Render images via Gemini API (gemini-3.1-flash-lite-image)")
    parser.add_argument("--generator", choices=["comfyui", "gemini"], default="comfyui", help="Image generation backend (default: comfyui)")
    parser.add_argument("--limit", type=int, help="Maximum number of items to process")
    parser.add_argument("--status", action="store_true", help="Display calendar status summary and exit")
    parser.add_argument("--list", action="store_true", help="List assets matching filters and exit")

    args = parser.parse_args()

    # Check if Cloud Mode is requested
    import os
    is_cloud = args.cloud or os.getenv("CLOUD_MODE", "").lower() == "true"

    if args.sync_sheets:
        from pipeline.google_sheets_manager import GoogleSheetsManager
        gsm = GoogleSheetsManager()
        if not gsm.is_connected():
            print("[ERROR] Cannot sync: Google Sheets credentials or Sheet ID missing.")
            return
        gsm.sync_from_excel(OPTIMIZED_CALENDAR_PATH)
        return

    if is_cloud:
        from pipeline.cloud_ingestion_runner import CloudIngestionRunner
        runner = CloudIngestionRunner()
        runner.run_batch(
            day=args.day,
            asset_id=args.asset_id,
            limit=args.limit,
            pending_only=args.pending_only
        )
        return

    manager = CalendarManager(OPTIMIZED_CALENDAR_PATH)

    if args.status:
        print_summary(manager)
        return

    if args.list:
        list_assets(manager, day=args.day)
        return

    # If no flags are provided, show status and help
    if not (args.day or args.asset_id or args.dry_run or args.limit or args.gemini_generate):
        print_summary(manager)
        print("Tip: Run with --help to see all execution options, or use --cloud for cloud ingestion.\n")
        return

    generator_engine = "gemini" if args.gemini_generate or args.generator == "gemini" else "comfyui"

    runner = PipelineRunner(calendar_manager=manager)
    runner.run_batch(
        day=args.day,
        asset_id=args.asset_id,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_download=args.skip_download,
        pending_only=args.pending_only,
        generator=generator_engine
    )

if __name__ == "__main__":
    main()
