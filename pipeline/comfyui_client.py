import urllib.request
import urllib.parse
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from .config import COMFYUI_URL, COMFYUI_OUTPUT_DIR, OUTPUTS_DIR

class ComfyUIClient:
    """Client for interacting with local ComfyUI instance."""

    def __init__(self, base_url: str = COMFYUI_URL, output_dir: Optional[Path] = None):
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_online(self, timeout: int = 3) -> bool:
        """Checks if ComfyUI server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def queue_prompt(self, workflow_prompt: Dict[str, Any]) -> Optional[str]:
        """Queues a prompt in ComfyUI and returns the prompt_id."""
        url = f"{self.base_url}/prompt"
        data = json.dumps(workflow_prompt).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("prompt_id")
        except Exception as e:
            print(f"[ComfyUIClient] Failed to queue prompt: {e}")
            return None

    def wait_for_completion(self, prompt_id: str, max_wait: int = 600, poll_interval: int = 2) -> Optional[Dict[str, Any]]:
        """Polls history until the prompt is executed."""
        url = f"{self.base_url}/history/{prompt_id}"
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    history = json.loads(resp.read().decode("utf-8"))
                    if prompt_id in history:
                        item = history[prompt_id]
                        status = item.get("status", {})
                        if status.get("completed", False):
                            return item
                        # Check outputs
                        outputs = item.get("outputs", {})
                        if outputs:
                            return item
            except Exception:
                pass
            time.sleep(poll_interval)

        return None

    def retrieve_and_save_image(self, history_item: Dict[str, Any], asset_id: str) -> Optional[Path]:
        """Extracts output image from history item and saves to output directory."""
        outputs = history_item.get("outputs", {})
        target_path = self.output_dir / f"{asset_id}.png"

        # Search for SaveImage output
        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            for img_info in images:
                filename = img_info.get("filename")
                subfolder = img_info.get("subfolder", "")
                
                # Check directly in ComfyUI output directory
                local_comfy_file = COMFYUI_OUTPUT_DIR / subfolder / filename
                if local_comfy_file.exists():
                    shutil.copy2(local_comfy_file, target_path)
                    return target_path

                # Fallback: Download via /view endpoint
                try:
                    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
                    view_url = f"{self.base_url}/view?{params}"
                    req = urllib.request.Request(view_url)
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        with open(target_path, "wb") as f:
                            f.write(resp.read())
                    return target_path
                except Exception as e:
                    print(f"[ComfyUIClient] Error downloading image from /view: {e}")

        return None

    def render_workflow(
        self,
        asset_id: str,
        workflow_prompt: Dict[str, Any],
        max_wait: int = 600
    ) -> Tuple[bool, Optional[Path], str]:
        """Submits and waits for a generation job."""
        if not self.is_online():
            return False, None, "ComfyUI server is offline at " + self.base_url

        prompt_id = self.queue_prompt(workflow_prompt)
        if not prompt_id:
            return False, None, "Failed to submit job to ComfyUI"

        print(f"[ComfyUIClient] Job queued (prompt_id: {prompt_id}), waiting for execution...")
        history_item = self.wait_for_completion(prompt_id, max_wait=max_wait)
        if not history_item:
            return False, None, "Job timed out or failed in ComfyUI"

        saved_path = self.retrieve_and_save_image(history_item, asset_id)
        if saved_path and saved_path.exists():
            return True, saved_path, "Success"

        return False, None, "Image was not saved from ComfyUI output"

    def create_mock_output(
        self,
        asset_id: str,
        width: int = 896,
        height: int = 1152,
        prompt: str = ""
    ) -> Path:
        """Creates a mock placeholder image for dry-run verification."""
        target_path = self.output_dir / f"{asset_id}.png"
        img = Image.new("RGB", (width, height), color=(30, 41, 59)) # Deep slate
        draw = ImageDraw.Draw(img)

        # Draw decorative border
        draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(218, 165, 32), width=4) # gold
        draw.rectangle([(28, 28), (width - 28, height - 28)], outline=(218, 165, 32), width=1)

        # Draw text info
        title = f"ZARA KHAN PIPELINE - DRY RUN"
        asset_label = f"Asset ID: {asset_id}"
        dim_label = f"Resolution: {width} x {height}"
        model_label = "Model: CyberPony SDXL + FaceDetailer"

        draw.text((50, 60), title, fill=(255, 215, 0))
        draw.text((50, 100), asset_label, fill=(255, 255, 255))
        draw.text((50, 130), dim_label, fill=(200, 200, 200))
        draw.text((50, 160), model_label, fill=(180, 180, 220))

        # Wrap and draw prompt snippet
        snippet = f"Prompt: {prompt[:300]}..."
        y_pos = 220
        words = snippet.split()
        line = ""
        for word in words:
            if len(line + " " + word) > 55:
                draw.text((50, y_pos), line, fill=(220, 220, 220))
                y_pos += 25
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            draw.text((50, y_pos), line, fill=(220, 220, 220))

        img.save(target_path)
        return target_path
