import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .config import (
    ROOT_DIR,
    CYBERPONY_CHECKPOINT,
    CYBERPONY_YOLO_MODEL,
    CYBERPONY_STEPS,
    CYBERPONY_CFG,
    CYBERPONY_SAMPLER,
    CYBERPONY_SCHEDULER,
    CYBERPONY_FACE_STEPS,
    CYBERPONY_FACE_CFG,
    CYBERPONY_FACE_DENOISE,
    ASPECT_RATIOS
)

WORKFLOWS_DIR = ROOT_DIR / "workflows"
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

class WorkflowBuilder:
    """Constructs ComfyUI API prompt workflows for CyberPony SDXL + FaceDetailer."""

    @staticmethod
    def get_dimensions(post_type: str, asset_type: str) -> Dict[str, int]:
        """Resolves dynamic aspect ratio dimensions based on post/asset type."""
        combined = f"{post_type} {asset_type}".lower()
        if "story" in combined:
            return ASPECT_RATIOS["Story"]
        elif "carousel" in combined or "post" in combined:
            return ASPECT_RATIOS["Post"]
        else:
            return ASPECT_RATIOS["Square"]

    @classmethod
    def build_cyberpony_workflow(
        cls,
        asset_id: str,
        post_type: str,
        asset_type: str,
        positive_prompt: str,
        negative_prompt: str,
        seed: Optional[int] = None,
        enable_face_detailer: bool = True
    ) -> Dict[str, Any]:
        """
        Builds a ComfyUI prompt API workflow for CyberPony SDXL with FaceDetailer.
        Pass 1: Base SDXL diffusion via cyberrealisticPony_v180Coreshift.safetensors.
        Pass 2: Face & eye refinement via Ultralytics YOLOv8m + FaceDetailer.
        """
        if seed is None:
            seed = random.randint(100000000000, 999999999999)

        dim = cls.get_dimensions(post_type, asset_type)
        width = dim["width"]
        height = dim["height"]

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "CyberPony SDXL Checkpoint"},
                "inputs": {
                    "ckpt_name": CYBERPONY_CHECKPOINT
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt (Pony Syntax)"},
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt (Pony Filter)"},
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "_meta": {"title": f"Dynamic Latent ({width}x{height})"},
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "_meta": {"title": "Pass 1: Base SDXL Sampler"},
                "inputs": {
                    "seed": seed,
                    "steps": CYBERPONY_STEPS,
                    "cfg": CYBERPONY_CFG,
                    "sampler_name": CYBERPONY_SAMPLER,
                    "scheduler": CYBERPONY_SCHEDULER,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "_meta": {"title": "Base VAE Decode"},
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            }
        }

        if enable_face_detailer:
            workflow["7"] = {
                "class_type": "UltralyticsDetectorProvider",
                "_meta": {"title": "YOLO Face Detector"},
                "inputs": {
                    "model_name": CYBERPONY_YOLO_MODEL
                }
            }
            workflow["8"] = {
                "class_type": "FaceDetailer",
                "_meta": {"title": "Pass 2: Face & Eye Refiner"},
                "inputs": {
                    "image": ["6", 0],
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "vae": ["1", 2],
                    "guide_size": 512,
                    "guide_size_for": True,
                    "max_size": 1024,
                    "seed": seed + 1,
                    "steps": CYBERPONY_FACE_STEPS,
                    "cfg": CYBERPONY_FACE_CFG,
                    "sampler_name": CYBERPONY_SAMPLER,
                    "scheduler": CYBERPONY_SCHEDULER,
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "denoise": CYBERPONY_FACE_DENOISE,
                    "feather": 5,
                    "noise_mask": True,
                    "force_inpaint": True,
                    "bbox_threshold": 0.5,
                    "bbox_dilation": 10,
                    "bbox_crop_factor": 3.0,
                    "sam_detection_hint": "center-1",
                    "sam_dilation": 0,
                    "sam_threshold": 0.93,
                    "sam_bbox_expansion": 0,
                    "sam_mask_hint_threshold": 0.7,
                    "sam_mask_hint_use_negative": "False",
                    "drop_size": 10,
                    "bbox_detector": ["7", 0],
                    "wildcard": "",
                    "cycle": 1
                }
            }
            workflow["9"] = {
                "class_type": "SaveImage",
                "_meta": {"title": "Save Refined Output"},
                "inputs": {
                    "filename_prefix": f"ZaraKhan_{asset_id}",
                    "images": ["8", 0]
                }
            }
        else:
            workflow["9"] = {
                "class_type": "SaveImage",
                "_meta": {"title": "Save Base Output"},
                "inputs": {
                    "filename_prefix": f"ZaraKhan_{asset_id}",
                    "images": ["6", 0]
                }
            }

        # Save workflow JSON to disk for inspection
        wf_file = WORKFLOWS_DIR / f"{asset_id}_workflow.json"
        try:
            with open(wf_file, "w", encoding="utf-8") as f:
                json.dump({"prompt": workflow}, f, indent=2)
        except Exception:
            pass

        return {"prompt": workflow}

    # Backward-compatible alias
    @classmethod
    def build_krea2_workflow(cls, *args, **kwargs):
        """Deprecated alias mapped directly to build_cyberpony_workflow."""
        # Strip legacy krea2 kwargs if passed
        kwargs.pop("avatar_image_name", None)
        kwargs.pop("use_vision_reference", None)
        return cls.build_cyberpony_workflow(*args, **kwargs)
