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
    FLUX_KLEIN_UNET,
    FLUX_KLEIN_CLIP,
    FLUX_KLEIN_VAE,
    FLUX_KLEIN_STEPS,
    FLUX_KLEIN_GUIDANCE,
    FLUX_KLEIN_SAMPLER,
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

    @classmethod
    def build_flux_klein_workflow(
        cls,
        asset_id: str,
        post_type: str,
        asset_type: str,
        positive_prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        avatar_image_name: str = "ZARA_KHAN.png",
        guidance: float = FLUX_KLEIN_GUIDANCE,
        steps: int = FLUX_KLEIN_STEPS
    ) -> Dict[str, Any]:
        """
        Builds a ComfyUI prompt API workflow for Flux 2 Klein 9B with avatar reference.
        - UNET: flux-2-klein-9b_int8_convrot.safetensors
        - CLIP: qwen_3_8b_fp4mixed.safetensors (type: flux2)
        - VAE: FLUX.2-Klein-Base-9B-VAE.safetensors
        - Avatar Conditioning: ReferenceLatent with ZARA_KHAN.png
        - Sampler: SamplerCustomAdvanced + Flux2Scheduler + Euler
        """
        if seed is None:
            seed = random.randint(100000000000, 999999999999)

        dim = cls.get_dimensions(post_type, asset_type)
        width = dim["width"]
        height = dim["height"]

        workflow = {
            "1": {
                "class_type": "UNETLoader",
                "_meta": {"title": "Flux 2 Klein 9B UNET"},
                "inputs": {
                    "unet_name": FLUX_KLEIN_UNET,
                    "weight_dtype": "default"
                }
            },
            "2": {
                "class_type": "CLIPLoader",
                "_meta": {"title": "Qwen 3 8B Text Encoder (Flux2)"},
                "inputs": {
                    "clip_name": FLUX_KLEIN_CLIP,
                    "type": "flux2"
                }
            },
            "3": {
                "class_type": "VAELoader",
                "_meta": {"title": "Flux 2 Klein VAE"},
                "inputs": {
                    "vae_name": FLUX_KLEIN_VAE
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["2", 0]
                }
            },
            "5": {
                "class_type": "FluxGuidance",
                "_meta": {"title": "Flux Guidance Scale"},
                "inputs": {
                    "guidance": guidance,
                    "conditioning": ["4", 0]
                }
            },
            "6": {
                "class_type": "LoadImage",
                "_meta": {"title": "Avatar Reference Image (Zara Khan)"},
                "inputs": {
                    "image": avatar_image_name
                }
            },
            "7": {
                "class_type": "VAEEncode",
                "_meta": {"title": "Encode Avatar to Latent"},
                "inputs": {
                    "pixels": ["6", 0],
                    "vae": ["3", 0]
                }
            },
            "8": {
                "class_type": "ReferenceLatent",
                "_meta": {"title": "Condition with Avatar Latent"},
                "inputs": {
                    "conditioning": ["5", 0],
                    "latent": ["7", 0]
                }
            },
            "9": {
                "class_type": "EmptyFlux2LatentImage",
                "_meta": {"title": f"Target Latent ({width}x{height})"},
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "10": {
                "class_type": "Flux2Scheduler",
                "_meta": {"title": "Flux 2 Scheduler"},
                "inputs": {
                    "steps": steps,
                    "width": width,
                    "height": height
                }
            },
            "11": {
                "class_type": "KSamplerSelect",
                "_meta": {"title": "Sampler Selection"},
                "inputs": {
                    "sampler_name": FLUX_KLEIN_SAMPLER
                }
            },
            "12": {
                "class_type": "RandomNoise",
                "_meta": {"title": "Random Noise"},
                "inputs": {
                    "noise_seed": seed
                }
            },
            "13": {
                "class_type": "BasicGuider",
                "_meta": {"title": "Basic Guider"},
                "inputs": {
                    "model": ["1", 0],
                    "conditioning": ["8", 0]
                }
            },
            "14": {
                "class_type": "SamplerCustomAdvanced",
                "_meta": {"title": "Custom Advanced Sampler"},
                "inputs": {
                    "noise": ["12", 0],
                    "guider": ["13", 0],
                    "sampler": ["11", 0],
                    "sigmas": ["10", 0],
                    "latent_image": ["9", 0]
                }
            },
            "15": {
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode to Image"},
                "inputs": {
                    "samples": ["14", 0],
                    "vae": ["3", 0]
                }
            },
            "16": {
                "class_type": "SaveImage",
                "_meta": {"title": "Save Rendered Image"},
                "inputs": {
                    "filename_prefix": f"ZaraKhan_{asset_id}",
                    "images": ["15", 0]
                }
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
