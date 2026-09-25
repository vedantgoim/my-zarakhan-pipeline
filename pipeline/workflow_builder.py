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

        # Build corresponding visual LiteGraph workflow for ComfyUI GUI canvas
        visual_workflow = {
            "id": f"zara-khan-flux-klein-{asset_id}",
            "revision": 0,
            "last_node_id": 16,
            "last_link_id": 16,
            "nodes": [
                {
                    "id": 1,
                    "type": "UNETLoader",
                    "pos": [-800, 100],
                    "size": [320, 90],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [
                        {"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": None},
                        {"name": "weight_dtype", "type": "COMBO", "widget": {"name": "weight_dtype"}, "link": None}
                    ],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}],
                    "title": "Flux 2 Klein 9B UNET",
                    "properties": {"Node name for S&R": "UNETLoader"},
                    "widgets_values": [FLUX_KLEIN_UNET, "default"]
                },
                {
                    "id": 2,
                    "type": "CLIPLoader",
                    "pos": [-800, 230],
                    "size": [320, 110],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {"name": "clip_name", "type": "COMBO", "widget": {"name": "clip_name"}, "link": None},
                        {"name": "type", "type": "COMBO", "widget": {"name": "type"}, "link": None},
                        {"name": "device", "type": "COMBO", "widget": {"name": "device"}, "link": None}
                    ],
                    "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2]}],
                    "title": "Qwen 3 8B Text Encoder",
                    "properties": {"Node name for S&R": "CLIPLoader"},
                    "widgets_values": [FLUX_KLEIN_CLIP, "flux2", "default"]
                },
                {
                    "id": 3,
                    "type": "VAELoader",
                    "pos": [-800, 380],
                    "size": [320, 70],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [{"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}, "link": None}],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": [3, 4]}],
                    "title": "Flux 2 Klein VAE",
                    "properties": {"Node name for S&R": "VAELoader"},
                    "widgets_values": [FLUX_KLEIN_VAE]
                },
                {
                    "id": 4,
                    "type": "CLIPTextEncode",
                    "pos": [-440, 100],
                    "size": [400, 200],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [
                        {"name": "clip", "type": "CLIP", "link": 2},
                        {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}
                    ],
                    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5]}],
                    "title": "Zara Khan Prompt",
                    "properties": {"Node name for S&R": "CLIPTextEncode"},
                    "widgets_values": [positive_prompt]
                },
                {
                    "id": 5,
                    "type": "FluxGuidance",
                    "pos": [-440, 340],
                    "size": [220, 60],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 5},
                        {"name": "guidance", "type": "FLOAT", "widget": {"name": "guidance"}, "link": None}
                    ],
                    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6]}],
                    "title": "Flux Guidance",
                    "properties": {"Node name for S&R": "FluxGuidance"},
                    "widgets_values": [guidance]
                },
                {
                    "id": 6,
                    "type": "LoadImage",
                    "pos": [-800, 490],
                    "size": [320, 314],
                    "flags": {},
                    "order": 5,
                    "mode": 0,
                    "inputs": [
                        {"name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": None},
                        {"name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None}
                    ],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [7]},
                        {"name": "MASK", "type": "MASK", "links": None}
                    ],
                    "title": "Avatar Reference (Zara Khan)",
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": [avatar_image_name, "image"]
                },
                {
                    "id": 7,
                    "type": "VAEEncode",
                    "pos": [-440, 490],
                    "size": [220, 60],
                    "flags": {},
                    "order": 6,
                    "mode": 0,
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": 7},
                        {"name": "vae", "type": "VAE", "link": 3}
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [8]}],
                    "title": "Encode Avatar to Latent",
                    "properties": {"Node name for S&R": "VAEEncode"}
                },
                {
                    "id": 8,
                    "type": "ReferenceLatent",
                    "pos": [-180, 340],
                    "size": [220, 60],
                    "flags": {},
                    "order": 7,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 6},
                        {"name": "latent", "type": "LATENT", "link": 8}
                    ],
                    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [9]}],
                    "title": "Avatar Reference Latent",
                    "properties": {"Node name for S&R": "ReferenceLatent"}
                },
                {
                    "id": 9,
                    "type": "EmptyFlux2LatentImage",
                    "pos": [-440, 590],
                    "size": [280, 110],
                    "flags": {},
                    "order": 8,
                    "mode": 0,
                    "inputs": [
                        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
                        {"name": "batch_size", "type": "INT", "widget": {"name": "batch_size"}, "link": None}
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [10]}],
                    "title": "Target Latent Dimensions",
                    "properties": {"Node name for S&R": "EmptyFlux2LatentImage"},
                    "widgets_values": [width, height, 1]
                },
                {
                    "id": 10,
                    "type": "Flux2Scheduler",
                    "pos": [-180, 440],
                    "size": [240, 110],
                    "flags": {},
                    "order": 9,
                    "mode": 0,
                    "inputs": [
                        {"name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None},
                        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None}
                    ],
                    "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [11]}],
                    "title": "Flux 2 Scheduler",
                    "properties": {"Node name for S&R": "Flux2Scheduler"},
                    "widgets_values": [steps, width, height]
                },
                {
                    "id": 11,
                    "type": "KSamplerSelect",
                    "pos": [-180, 590],
                    "size": [240, 60],
                    "flags": {},
                    "order": 10,
                    "mode": 0,
                    "inputs": [{"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": None}],
                    "outputs": [{"name": "SAMPLER", "type": "SAMPLER", "links": [12]}],
                    "title": "Sampler Selection",
                    "properties": {"Node name for S&R": "KSamplerSelect"},
                    "widgets_values": [FLUX_KLEIN_SAMPLER]
                },
                {
                    "id": 12,
                    "type": "RandomNoise",
                    "pos": [-180, 690],
                    "size": [240, 80],
                    "flags": {},
                    "order": 11,
                    "mode": 0,
                    "inputs": [{"name": "noise_seed", "type": "INT", "widget": {"name": "noise_seed"}, "link": None}],
                    "outputs": [{"name": "NOISE", "type": "NOISE", "links": [13]}],
                    "title": "Seed / Noise",
                    "properties": {"Node name for S&R": "RandomNoise"},
                    "widgets_values": [seed, "randomize"]
                },
                {
                    "id": 13,
                    "type": "BasicGuider",
                    "pos": [100, 100],
                    "size": [240, 60],
                    "flags": {},
                    "order": 12,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 1},
                        {"name": "conditioning", "type": "CONDITIONING", "link": 9}
                    ],
                    "outputs": [{"name": "GUIDER", "type": "GUIDER", "links": [14]}],
                    "title": "Basic Guider",
                    "properties": {"Node name for S&R": "BasicGuider"}
                },
                {
                    "id": 14,
                    "type": "SamplerCustomAdvanced",
                    "pos": [380, 100],
                    "size": [280, 160],
                    "flags": {},
                    "order": 13,
                    "mode": 0,
                    "inputs": [
                        {"name": "noise", "type": "NOISE", "link": 13},
                        {"name": "guider", "type": "GUIDER", "link": 14},
                        {"name": "sampler", "type": "SAMPLER", "link": 12},
                        {"name": "sigmas", "type": "SIGMAS", "link": 11},
                        {"name": "latent_image", "type": "LATENT", "link": 10}
                    ],
                    "outputs": [
                        {"name": "output", "type": "LATENT", "links": [15]},
                        {"name": "denoised_output", "type": "LATENT", "links": []}
                    ],
                    "title": "Sampler Custom Advanced",
                    "properties": {"Node name for S&R": "SamplerCustomAdvanced"}
                },
                {
                    "id": 15,
                    "type": "VAEDecode",
                    "pos": [700, 100],
                    "size": [210, 60],
                    "flags": {},
                    "order": 14,
                    "mode": 0,
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": 15},
                        {"name": "vae", "type": "VAE", "link": 4}
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                    "title": "VAE Decode",
                    "properties": {"Node name for S&R": "VAEDecode"}
                },
                {
                    "id": 16,
                    "type": "SaveImage",
                    "pos": [950, 100],
                    "size": [400, 480],
                    "flags": {},
                    "order": 15,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 16},
                        {"name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None}
                    ],
                    "outputs": [],
                    "title": "Save Rendered Image",
                    "properties": {"Node name for S&R": "SaveImage"},
                    "widgets_values": [f"ZaraKhan_{asset_id}"]
                }
            ],
            "links": [
                [1, 1, 0, 13, 0, "MODEL"],
                [2, 2, 0, 4, 0, "CLIP"],
                [3, 3, 0, 7, 1, "VAE"],
                [4, 3, 0, 15, 1, "VAE"],
                [5, 4, 0, 5, 0, "CONDITIONING"],
                [6, 5, 0, 8, 0, "CONDITIONING"],
                [7, 6, 0, 7, 0, "IMAGE"],
                [8, 7, 0, 8, 1, "LATENT"],
                [9, 8, 0, 13, 1, "CONDITIONING"],
                [10, 9, 0, 14, 4, "LATENT"],
                [11, 10, 0, 14, 3, "SIGMAS"],
                [12, 11, 0, 14, 2, "SAMPLER"],
                [13, 12, 0, 14, 0, "NOISE"],
                [14, 13, 0, 14, 1, "GUIDER"],
                [15, 14, 0, 15, 0, "LATENT"],
                [16, 15, 0, 16, 0, "IMAGE"]
            ],
            "groups": [
                {"id": 1, "title": "1. Base Model & VAE Loaders", "bounding": [-820, 40, 360, 430], "color": "#3f51b5", "font_size": 24},
                {"id": 2, "title": "2. Zara Khan Avatar Conditioning", "bounding": [-820, 460, 620, 370], "color": "#ff9800", "font_size": 24},
                {"id": 3, "title": "3. Prompt & Guidance", "bounding": [-460, 40, 440, 380], "color": "#4caf50", "font_size": 24},
                {"id": 4, "title": "4. Flux 2 Sampler & Decode", "bounding": [60, 40, 1310, 560], "color": "#9c27b0", "font_size": 24}
            ],
            "config": {},
            "extra": {"ds": {"scale": 0.85, "offset": [900, 200]}},
            "version": 0.4
        }

        # Save visual LiteGraph workflow to disk for GUI drag-and-drop
        try:
            wf_file = WORKFLOWS_DIR / f"{asset_id}_workflow.json"
            with open(wf_file, "w", encoding="utf-8") as f:
                json.dump(visual_workflow, f, indent=2)
            # Also save master template
            with open(WORKFLOWS_DIR / "flux_klein_9b_avatar_workflow.json", "w", encoding="utf-8") as f:
                json.dump(visual_workflow, f, indent=2)
            # Sync to ComfyUI user default workflows directory
            comfy_user_wf = Path(r"E:\Comfy-Desktop\ComfyUI-Installs\astor\ComfyUI\user\default\workflows\Zara_Khan_Flux_Klein_9B.json")
            with open(comfy_user_wf, "w", encoding="utf-8") as f:
                json.dump(visual_workflow, f, indent=2)
        except Exception:
            pass

        return {
            "prompt": workflow,
            "extra_data": {
                "extra_pnginfo": {
                    "workflow": visual_workflow
                }
            }
        }

    # Backward-compatible alias
    @classmethod
    def build_krea2_workflow(cls, *args, **kwargs):
        """Deprecated alias mapped directly to build_cyberpony_workflow."""
        # Strip legacy krea2 kwargs if passed
        kwargs.pop("avatar_image_name", None)
        kwargs.pop("use_vision_reference", None)
        return cls.build_cyberpony_workflow(*args, **kwargs)
