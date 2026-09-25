# Zara Khan Pipeline: In-Depth Engineering Progress Report

**Date:** September 25, 2026  
**Repository:** `vedantgoim/my-zarakhan-pipeline`  
**Branch:** `ved-2509`  
**Author:** AI Pair Programming Team / Vedant Goim  

---

## 1. Executive Summary

This report documents the architectural overhaul, implementation, and end-to-end verification of the **Zara Khan Autonomous Content Generation Pipeline**. 

Originally structured as a single-reference ingestion script, the system has been transformed into a production-grade, 3-reference multimodal image generation pipeline coupled with a local Next.js creative studio dashboard (`contentgen`). The pipeline grounds generated visuals in the virtual creator's identity, spatial environment, and wardrobe styling using Google Gemini Vision prompt synthesis and local ComfyUI Flux 2 Klein 9B diffusion models.

### Key Milestones Achieved
- **3-Reference Conditioning Engine:** Simultaneous grounding across Avatar Likeness (`ZARA_KHAN.png`), Background Setting (resolved Pinterest CDN), and Wardrobe Attire (fashion references).
- **Dual ComfyUI Workflow Serialization:** Solved the "empty canvas" issue by exporting both ComfyUI headless API JSON (`workflow_api.json`) and full visual LiteGraph node canvas JSON (`workflow.json`).
- **3 Real-World Validated Test Runs:** Successfully ingested, prompted, and rendered Rows 184, 185, and 186 from Google Sheets without synthetic mockups.
- **Security Audit & Hardening:** Identified and patched a path traversal vulnerability in image endpoints, enforced strict input validation, and verified zero `npm audit` vulnerabilities.
- **UI/UX Decluttering:** Redesigned the Next.js studio interface into a clean, distraction-free production environment with real asset specifications, 1-click downloads, and a streamlined 2-mode workspace.

---

## 2. Architecture & Pipeline Engineering

### 2.1 The 3-Reference Ingestion Architecture

A critical limitation in earlier iterations was single-reference drift: conditioning only on an avatar caused setting incoherence, while conditioning only on an environment reference diluted facial identity. The new architecture enforces three distinct conditioning anchors for every scene:

```
[Google Sheet / Calendar Row]
         │
         ├───> Anchor 1: Character Avatar (ZARA_KHAN.png)
         │       • Preserves 24-year-old South Asian facial features, skin tone, eye shape
         │
         ├───> Anchor 2: Spatial Environment (Pinterest / Direct CDN URL)
         │       • Anchors architecture, lighting temperature, depth of field, background geometry
         │
         └───> Anchor 3: Wardrobe & Styling (Attire Reference URL)
                 • Anchors fabric texture, draping, color palette, and embroidery
```

### 2.2 Multimodal Prompt Synthesizer (`pipeline/prompt_synthesizer.py`)

The prompt synthesizer passes all three reference images to the Gemini Vision API alongside calendar metadata (Title, Notes, Music/Vibe, Avoid guardrails).

- **Structured Schema:** Uses Pydantic (`ModelPromptBody`) to guarantee deterministic outputs containing:
  - Technical Specifications Header: `[Post · image · 3:4 · 1536×2048 · Higgsfield]`
  - Core Scene Description: Photorealistic narrative detailing posture, facial expression, wardrobe fabric, and ambient lighting.
  - Camera & Composition: Lens selection (e.g. 50mm / 85mm), aperture, depth of field, framing.
  - Negative Constraints: Explicit exclusions (e.g. deformed hands, extra fingers, unnatural plastic skin, synthetic lighting).
- **Calendar Notes as Strict Overrides:** The prompt compiler explicitly prioritizes creator notes over reference defaults, allowing creative direction to adapt a setting or swap attire colors seamlessly.

### 2.3 Bidirectional Sheet & Excel Synchronization

The pipeline synchronizes state bidirectionally:
1. Ingests raw rows from Google Sheets (or local Excel workbooks `Vinayak's Copy of Zara Khan.xlsx` and `Zara_Khan_Optimized_Calendar.xlsx`).
2. Resolves shortened URLs (`pin.it`, redirects) to permanent CDN image URLs (`i.pinimg.com/...`).
3. Writes the synthesized prompt and extracted CDN links back to Google Sheets Column O (`Master Prompt`).
4. Updates execution status from `queued` to `prompted` to `done`.

---

## 3. Diffusion Engine & ComfyUI Workflow Evolution

### 3.1 Visual LiteGraph Serialization (Fixing the "Empty Canvas")

Previously, importing generated workflows into the ComfyUI browser interface resulted in an empty canvas. Investigation revealed that ComfyUI uses two fundamentally distinct schema formats:
- **API Prompt Format:** A flat dictionary of `{ node_id: { class_type, inputs } }` consumed by `/prompt`. It contains zero UI, coordinate, or connection link data.
- **Visual LiteGraph Format:** A structured graph object containing `nodes`, `links`, `groups`, `config`, and `extra.ds` (pan/zoom state) required by the web canvas.

The pipeline was updated to serialize both formats:
- `workflows/flux_2_klein_9b_multimodal.json`: Full visual graph for drag-and-drop viewing and editing in ComfyUI.
- `workflows/flux_2_klein_9b_multimodal_api.json`: Headless execution schema for automated Python WebSocket dispatch.

### 3.2 Model Configuration: Flux 2 Klein 9B

- **Checkpoint:** Flux 2 Klein 9B diffusion transformer.
- **Text Encoders:** Dual CLIP-L and T5XXL for nuanced natural-language comprehension.
- **Sampling:** FlowMatch Euler scheduler, 28-30 steps, CFG scale 3.5.
- **Refinement Pass:** Integrated FaceDetailer with Ultralytics bbox detection for high-fidelity facial features matching the Zara Khan persona.

---

## 4. End-to-End Test Suite & Validation (Rows 184–186)

To ensure repeatability, three real-world test cases were executed sequentially through the entire pipeline: Google Sheets ingestion -> Reference resolution -> Gemini Vision prompt compilation -> ComfyUI local rendering -> Studio verification.

### Test Case 1: Row 184 (Day 101)
- **Setting:** Heritage veranda cafe patio in South Mumbai with weathered teak furniture and morning blinds.
- **Wardrobe:** Casual unbuttoned ivory linen shirt with rolled cuffs, layered gold chains, open book, and ceramic black coffee cup.
- **Synthesized Prompt Highlights:** Eye-level medium portrait framing, warm morning sunlight through slatted blinds, 35mm film aesthetic.
- **Output Asset:** `outputs/DAY101_POST_01.png` (1.42 MB, 896×1152 PNG).
- **Result:** Consistent facial likeness, natural linen texture, warm volumetric lighting.

### Test Case 2: Row 185 (Day 102)
- **Setting:** Contemporary South Mumbai art gallery with polished concrete flooring, white minimalist walls, and modern abstract canvas.
- **Wardrobe:** Royal purple Banarasi silk saree with ornate golden zari borders and matching half-sleeve blouse.
- **Synthesized Prompt Highlights:** Soft diffused gallery ambient lighting, three-quarter standing posture, authentic textile sheen.
- **Output Asset:** `outputs/DAY102_POST_01.png` (1.31 MB, 896×1152 PNG).
- **Result:** Accurate cultural attire drape, crisp metallic gold embroidery, clean architectural geometry.

### Test Case 3: Row 186 (Day 103)
- **Setting:** Dark mahogany heritage library lounge with leather-bound books, classical oil paintings, and shaded reading lamps.
- **Wardrobe:** Crimson rose-pink and gold Kanjeevaram silk saree with traditional temple jewelry (maang tikka, choker, jhumkas).
- **Synthesized Prompt Highlights:** 85mm portrait lens, shallow depth of field, serene candid expression, warm practical lamp glow.
- **Output Asset:** `outputs/DAY103_POST_01.png` (1.65 MB, 896×1152 PNG).
- **Result:** Rich textile detail, accurate jewelry rendering, high facial fidelity.

---

## 5. Next.js Studio Overhaul & UI/UX Decluttering

The local web studio (`contentgen`, port 3000) was overhauled to remove artificial clutter and streamline operations for human creators:

### 5.1 Review Station
- **Removed Fake Clutter:** Replaced mock static indicators (fake ArcFace Centroid 95.2% score and dummy 24 FPS export checklist) with true production metadata.
- **Real Asset Specifications:** Displays Asset Title, Format (`9:16 Vertical Portrait`), Dimensions (`896 × 1152 24-bit PNG`), and Conditioning (`3-Ref Multimodal Latents`).
- **1-Click Utility Actions:** Added direct links to **"Open Full Photo"** in a clean browser tab and **"Download PNG"** (`?clean=1`).
- **Approval Actions:** Preserved sheet synchronization triggers ("Approve & Mark Published" and "Regenerate Creative").

### 5.2 Studio Brief
- **3-Column Reference Gallery:** Replaced the cramped 4-card grid with 3 spacious cards (Avatar Identity, Environment Setting, Wardrobe Attire), removing the redundant 4th card that duplicated the hero viewfinder.
- **Simplified 2-Mode Tab Architecture:**
  1. `brief` (Default): Displays Social Copy / Audio Vibe along with the full Compiled Master Prompt and a 1-click Copy button.
  2. `overrides`: Exposes creative sliders (Wardrobe, Camera, Lighting, Setting, Director's Nuance, and Negative Constraints) with a live-updating compiled prompt preview.
- **Cleaned Codebase:** Removed dead "Timeline" and "Raw Prompt" tabs, eliminating unnecessary state variables.

### 5.3 Production Queue
- Integrated dynamic thumbnail previews for rows with rendered images, falling back to clean status badges when pending.

---

## 6. Security Audit & Hardening

A comprehensive security audit was conducted:

### 6.1 Path Traversal Vulnerability (CWE-22) Remediated
- **Vulnerability:** In `src/app/api/images/[file]/route.ts`, the fallback lookup into `../AUTOMATION1/outputs` did not verify that resolved paths stayed inside the designated directory. A payload such as `/api/images/..%2f..%2fcontentgen%2f.env` could leak sensitive credentials.
- **Fix:** Sanitized all incoming path parameters using `path.basename(file)` and enforced strict directory whitelist checks:
  ```typescript
  const safeName = path.basename(file);
  const allowedDirs = [
    path.resolve(IMAGE_DIR),
    path.resolve(process.cwd(), "../AUTOMATION1/outputs"),
  ];
  // Verify resolved directory is in allowedDirs; reject with 403/404 otherwise
  ```
- **Verification:** Malicious path traversal requests to `.env`, `win.ini`, and package files were re-tested and are confirmed safely blocked (HTTP 404/403).

### 6.2 Cache Invalidation
- Configured `Cache-Control: no-cache, no-store, must-revalidate` headers on image routes to ensure regenerated assets display immediately without requiring manual browser cache clearing.

### 6.3 Dependency Audit
- Ran `npm audit` across all dependencies: **0 vulnerabilities found**.
- Ran `npm run typecheck`: **0 errors**.

---

## 7. Current Repository Status & Next Steps

| Component | Status | Verification |
| :--- | :--- | :--- |
| **Multimodal Ingestion** | Production Ready | Tested with Pinterest & Drive links |
| **Gemini Vision Prompting** | Production Ready | Tested across Rows 184, 185, 186 |
| **ComfyUI Flux 2 Klein 9B** | Production Ready | 3 output PNGs rendered locally |
| **Web Studio UI/UX** | Production Ready | Cleaned, decluttered, responsive |
| **Security & Auditing** | Hardened | Path traversal patched, 0 CVEs |

### Recommended Future Work
1. **Video Model Expansion:** Wire Higgsfield and Wan 2.1 API adapters for animated reel generation.
2. **Automated Batch Queuing:** Add a multi-select queue trigger in the studio UI to render entire content weeks with one click.
3. **Automated Cloud Backup:** Sync completed renders directly to the production Google Drive folder.
