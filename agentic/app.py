#!/usr/bin/env python3
"""
Row-based Gradio UI - One row per object with masked image, generated image, edit prompt, and 3D button.
"""

import os
import sys

# CRITICAL: Set CUDA memory allocator configuration BEFORE any torch imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Force use of cached models - don't check remote servers
# Models will be downloaded on first run, then cached forever
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from backend.orchestrator import AgenticOrchestrator
from backend.models import Job

# Global state
current_job: Optional[Job] = None
object_rows: Dict[str, Dict[str, Any]] = {}  # Store Gradio components for each object


def initialize_orchestrator() -> AgenticOrchestrator:
    """Initialize the orchestrator."""
    # Use default config path for SAM3D (will use ../checkpoints/hf/pipeline.yaml)
    # The path is relative to where sam3d_service.py is located
    sam3d_config = "../checkpoints/hf/pipeline.yaml"  # Default, relative to backend/services/
    storage_base_dir = str(Path(__file__).parent / "data" / "jobs")
    
    return AgenticOrchestrator(
        sam3d_config=sam3d_config,
        storage_base_dir=storage_base_dir
    )


def upload_and_segment(image: np.ndarray) -> Tuple[str, str, str, Dict]:
    """Upload image and segment objects."""
    global current_job
    
    if image is None:
        return "❌ Please upload an image", None, "", gr.update(visible=False)
    
    try:
        orch = initialize_orchestrator()
        pil_image = Image.fromarray(image)
        
        logger.info("Creating job from uploaded image...")
        current_job = orch.create_job_from_image(pil_image)
        
        logger.info("Segmenting objects...")
        current_job, overlay_image = orch.segment_image(current_job)
        
        shareable_url = f"Job ID: {current_job.job_id}\nAccess at: http://0.0.0.0:7860/?job_id={current_job.job_id}"
        
        status = f"✅ Segmentation complete! Found {len(current_job.objects)} objects.\n"
        status += f"Click 'Generate Clean Images' to process all objects."
        
        # Show the objects container
        return status, current_job.overlay_mask_path, shareable_url, gr.update(visible=True)
        
    except Exception as e:
        logger.error(f"Upload and segment failed: {e}")
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}", None, "", gr.update(visible=False)


def generate_all_clean_images() -> Tuple[str, List[Dict]]:
    """Generate clean images for all objects and return updates for each row."""
    global current_job
    
    if current_job is None:
        return "❌ Please upload and segment an image first", []
    
    try:
        orch = initialize_orchestrator()
        
        logger.info("Generating clean images for all objects...")
        orch.generate_clean_images(current_job)
        
        # Reload job to get updated paths
        current_job = orch.get_job(current_job.job_id)
        
        # Prepare updates for all object rows
        updates = []
        successful_count = 0
        
        for obj in current_job.objects:
            obj_update = {
                'object_id': obj.object_id,
                'label': obj.label,
                'masked_path': obj.masked_original_path if obj.masked_original_path and os.path.exists(obj.masked_original_path) else None,
                'generated_path': obj.generated_image_path if obj.generated_image_path and os.path.exists(obj.generated_image_path) else None,
                'has_3d': obj.asset_3d is not None and obj.asset_3d.asset_path and os.path.exists(obj.asset_3d.asset_path),
                '3d_path': obj.asset_3d.asset_path if obj.asset_3d and obj.asset_3d.asset_path else None
            }
            updates.append(obj_update)
            
            if obj_update['generated_path']:
                successful_count += 1
        
        status = f"✅ Clean images generated!\n"
        status += f"Success: {successful_count}/{len(current_job.objects)} objects\n"
        status += "You can now edit styles or generate 3D assets."
        
        return status, updates
        
    except Exception as e:
        logger.error(f"Clean image generation failed: {e}")
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}", []


def edit_single_object(object_id: str, edit_prompt: str) -> Tuple[str, str]:
    """Edit a specific object's image."""
    global current_job
    
    if current_job is None:
        return "❌ No job loaded", None
    
    if not edit_prompt.strip():
        return "❌ Please enter an edit prompt", None
    
    try:
        orch = initialize_orchestrator()
        
        logger.info(f"Editing object {object_id} with prompt: {edit_prompt}")
        orch.edit_object_image(current_job, object_id, edit_prompt)
        
        # Reload job
        current_job = orch.get_job(current_job.job_id)
        
        # Find the object and return updated image
        for obj in current_job.objects:
            if obj.object_id == object_id:
                if obj.edited_image_path and os.path.exists(obj.edited_image_path):
                    return f"✅ {obj.label} edited successfully", obj.edited_image_path
                elif obj.generated_image_path and os.path.exists(obj.generated_image_path):
                    return f"✅ {obj.label} edited successfully", obj.generated_image_path
        
        return "⚠️ Edit completed but image not found", None
        
    except Exception as e:
        logger.error(f"Edit failed: {e}")
        return f"❌ Error: {str(e)}", None


def generate_3d_single_object(object_id: str) -> Tuple[str, str]:
    """Generate 3D asset for a specific object."""
    global current_job
    
    if current_job is None:
        return "❌ No job loaded", None
    
    try:
        orch = initialize_orchestrator()
        
        logger.info(f"Generating 3D asset for {object_id}")
        
        # Submit and process
        orch.submit_3d_generation(current_job, [object_id])
        results = orch.process_3d_queue(max_iterations=1)
        
        # Reload job
        current_job = orch.get_job(current_job.job_id)
        
        # Find the object and return 3D asset path
        for obj in current_job.objects:
            if obj.object_id == object_id:
                if obj.asset_3d:
                    # Try GLB first (better for web viewing), then PLY
                    asset_path = obj.asset_3d.glb_path or obj.asset_3d.ply_path
                    if asset_path and os.path.exists(asset_path):
                        return f"✅ 3D asset generated for {obj.label}", asset_path
        
        return "⚠️ 3D generation completed but asset not found", None
        
    except Exception as e:
        logger.error(f"3D generation failed: {e}")
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}", None


def run_full_pipeline(image: np.ndarray):
    """
    Generator that runs the full pipeline and yields after each major step so the UI
    updates incrementally: segmentation → clean images → each 3D asset.
    """
    global current_job
    
    NUM_ROWS = 20
    
    def _empty_outputs():
        err = "❌ Please upload an image"
        return (
            err, None, "", gr.update(visible=False),
            *([gr.update(visible=False)] * NUM_ROWS),
            *([None] * NUM_ROWS),
            "",
            *([None] * NUM_ROWS),
            *([None] * NUM_ROWS),
            *([None] * NUM_ROWS),
            *([gr.update(visible=False)] * NUM_ROWS),
        )
    
    if image is None:
        yield _empty_outputs()
        return
    
    try:
        orch = initialize_orchestrator()
        pil_image = Image.fromarray(image)
        
        # Helper to build the full output tuple from current job state (optional partial updates).
        def _build_output(
            status_segment_val,
            overlay_path_val,
            status_generate_val,
            generated_paths=None,
            obj_statuses=None,
            model_3d_paths=None,
            viewer_visibilities=None,
        ):
            n = len(current_job.objects) if current_job else 0
            group_updates = [gr.update(visible=True) if i < n else gr.update(visible=False) for i in range(NUM_ROWS)]
            masked_updates = []
            for i in range(NUM_ROWS):
                if i < n and current_job.objects[i].masked_original_path and os.path.exists(current_job.objects[i].masked_original_path):
                    masked_updates.append(current_job.objects[i].masked_original_path)
                else:
                    masked_updates.append(None)
            if generated_paths is None:
                generated_paths = [None] * NUM_ROWS
            if obj_statuses is None:
                obj_statuses = [None] * NUM_ROWS
            if model_3d_paths is None:
                model_3d_paths = [None] * NUM_ROWS
            if viewer_visibilities is None:
                viewer_visibilities = [gr.update(visible=False)] * NUM_ROWS
            # Pad to NUM_ROWS so the tuple length is always correct.
            def _pad(lst, default):
                return (list(lst) + [default] * NUM_ROWS)[:NUM_ROWS]
            generated_paths = _pad(generated_paths, None)
            obj_statuses = _pad(obj_statuses, None)
            model_3d_paths = _pad(model_3d_paths, None)
            viewer_visibilities = _pad(viewer_visibilities, gr.update(visible=False))
            shareable_url = f"Job ID: {current_job.job_id}\nAccess at: http://0.0.0.0:7860/?job_id={current_job.job_id}" if current_job else ""
            return (
                status_segment_val,
                overlay_path_val,
                shareable_url,
                gr.update(visible=True),
                *group_updates,
                *masked_updates,
                status_generate_val,
                *generated_paths,
                *obj_statuses,
                *model_3d_paths,
                *viewer_visibilities,
            )
        
        # 1. Create job and segment
        logger.info("Pipeline: Creating job and segmenting...")
        current_job = orch.create_job_from_image(pil_image)
        current_job, overlay_image = orch.segment_image(current_job)
        overlay_path = current_job.overlay_mask_path
        n_objs = len(current_job.objects)
        status_segment = f"✅ Segmentation complete! Found {n_objs} objects.\nNext: Generating clean images..."
        yield _build_output(status_segment, overlay_path, "")
        
        # 2. Generate clean images for all
        logger.info("Pipeline: Generating clean images for all objects...")
        orch.generate_clean_images(current_job)
        current_job = orch.get_job(current_job.job_id)
        generated_paths = []
        for i in range(NUM_ROWS):
            if i < len(current_job.objects) and current_job.objects[i].generated_image_path and os.path.exists(current_job.objects[i].generated_image_path):
                generated_paths.append(current_job.objects[i].generated_image_path)
            else:
                generated_paths.append(None)
        status_generate = f"✅ Clean images generated for all {n_objs} objects.\nNext: Generating 3D assets one by one..."
        yield _build_output(
            f"✅ Segmentation complete! Found {n_objs} objects.",
            overlay_path,
            status_generate,
            generated_paths=generated_paths,
        )
        
        # 3. Process 3D queue one by one and yield after each
        object_ids = [obj.object_id for obj in current_job.objects]
        orch.submit_3d_generation(current_job, object_ids)
        n_3d_ok = 0
        for idx in range(len(object_ids)):
            logger.info(f"Pipeline: Generating 3D asset {idx + 1}/{len(object_ids)}...")
            results = orch.process_3d_queue(max_iterations=1)
            current_job = orch.get_job(current_job.job_id)
            if results and results[0][2]:
                n_3d_ok += 1
            # Build current state for generated images and 3D
            generated_paths = []
            obj_statuses = []
            model_3d_paths = []
            viewer_visibilities = []
            for i in range(NUM_ROWS):
                if i < len(current_job.objects):
                    obj = current_job.objects[i]
                    generated_paths.append(
                        obj.generated_image_path if obj.generated_image_path and os.path.exists(obj.generated_image_path) else None
                    )
                    if obj.asset_3d:
                        asset_path = obj.asset_3d.glb_path or obj.asset_3d.ply_path
                        if asset_path and os.path.exists(asset_path):
                            obj_statuses.append(f"✅ 3D: {obj.label}")
                            model_3d_paths.append(asset_path)
                            viewer_visibilities.append(gr.update(visible=True))
                        else:
                            obj_statuses.append("")
                            model_3d_paths.append(None)
                            viewer_visibilities.append(gr.update(visible=False))
                    else:
                        obj_statuses.append("")
                        model_3d_paths.append(None)
                        viewer_visibilities.append(gr.update(visible=False))
                else:
                    generated_paths.append(None)
                    obj_statuses.append(None)
                    model_3d_paths.append(None)
                    viewer_visibilities.append(gr.update(visible=False))
            status_segment = f"✅ Segmentation: {n_objs} objects."
            status_generate = f"✅ Clean images done. 3D: {n_3d_ok}/{len(object_ids)} generated (just finished object {idx + 1})."
            yield _build_output(
                status_segment,
                overlay_path,
                status_generate,
                generated_paths=generated_paths,
                obj_statuses=obj_statuses,
                model_3d_paths=model_3d_paths,
                viewer_visibilities=viewer_visibilities,
            )
        
        # Final status
        status_segment = (
            f"✅ Pipeline complete!\n"
            f"• Segmented: {n_objs} objects\n"
            f"• Clean images: generated for all\n"
            f"• 3D assets: {n_3d_ok}/{len(object_ids)} generated"
        )
        status_generate = f"✅ All done. 3D assets: {n_3d_ok}/{len(object_ids)}."
        yield _build_output(
            status_segment,
            overlay_path,
            status_generate,
            generated_paths=generated_paths,
            obj_statuses=obj_statuses,
            model_3d_paths=model_3d_paths,
            viewer_visibilities=viewer_visibilities,
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        err = f"❌ Pipeline error: {str(e)}\n{traceback.format_exc()}"
        yield (
            err, None, "", gr.update(visible=False),
            *([gr.update(visible=False)] * NUM_ROWS),
            *([None] * NUM_ROWS),
            "",
            *([None] * NUM_ROWS),
            *([None] * NUM_ROWS),
            *([None] * NUM_ROWS),
            *([gr.update(visible=False)] * NUM_ROWS),
        )


def create_object_row(obj_id: str, label: str, idx: int):
    """Create UI components for a single object row."""
    with gr.Group():
        gr.Markdown(f"### 🎯 Object {idx + 1}: {label} (ID: `{obj_id}`)")
        
        with gr.Row():
            # Left: Masked image
            with gr.Column(scale=1):
                gr.Markdown("**📷 Masked Original**")
                masked_img = gr.Image(label="", height=250, interactive=False)
            
            # Right: Generated/Edited image
            with gr.Column(scale=1):
                gr.Markdown("**✨ Generated/Edited**")
                generated_img = gr.Image(label="", height=250, interactive=False)
        
        # Edit and 3D generation controls
        with gr.Row():
            edit_prompt = gr.Textbox(
                label="💬 Edit Prompt (optional)",
                placeholder="e.g., 'make it blue', 'add racing stripes'",
                scale=3
            )
            edit_btn = gr.Button("🎨 Edit Style", scale=1, variant="secondary")
            gen3d_btn = gr.Button("🎲 Generate 3D", scale=1, variant="primary")
        
        # Status for this object
        obj_status = gr.Textbox(label="Status", lines=1, interactive=False)
        
        # 3D viewer (hidden by default)
        with gr.Column(visible=False) as viewer_col:
            gr.Markdown("**🎲 3D Asset**")
            model_3d = gr.Model3D(label="", height=400)
        
        # Wire up edit button
        edit_btn.click(
            fn=lambda prompt: edit_single_object(obj_id, prompt),
            inputs=[edit_prompt],
            outputs=[obj_status, generated_img]
        )
        
        # Wire up 3D button
        gen3d_btn.click(
            fn=lambda: generate_3d_single_object(obj_id),
            inputs=[],
            outputs=[obj_status, model_3d]
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[viewer_col]
        )
        
    return {
        'masked_img': masked_img,
        'generated_img': generated_img,
        'edit_prompt': edit_prompt,
        'obj_status': obj_status,
        'model_3d': model_3d,
        'viewer_col': viewer_col
    }


def create_ui():
    """Create row-based Gradio UI."""
    
    with gr.Blocks(title="🤖 Agentic 2D→3D System", theme=gr.themes.Soft(), css="""
        .object-row { border: 2px solid #444; border-radius: 8px; padding: 16px; margin: 16px 0; }
    """) as demo:
        gr.Markdown("""
        # 🤖 Agentic AI System for 2D→3D Asset Generation
        
        **Simple Workflow:** Upload → Segment → Generate Images → Edit (optional) → Create 3D Assets
        """)
        
        # Hidden for URL job_id
        url_job_id = gr.Textbox(visible=False, elem_id="url_job_id")
        
        # ============ Step 1: Upload & Segment ============
        gr.Markdown("## 📤 Step 1: Upload & Segment Objects")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="numpy", height=400)
                with gr.Row():
                    segment_button = gr.Button("🔍 Segment Objects", variant="primary", size="lg")
                    start_pipeline_button = gr.Button("🚀 Start Pipeline", variant="secondary", size="lg")
            
            with gr.Column(scale=1):
                overlay_output = gr.Image(label="Detected Objects Overlay", height=400)
                status_segment = gr.Textbox(label="Status", lines=4, interactive=False)
                shareable_url = gr.Textbox(label="Job Info", lines=2, interactive=False)
        
        # ============ Step 2: Generate Clean Images ============
        gr.Markdown("## ✨ Step 2: Generate Clean Images")
        
        with gr.Row():
            generate_button = gr.Button("✨ Generate Clean Images (All Objects)", variant="primary", size="lg", scale=3)
            status_generate = gr.Textbox(label="Generation Status", lines=2, interactive=False, scale=2)
        
        # ============ Step 3: Object Rows (Dynamic) ============
        gr.Markdown("## 🎨 Step 3: Review, Edit & Generate 3D")
        gr.Markdown("_Each object appears as a row below. Edit styles or generate 3D assets individually._")
        
        # Container for object rows (initially hidden)
        with gr.Column(visible=False) as objects_container:
            # We'll create a fixed number of rows (max 20 objects)
            object_components = []
            for i in range(20):
                with gr.Group(visible=False) as obj_group:
                    components = create_object_row(f"obj_{i}", f"Object {i+1}", i)
                    components['group'] = obj_group
                    object_components.append(components)
        
        # ============ Event Handlers ============
        
        # Segment button
        segment_button.click(
            fn=upload_and_segment,
            inputs=[image_input],
            outputs=[status_segment, overlay_output, shareable_url, objects_container]
        ).then(
            fn=lambda: setup_object_rows(),
            inputs=[],
            outputs=[comp['group'] for comp in object_components] + 
                    [comp['masked_img'] for comp in object_components]
        )
        
        # Start Pipeline button: segment → generate images → 3D for all
        pipeline_outputs = (
            [status_segment, overlay_output, shareable_url, objects_container]
            + [comp['group'] for comp in object_components]
            + [comp['masked_img'] for comp in object_components]
            + [status_generate]
            + [comp['generated_img'] for comp in object_components]
            + [comp['obj_status'] for comp in object_components]
            + [comp['model_3d'] for comp in object_components]
            + [comp['viewer_col'] for comp in object_components]
        )
        start_pipeline_button.click(
            fn=run_full_pipeline,
            inputs=[image_input],
            outputs=pipeline_outputs
        )
        
        # Generate button - properly chain the updates
        def generate_and_update_images():
            """Generate images and return updates for UI."""
            status, updates = generate_all_clean_images()
            
            # Convert updates to image paths for each component
            image_updates = []
            for i in range(20):
                if i < len(updates) and updates[i].get('generated_path'):
                    image_updates.append(updates[i]['generated_path'])
                else:
                    image_updates.append(None)
            
            return [status] + image_updates
        
        generate_button.click(
            fn=generate_and_update_images,
            inputs=[],
            outputs=[status_generate] + [comp['generated_img'] for comp in object_components]
        )
        
        # ============ Helper Functions for Dynamic Updates ============
        
        def setup_object_rows():
            """Show/hide rows based on number of objects."""
            global current_job
            
            if current_job is None:
                return [gr.update(visible=False)] * 20 + [None] * 20
            
            updates_visibility = []
            updates_images = []
            
            for i in range(20):
                if i < len(current_job.objects):
                    obj = current_job.objects[i]
                    # Show row
                    updates_visibility.append(gr.update(visible=True))
                    # Set masked image
                    if obj.masked_original_path and os.path.exists(obj.masked_original_path):
                        updates_images.append(obj.masked_original_path)
                    else:
                        updates_images.append(None)
                else:
                    # Hide row
                    updates_visibility.append(gr.update(visible=False))
                    updates_images.append(None)
            
            return updates_visibility + updates_images
        
        # ============ Load Previous Job ============
        gr.Markdown("---")
        gr.Markdown("### 📂 Load Previous Job")
        gr.Markdown("💡 **Tip:** Use URL: `http://0.0.0.0:7860/?job_id=YOUR_JOB_ID`")
        
        with gr.Row():
            job_id_input = gr.Textbox(label="Job ID", placeholder="Enter job ID to load", scale=3)
            load_button = gr.Button("📂 Load Job", scale=1)
        
        load_status = gr.Textbox(label="Load Status", lines=2, interactive=False)
    
    return demo


def main():
    """Launch Gradio app."""
    logger.info("Starting Row-Based Agentic 2D→3D Gradio App")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
