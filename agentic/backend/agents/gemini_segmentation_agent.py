#!/usr/bin/env python3
"""
Gemini-based segmentation agent.
"""

from typing import List, Tuple
from PIL import Image
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.gemini_service import GeminiService
from backend.services.storage_service import StorageService
from backend.models import SegmentedObject, ObjectStatus
import uuid


class GeminiSegmentationAgent:
    """Agent for segmenting images using Gemini API."""
    
    def __init__(self, gemini_service: GeminiService, storage_service: StorageService):
        """
        Initialize segmentation agent.
        
        Args:
            gemini_service: Gemini API service
            storage_service: Storage service
        """
        self.gemini = gemini_service
        self.storage = storage_service
        logger.info("Gemini Segmentation Agent initialized")
    
    def segment_and_process(
        self,
        job_id: str,
        image: Image.Image
    ) -> Tuple[List[SegmentedObject], Image.Image]:
        """
        Segment image and create object records.
        
        Args:
            job_id: Job ID
            image: Original image
            
        Returns:
            Tuple of (list of SegmentedObject, overlay visualization)
        """
        logger.info(f"Starting segmentation for job {job_id}")
        
        # Call Gemini API for segmentation.
        masks_data = self.gemini.segment_objects(image)
        
        if not masks_data:
            logger.warning("No objects detected in image")
            return [], image
        
        # Parse masks.
        img_width, img_height = image.size
        processed_masks = self.gemini.parse_segmentation_masks(
            masks_data, img_width, img_height
        )
        
        # Create overlay visualization.
        overlay_image = self._create_overlay(image, processed_masks)
        overlay_path = self.storage.save_overlay_mask(job_id, overlay_image)
        
        # Process masks in parallel.
        logger.info(f"Processing {len(processed_masks)} masks in parallel...")
        objects = self._process_masks_parallel(job_id, image, processed_masks)
        
        logger.info(f"✅ Segmentation complete: {len(objects)} objects detected")
        return objects, overlay_image
    
    def _process_masks_parallel(
        self,
        job_id: str,
        image: Image.Image,
        processed_masks: List[dict],
        max_workers: int = 4
    ) -> List[SegmentedObject]:
        """
        Process masks in parallel to create SegmentedObject instances.
        
        Args:
            job_id: Job ID
            image: Original image
            processed_masks: List of mask info dictionaries
            max_workers: Number of parallel workers
            
        Returns:
            List of SegmentedObject instances
        """
        def process_single_mask(idx: int, mask_info: dict) -> SegmentedObject:
            """Process a single mask."""
            object_id = f"obj_{idx}"
            
            # Extract bounding box.
            bbox = mask_info.get('bbox')
            if bbox is None:
                bbox = self._compute_bbox(mask_info['mask'])
            
            logger.info(f"Processing {object_id} ({mask_info['label']}): bbox={bbox}")
            
            # Save mask.
            mask_path = self.storage.save_object_mask(
                job_id, object_id, mask_info['mask']
            )
            
            # Create masked original (show object in context).
            masked_original = self._apply_mask_highlight(
                image, mask_info['mask']
            )
            masked_original_path = self.storage.save_masked_original(
                job_id, object_id, masked_original
            )
            
            # Create object record.
            import json
            obj = SegmentedObject(
                object_id=object_id,
                label=mask_info['label'],
                confidence=mask_info.get('confidence', 1.0),
                bbox=tuple(bbox),
                mask_path=mask_path,
                masked_original_path=masked_original_path,
                original_mask_data=json.dumps(mask_info.get('original_mask_data', {})),  # Store as JSON string
                status=ObjectStatus.DETECTED.value
            )
            
            logger.info(f"Processed object {object_id}: {obj.label}")
            return obj
        
        # Process masks in parallel.
        objects = [None] * len(processed_masks)  # Pre-allocate list to maintain order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks.
            future_to_idx = {
                executor.submit(process_single_mask, idx, mask_info): idx
                for idx, mask_info in enumerate(processed_masks)
            }
            
            # Collect results in order.
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    obj = future.result()
                    objects[idx] = obj
                except Exception as e:
                    logger.error(f"Error processing mask {idx}: {e}")
                    # Create a fallback object to maintain consistency.
                    import json
                    mask_info = processed_masks[idx]
                    objects[idx] = SegmentedObject(
                        object_id=f"obj_{idx}",
                        label=mask_info['label'],
                        confidence=0.0,
                        bbox=(0, 0, 0, 0),
                        mask_path="",
                        original_mask_data=json.dumps(mask_info.get('original_mask_data', {})),
                        status=ObjectStatus.ERROR.value,
                        error_message=str(e)
                    )
        
        return objects
    
    def _create_overlay(self, image: Image.Image, masks: List[dict]) -> Image.Image:
        """Create overlay visualization with all masks."""
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        
        # Use different colors for each mask.
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(masks)))
        
        for idx, mask_info in enumerate(masks):
            mask = mask_info['mask']
            label = mask_info['label']
            
            # Create colored mask overlay.
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0] = [*colors[idx][:3], 0.5]
            
            ax.imshow(colored_mask)
            
            # Add label.
            bbox = mask_info.get('bbox')
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor=colors[idx], facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x_min, y_min - 10, label,
                    color='white', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.7)
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to PIL Image (using buffer_rgba() instead of tostring_rgb()).
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        overlay_array = np.asarray(buf)
        # Convert RGBA to RGB
        overlay_array = overlay_array[:, :, :3]
        overlay_image = Image.fromarray(overlay_array)
        
        plt.close(fig)
        
        return overlay_image
    
    def _apply_mask_highlight(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply mask to highlight object in original image."""
        img_array = np.array(image)
        
        # Debug: Check mask statistics
        logger.info(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}, Max: {mask.max()}, Min: {mask.min()}")
        
        # Check if mask is empty
        if mask.sum() == 0:
            logger.warning("⚠️ Mask is empty (all zeros)! Returning original image.")
            return image
        
        # Create semi-transparent overlay.
        overlay = img_array.copy()
        
        # Dim background.
        background_mask = mask == 0
        overlay[background_mask] = (overlay[background_mask] * 0.3).astype(np.uint8)
        
        # Highlight border.
        from scipy import ndimage
        border = ndimage.binary_dilation(mask > 0) & (mask == 0)
        overlay[border] = [255, 255, 0]  # Yellow border
        
        logger.info(f"✅ Applied mask highlight - object pixels: {(mask > 0).sum()}, background pixels: {background_mask.sum()}")
        
        return Image.fromarray(overlay)
    
    def _compute_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute bounding box from mask."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return (0, 0, 0, 0)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
