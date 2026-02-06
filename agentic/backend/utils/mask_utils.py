#!/usr/bin/env python3
"""
Mask parsing utilities - matches notebook implementation exactly.
From: @agentic_image2model/gemini_image_segmentation.ipynb lines 160-242
"""

import base64
import cv2
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple


def parse_masks_from_gemini(
    masks_data: List[Dict],
    img_width: int,
    img_height: int
) -> List[Tuple[np.ndarray, str]]:
    """
    Parse Gemini segmentation output into usable masks.
    
    Exactly matches notebook implementation at lines 160-242.
    
    Args:
        masks_data: List of mask dictionaries from Gemini
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of (mask, label) tuples
    """
    segmentation_data = []
    default_label = "unknown"
    
    for item_idx, item in enumerate(masks_data):
        if not isinstance(item, dict) or "box_2d" not in item or "mask" not in item:
            logger.warning(f"Skipping invalid item structure at index {item_idx}: {item}")
            continue
        
        label = item.get("label", default_label)
        if not isinstance(label, str) or not label:
            label = default_label
        
        png_str = item["mask"]
        if not isinstance(png_str, str) or not png_str.startswith("data:image/png;base64,"):
            logger.warning(f"Skipping item {item_idx} (label: {label}) with invalid mask format.")
            continue
        
        png_str = png_str.removeprefix("data:image/png;base64,")
        
        try:
            png_bytes = base64.b64decode(png_str)
            bbox_mask = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if bbox_mask is None:
                logger.warning(f"Skipping item {item_idx} (label: {label}) because mask decoding failed.")
                continue
        except (base64.binascii.Error, ValueError, Exception) as e:
            logger.error(f"Error decoding base64 or image data for item {item_idx} (label: {label}): {e}")
            continue
        
        try:
            box = item["box_2d"]
            if not isinstance(box, list) or len(box) != 4:
                logger.warning(f"Skipping item {item_idx} (label: {label}) with invalid box_2d format: {box}")
                continue
            
            # CRITICAL: Notebook uses [y0, x0, y1, x1] format (line 200)
            y0_norm, x0_norm, y1_norm, x1_norm = map(float, box)
            abs_y0 = max(0, min(int(y0_norm / 1000.0 * img_height), img_height - 1))
            abs_x0 = max(0, min(int(x0_norm / 1000.0 * img_width), img_width - 1))
            abs_y1 = max(0, min(int(y1_norm / 1000.0 * img_height), img_height))
            abs_x1 = max(0, min(int(x1_norm / 1000.0 * img_width), img_width))
            
            bbox_height = abs_y1 - abs_y0
            bbox_width = abs_x1 - abs_x0
            
            if bbox_height <= 0 or bbox_width <= 0:
                logger.warning(f"Skipping item {item_idx} (label: {label}) with invalid bbox dims: {box} -> ({bbox_width}x{bbox_height})")
                continue
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping item {item_idx} (label: {label}) due to error processing box_2d: {e}")
            continue
        
        try:
            if bbox_mask.shape[0] > 0 and bbox_mask.shape[1] > 0:
                resized_bbox_mask = cv2.resize(
                    bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR
                )
            else:
                logger.warning(f"Skipping item {item_idx} (label: {label}) due to empty decoded mask before resize.")
                continue
        except cv2.error as e:
            logger.error(f"Error resizing mask for item {item_idx} (label: {label}): {e}")
            continue
        
        # Create full-sized mask
        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        try:
            full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask
        except ValueError as e:
            logger.error(f"Error placing mask for item {item_idx} (label: {label}): {e}. Shape mismatch: slice=({bbox_height},{bbox_width}), resized={resized_bbox_mask.shape}. Attempting correction.")
            try:
                resized_bbox_mask_corrected = cv2.resize(bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR)
                full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask_corrected
                logger.info("  -> Corrected placement.")
            except Exception as inner_e:
                logger.error(f"  -> Failed to correct placement: {inner_e}")
                continue
        
        segmentation_data.append((full_mask, label))
        logger.info(f"✅ Parsed mask {item_idx}: {label}")
    
    logger.info(f"✅ Successfully parsed {len(segmentation_data)}/{len(masks_data)} masks")
    return segmentation_data
