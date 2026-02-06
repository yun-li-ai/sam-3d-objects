#!/usr/bin/env python3
"""
Gemini API service for segmentation and image generation.
Matches implementation from @agentic_image2model/gemini_image_segmentation.ipynb
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
from loguru import logger
import io
import base64
import json
from google import genai
from google.genai import types

# Import notebook's mask parsing utility
from backend.utils.mask_utils import parse_masks_from_gemini


class GeminiService:
    """Wrapper for Google Gemini API operations."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        segmentation_model: str = "gemini-2.5-pro", # "gemini-3-pro-preview", # "gemini-3-flash-preview", #"gemini-3-pro-preview", #"gemini-2.5-pro",
        image_model: str =  "gemini-3-pro-image-preview" # "gemini-2.5-flash-image"
    ):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            segmentation_model: Model for segmentation (text/structured output)
            image_model: Model for image generation
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        
        self.segmentation_model = segmentation_model
        self.image_model = image_model
        
        # Initialize client immediately (not lazy)
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("✅ Gemini client initialized")
        except ImportError:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        
        logger.info(f"Initializing Gemini service:")
        logger.info(f"  - Segmentation model: {segmentation_model}")
        logger.info(f"  - Image model: {image_model}")
    
    def segment_objects(self, image: Image.Image) -> List[Dict]:
        """
        Segment objects in image using Gemini.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            List of segmentation masks with labels, boxes, and masks
        """
        # Resize image to max 1024x1024 (matches notebook line 566-570)
        img_to_send = image.copy()
        img_to_send.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {image.size} to {img_to_send.size} for Gemini API")
        
        logger.info(f"Calling Gemini API for object segmentation (model: {self.segmentation_model})...")
        
        # Matches notebook prompt (simplified version that works faster)
        prompt = """You are part of a project to create 3D asset of objects in a 2D image. 
Your task is to detect objects of interest and return a segmentation mask for each object in the image. 
Find noticeable objects that might be useful for generating 3D assets in the image, ignore background, walls, floors, and very small objects. 
Return at most 8 objects, ordered by the visual importance of the object in the image (most important first).

Output a JSON list of segmentation masks where each entry contains 
- the descriptive text label in the key "label", 
- the 2D bounding box in the key "box_2d", 
- and the segmentation mask in key "mask".
"""
        

        try:
            # Matches notebook at line 584-592:
            # response = client.models.generate_content(
            #     model=MODEL_ID,
            #     contents=[prompt, img],
            #     config=types.GenerateContentConfig(
            #         temperature=0.5,
            #         safety_settings=safety_settings,
            #     )
            # )
            # return response.text
            
            response = self.client.models.generate_content(
                model=self.segmentation_model,
                contents=[prompt, image],  # Image already resized
                config=types.GenerateContentConfig(
                    temperature=0.5
                )
            )
            print("response.text = ", response.text)
            
            # Notebook uses response.text for text output
            masks_data = json.loads(self.parse_json(response.text))
            print("masks_data = ", masks_data)
            logger.info(f"✅ Gemini segmented {len(masks_data)} objects")
            
            return masks_data
            
        except Exception as e:
            logger.error(f"Gemini segmentation failed: {e}")
            raise
    

    def parse_json(self, text: str) -> str:
        return text.strip().removeprefix("```json").removesuffix("```")
    
    def generate_clean_image(self, image: Image.Image, mask_info: Dict) -> Image.Image:
        """
        Generate a clean image of the object without background.
        
        Args:
            image: Cropped object region (already at reasonable size)
            mask_info: Mask information including label
            
        Returns:
            Generated clean image
        """
        logger.info(f"Generating clean image for object: {mask_info.get('label', 'unknown')} (model: {self.image_model})")
        logger.info(f"Image size: {image.size}")
        
        # Matches notebook prompt at line 732-736
        prompt = f"""You are given an image and a segmentation masks of a target object in the image. 
You need to generate a new image for that object, there should be only the target object in the image, no other objects.
The target object should be shown from an angle that best captures its shape and visual features. 
The object in the new image should be as similar to the original image as possible, keep the original object style (color, texture, etc.).

If the object is a famous iconic object (such as statue of liberty, golden gate bridge, etc.), you can use a real image of that object if you can find one.
IMPORTANT: Use a background color that is easily distinguishable from the object, so background remove tool like rembg can easily remove the background.
The mask is as follow:
{str(mask_info)}"""
        
        try:
            # Matches notebook at line 740-743:
            # response = client.models.generate_content(
            #     model=IMAGE_MODEL_ID,
            #     contents=[prompt, img],
            # )
            
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=[prompt, image]  # Image already cropped/resized
            )
            
            # Matches notebook at line 745-751:
            # for part in response.parts:
            #     if part.text is not None:
            #         print(part.text)
            #     elif part.inline_data is not None:
            #         print("inline data not none")
            #         image = part.as_image()
            #         img_path = f"{IMG_DIR}/{job_id}/{idx}.png"
            
            # Check if response has parts
            if not hasattr(response, 'parts') or response.parts is None:
                logger.error(f"Gemini response has no parts. Response: {response}")
                raise ValueError("Gemini returned empty response (no parts)")
            
            for part in response.parts:
                if part.inline_data is not None:
                    # Use raw bytes and convert to PIL so rembg/save get a real PIL Image.
                    raw = getattr(part.inline_data, "data", None) or getattr(part.inline_data, "bytes", None)
                    if raw is None and hasattr(part.inline_data, "image"):
                        # Some SDKs expose image; try to get bytes from it.
                        img_obj = part.inline_data.image
                        if hasattr(img_obj, "image_bytes"):
                            raw = img_obj.image_bytes
                    if raw is not None:
                        generated_image = Image.open(io.BytesIO(bytes(raw))).convert("RGBA")
                        logger.info(f"✅ Generated clean image")
                        return generated_image
                    # Fallback: as_image() may return PIL in some SDK versions.
                    generated_image = part.as_image()
                    if isinstance(generated_image, Image.Image):
                        return generated_image
                    # If it's google.genai.types.Image, try to get bytes and convert.
                    if hasattr(generated_image, "image_bytes"):
                        generated_image = Image.open(io.BytesIO(generated_image.image_bytes)).convert("RGBA")
                        return generated_image
                    raise ValueError("Could not get image bytes from Gemini response. Try using force_return_bytes=True.")
            
            logger.warning("No image generated in response, returning original")
            return image
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    def edit_image(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Edit image based on text prompt.
        
        Args:
            image: Image to edit (already at reasonable size)
            prompt: Edit instruction (e.g., "make it red")
            
        Returns:
            Edited image
        """
        logger.info(f"Editing image with prompt: {prompt} (model: {self.image_model})")
        logger.info(f"Image size: {image.size}")
        
        full_prompt = f"""Edit the object in this image according to the following instruction:
{prompt}

Keep the object in the same position and angle when possible. Only modify what is requested.
IMPORTANT: Maintain the TRANSPARENT BACKGROUND (alpha = 0) if the image has one, or make the background transparent if it isn't already."""
        
        try:
            # Matches notebook edit_image at line 781-784:
            # response = client.models.generate_content(
            #     model=IMAGE_MODEL_ID,
            #     contents=[prompt, img],
            # )
            
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=[full_prompt, image]  # Image already resized
            )
            
            # Matches notebook at line 786-792:
            # for part in response.parts:
            #     if part.text is not None:
            #         print(part.text)
            #     elif part.inline_data is not None:
            #         print("inline data not none")
            #         image = part.as_image()
            #         img_path = f"{IMG_DIR}/{job_id}/{idx}_edited.png"
            
            # Check if response has parts
            if not hasattr(response, 'parts') or response.parts is None:
                logger.error(f"Gemini response has no parts. Response: {response}")
                raise ValueError("Gemini returned empty response (no parts)")
            
            for part in response.parts:
                if part.inline_data is not None:
                    # Convert to PIL Image so storage/rembg get a real PIL Image (same as generate_clean_image).
                    raw = getattr(part.inline_data, "data", None) or getattr(part.inline_data, "bytes", None)
                    if raw is None and hasattr(part.inline_data, "image"):
                        img_obj = part.inline_data.image
                        if hasattr(img_obj, "image_bytes"):
                            raw = img_obj.image_bytes
                    if raw is not None:
                        edited_image = Image.open(io.BytesIO(bytes(raw))).convert("RGBA")
                        logger.info(f"✅ Image edited successfully")
                        return edited_image
                    edited_image = part.as_image()
                    if isinstance(edited_image, Image.Image):
                        logger.info(f"✅ Image edited successfully")
                        return edited_image
                    if hasattr(edited_image, "image_bytes"):
                        edited_image = Image.open(io.BytesIO(edited_image.image_bytes)).convert("RGBA")
                        logger.info(f"✅ Image edited successfully")
                        return edited_image
                    raise ValueError("Could not get image bytes from Gemini edit response. Try using force_return_bytes=True.")
            
            logger.warning("No image generated in response, returning original")
            return image
            
        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise
    
    def parse_segmentation_masks(self, masks_data: List[Dict], img_width: int, img_height: int) -> List[Dict]:
        """
        Parse Gemini segmentation output using notebook's exact implementation.
        
        Args:
            masks_data: List of mask dictionaries from Gemini
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            List of parsed masks with numpy arrays
        """
        logger.info(f"Parsing {len(masks_data)} segmentation masks using notebook utils")
        
        # Use notebook's parsing function
        segmentation_data = parse_masks_from_gemini(masks_data, img_width, img_height)
        
        # Convert to our format
        parsed_masks = []
        for idx, (mask, label) in enumerate(segmentation_data):
            # Get the original mask data from Gemini for this item
            original_mask_dict = masks_data[idx] if idx < len(masks_data) else {}
            
            # Compute bbox from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if rows.any() and cols.any():
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                # Convert numpy int64 to Python int for JSON serialization
                bbox = [int(x_indices[0]), int(y_indices[0]), int(x_indices[-1]), int(y_indices[-1])]
                logger.info(f"Mask {idx} ({label}): bbox={bbox}, mask_shape={mask.shape}, non-zero={np.count_nonzero(mask)}")
            else:
                # Empty mask, skip
                logger.warning(f"Mask {idx} ({label}) is empty, skipping")
                continue
            
            parsed_masks.append({
                'label': label,
                'bbox': bbox,
                'mask': mask,
                'confidence': 1.0,
                'original_mask_data': original_mask_dict  # Keep original for passing to image generation
            })
        
        logger.info(f"✅ Successfully converted {len(parsed_masks)} masks")
        return parsed_masks
