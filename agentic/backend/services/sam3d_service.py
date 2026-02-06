#!/usr/bin/env python3
"""
SAM-3D service for 3D asset generation.
"""

import sys
import os
from pathlib import Path
from typing import Union, Optional, Dict
import numpy as np
from PIL import Image
import torch
from loguru import logger
import gc

# Add notebook to path for existing inference code.
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "notebook"))

try:
    from inference import Inference
except ImportError:
    logger.warning("Could not import inference module. Make sure notebook/inference.py exists.")
    Inference = None


class SAM3DService:
    """Wrapper for SAM-3D-Objects pipeline."""
    
    def __init__(self, config_path: str = "../checkpoints/hf/pipeline.yaml", compile_model: bool = False):
        """
        Initialize SAM-3D service.
        
        Args:
            config_path: Path to pipeline configuration
            compile_model: Whether to compile model for faster inference
        """
        logger.info("Initializing SAM-3D service...")
        
        if Inference is None:
            raise ImportError("Inference module not available. Check notebook/inference.py")
        
        self.pipeline = Inference(config_path, compile=compile_model)
        logger.info("✅ SAM-3D pipeline initialized")
    
    def generate_3d(
        self,
        image: Union[np.ndarray, Image.Image, str],
        mask: Optional[Union[np.ndarray, Image.Image]] = None,
        seed: int = 42
    ) -> Dict:
        """
        Generate 3D asset from 2D image.
        
        Args:
            image: Input image (can be path, numpy array, or PIL Image)
            mask: Optional segmentation mask (None to use full image)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'gs' (Gaussian splat) and 'glb' (mesh) outputs
        """
        logger.info(f"Generating 3D asset (seed={seed})")
        
        try:
            # Run inference.
            # Note: Inference.__call__() only accepts (image, mask, seed, pointmap).
            # Stage steps are hardcoded internally.
            output = self.pipeline(
                image,
                mask,
                seed=seed
            )
            
            logger.info("✅ 3D generation complete")
            
            # Clean up GPU memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                gc.collect()
            
            return output
            
        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            # Clean up GPU memory even on failure.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                gc.collect()
            raise
    
    def save_outputs(self, output: Dict, output_dir: Path, filename_prefix: str = "model") -> Dict[str, str]:
        """
        Save 3D outputs to files.
        
        Args:
            output: Output from generate_3d
            output_dir: Directory to save outputs
            filename_prefix: Prefix for output files
            
        Returns:
            Dictionary mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save Gaussian splat (.ply).
        if 'gs' in output and output['gs'] is not None:
            ply_path = output_dir / f"{filename_prefix}.ply"
            output['gs'].save_ply(str(ply_path))
            saved_paths['ply'] = str(ply_path)
            logger.info(f"Saved Gaussian splat: {ply_path}")
        
        # Save GLB mesh.
        if 'glb' in output and output['glb'] is not None:
            glb_path = output_dir / f"{filename_prefix}.glb"
            output['glb'].export(str(glb_path))
            saved_paths['glb'] = str(glb_path)
            logger.info(f"Saved GLB mesh: {glb_path}")
        
        # Clean up output objects to free GPU memory immediately after saving
        if 'gs' in output:
            del output['gs']
        if 'glb' in output:
            del output['glb']
        if 'mesh' in output:
            del output['mesh']
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("Cleaned up GPU memory after saving outputs")
        
        return saved_paths
    
    def validate_quality(self, output: Dict) -> Dict[str, float]:
        """
        Assess quality of generated 3D asset.
        
        Args:
            output: Output from generate_3d
            
        Returns:
            Dictionary of quality scores
        """
        scores = {
            'has_gaussian_splat': 0.0,
            'has_mesh': 0.0,
            'overall': 0.0
        }
        
        # Check if outputs exist.
        if 'gs' in output and output['gs'] is not None:
            scores['has_gaussian_splat'] = 1.0
        
        if 'glb' in output and output['glb'] is not None:
            scores['has_mesh'] = 1.0
        
        # Simple overall score.
        scores['overall'] = (scores['has_gaussian_splat'] + scores['has_mesh']) / 2
        
        return scores
