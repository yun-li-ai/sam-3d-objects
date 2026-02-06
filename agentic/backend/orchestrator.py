#!/usr/bin/env python3
"""
Main orchestrator for the agentic 2D→3D system.
"""

from typing import List, Optional, Tuple
from PIL import Image
from loguru import logger
from pathlib import Path
import time
import os

from backend.models import Job, JobStatus, SegmentedObject, GenerationQueue
from backend.services.gemini_service import GeminiService
from backend.services.sam3d_service import SAM3DService
from backend.services.storage_service import StorageService
from backend.agents.gemini_segmentation_agent import GeminiSegmentationAgent
from backend.agents.image_generation_agent import ImageGenerationAgent
from backend.agents.generation_3d_agent import Generation3DAgent


class AgenticOrchestrator:
    """
    Main orchestrator coordinating all agents and services.
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        sam3d_config: str = "../checkpoints/hf/pipeline.yaml",
        storage_base_dir: str = "data/jobs"  # Now under agentic/
    ):
        """
        Initialize orchestrator with all services and agents.
        
        Args:
            gemini_api_key: Gemini API key (or use env var)
            sam3d_config: Path to SAM-3D config
            storage_base_dir: Base directory for storage
        """
        logger.info("🤖 Initializing Agentic Orchestrator")
        
        # Initialize services.
        self.gemini_service = GeminiService(api_key=gemini_api_key)
        self.sam3d_service = SAM3DService(config_path=sam3d_config)
        self.storage_service = StorageService(base_dir=storage_base_dir)
        
        # Initialize agents.
        self.segmentation_agent = GeminiSegmentationAgent(
            self.gemini_service, self.storage_service
        )
        self.image_agent = ImageGenerationAgent(
            self.gemini_service, self.storage_service
        )
        self.generation_3d_agent = Generation3DAgent(
            self.sam3d_service, self.storage_service
        )
        
        # Generation queue.
        self.generation_queue = GenerationQueue()
        
        logger.info("✅ Orchestrator initialized")
    
    def create_job_from_image(self, image: Image.Image, job_id: Optional[str] = None) -> Job:
        """
        Create a new job from uploaded image.
        
        Args:
            image: Uploaded image
            job_id: Optional job ID
            
        Returns:
            Created Job object
        """
        logger.info("Creating new job from uploaded image")
        
        # Create job directory.
        if job_id is None:
            job_id = self.storage_service.create_job_directory()
        
        # Save both original and resized images.
        original_path, resized_path = self.storage_service.save_original_image(job_id, image)
        
        # Create job.
        job = Job(
            job_id=job_id,
            original_image_path=original_path,
            resized_image_path=resized_path,
            status=JobStatus.UPLOADED.value
        )
        
        # Save metadata.
        job.save(Path(self.storage_service.base_dir))
        
        logger.info(f"✅ Job created: {job_id}")
        return job
    
    def segment_image(self, job: Job) -> Tuple[Job, Image.Image]:
        """
        Segment objects in the job's image.
        
        Args:
            job: Job to process
            
        Returns:
            Tuple of (updated Job, overlay image)
        """
        logger.info(f"Segmenting image for job {job.job_id}")
        job.status = JobStatus.SEGMENTING.value
        
        try:
            # Load resized image for processing.
            resized_image = Image.open(job.resized_image_path)
            
            # Segment.
            objects, overlay_image = self.segmentation_agent.segment_and_process(
                job.job_id, resized_image
            )
            
            # Update job.
            job.objects = objects
            job.overlay_mask_path = str(
                Path(self.storage_service.base_dir) / job.job_id / "overlay_masks.png"
            )
            job.status = JobStatus.SEGMENTED.value
            
            # Save.
            job.save(Path(self.storage_service.base_dir))
            
            logger.info(f"✅ Segmentation complete: {len(objects)} objects")
            return job, overlay_image
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            job.status = JobStatus.ERROR.value
            job.error_message = str(e)
            job.save(Path(self.storage_service.base_dir))
            raise
    
    def generate_clean_images(self, job: Job, object_ids: Optional[List[str]] = None, progress_callback=None):
        """
        Generate clean images for objects.
        
        Args:
            job: Job to process
            object_ids: Optional list of object IDs (if None, process all)
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Generating clean images for job {job.job_id}")
        
        # Load resized image for processing.
        resized_image = Image.open(job.resized_image_path)
        
        # Filter objects.
        if object_ids:
            objects = [obj for obj in job.objects if obj.object_id in object_ids]
        else:
            objects = job.objects
        
        # Generate images.
        self.image_agent.batch_generate_clean_images(
            job.job_id, objects, resized_image, progress_callback=progress_callback
        )
        
        # Save job.
        job.save(Path(self.storage_service.base_dir))
        
        logger.info("✅ Clean image generation complete")
    
    def edit_object_image(self, job: Job, object_id: str, edit_prompt: str) -> str:
        """
        Edit an object's image.
        
        Args:
            job: Job
            object_id: Object ID
            edit_prompt: Edit instruction
            
        Returns:
            Path to edited image
        """
        logger.info(f"Editing object {object_id} with prompt: {edit_prompt}")
        
        obj = job.get_object(object_id)
        if not obj:
            raise ValueError(f"Object {object_id} not found in job {job.job_id}")
        
        # Edit image.
        edited_path = self.image_agent.edit_image(job.job_id, obj, edit_prompt)
        
        # Save job.
        job.save(Path(self.storage_service.base_dir))
        
        logger.info(f"✅ Image edited: {edited_path}")
        return edited_path
    
    def submit_3d_generation(self, job: Job, object_ids: List[str]):
        """
        Submit 3D generation jobs to queue.
        
        Args:
            job: Job
            object_ids: List of object IDs to generate
        """
        logger.info(f"Submitting {len(object_ids)} objects to 3D generation queue")
        
        # Add to queue.
        self.generation_queue.add_batch(job.job_id, object_ids)
        
        logger.info(f"✅ Added to queue. Queue size: {self.generation_queue.size()}")
    
    def process_3d_queue(self, max_iterations: int = 1) -> List[Tuple[str, str, bool]]:
        """
        Process items in the 3D generation queue.
        
        Args:
            max_iterations: Maximum number of items to process
            
        Returns:
            List of (job_id, object_id, success) tuples
        """
        results = []
        
        for _ in range(max_iterations):
            if self.generation_queue.is_empty():
                break
            
            # Get next item.
            item = self.generation_queue.get_next()
            if item is None:
                break
            
            job_id, object_id = item
            logger.info(f"Processing 3D generation: {job_id}/{object_id}")
            
            try:
                # Load job.
                job = Job.load(Path(self.storage_service.base_dir), job_id)
                if not job:
                    logger.error(f"Job {job_id} not found")
                    self.generation_queue.mark_error()
                    results.append((job_id, object_id, False))
                    continue
                
                # Get object.
                obj = job.get_object(object_id)
                if not obj:
                    logger.error(f"Object {object_id} not found")
                    self.generation_queue.mark_error()
                    results.append((job_id, object_id, False))
                    continue
                
                # Generate 3D asset.
                asset = self.generation_3d_agent.generate_3d_asset(job_id, obj)
                
                # Save job.
                job.save(Path(self.storage_service.base_dir))
                
                # Mark complete.
                self.generation_queue.mark_complete(job_id, object_id)
                results.append((job_id, object_id, True))
                
                logger.info(f"✅ 3D generation complete for {object_id}")
                
            except Exception as e:
                logger.error(f"3D generation failed: {e}")
                self.generation_queue.mark_error()
                results.append((job_id, object_id, False))
        
        return results
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Load job from storage.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job or None
        """
        return Job.load(Path(self.storage_service.base_dir), job_id)
    
    def list_jobs(self) -> List[str]:
        """
        List all job IDs.
        
        Returns:
            List of job IDs
        """
        return self.storage_service.list_jobs()
    
    def get_queue_status(self) -> dict:
        """
        Get queue status information.
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            'pending': len(self.generation_queue.pending),
            'processing': 1 if self.generation_queue.processing else 0,
            'completed': len(self.generation_queue.completed),
            'current': self.generation_queue.processing
        }
