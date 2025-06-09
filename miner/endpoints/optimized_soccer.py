import os
import json
import time
from typing import Optional, Dict, Any, List
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.optimized_model_manager import OptimizedModelManager
from miner.utils.optimized_video_processor import OptimizedVideoProcessor
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global optimized model manager instance
optimized_model_manager = None

# Concurrent processing semaphore (allow multiple challenges)
MAX_CONCURRENT_CHALLENGES = 2
challenge_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHALLENGES)

def get_optimized_model_manager(config: Config = Depends(get_config)) -> OptimizedModelManager:
    global optimized_model_manager
    if optimized_model_manager is None:
        optimized_model_manager = OptimizedModelManager(
            device=config.device,
            enable_multi_gpu=True
        )
        optimized_model_manager.load_all_models()
    return optimized_model_manager

async def process_soccer_video_optimized(
    video_path: str,
    model_manager: OptimizedModelManager,
) -> Dict[str, Any]:
    """Process a soccer video with optimized batch processing and multi-GPU support."""
    start_time = time.time()
    
    try:
        # Get video info for optimization
        video_info = OptimizedVideoProcessor.get_video_info(video_path)
        logger.info(f"Video info: {video_info.width}x{video_info.height}, {video_info.total_frames} frames")
        
        # Initialize optimized video processor
        video_processor = OptimizedVideoProcessor(
            device=model_manager.primary_device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0,
            max_batch_size=16,  # Optimize for 4090s
            prefetch_frames=32,
            num_workers=4
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        # Get optimal batch size based on video properties
        optimal_batch_size = video_processor.get_optimal_batch_size(video_info)
        logger.info(f"Using optimal batch size: {optimal_batch_size}")
        
        # Load all frames for maximum speed
        all_frames, load_time = await video_processor.load_all_frames_batch(video_path)
        logger.info(f"Loaded {len(all_frames)} frames in {load_time:.2f}s")
        
        # Initialize tracker
        tracker = sv.ByteTrack()
        
        tracking_data = {"frames": []}
        
        # Process frames in optimized batches
        processed_frames = 0
        
        # Batch process pitch detection
        logger.info("Starting batch pitch detection...")
        pitch_start = time.time()
        
        pitch_results = await model_manager.batch_inference_multi_gpu(
            "pitch", 
            all_frames, 
            batch_size=optimal_batch_size,
            imgsz=1280
        )
        
        pitch_time = time.time() - pitch_start
        logger.info(f"Pitch detection completed in {pitch_time:.2f}s")
        
        # Batch process player detection
        logger.info("Starting batch player detection...")
        player_start = time.time()
        
        player_results = await model_manager.batch_inference_multi_gpu(
            "player", 
            all_frames, 
            batch_size=optimal_batch_size,
            imgsz=1280
        )
        
        player_time = time.time() - player_start
        logger.info(f"Player detection completed in {player_time:.2f}s")
        
        # Process results and apply tracking
        logger.info("Processing results and applying tracking...")
        tracking_start = time.time()
        
        for frame_number, (pitch_result, player_result) in enumerate(zip(pitch_results, player_results)):
            # Extract keypoints from pitch detection
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # Extract detections from player detection
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            # Convert numpy arrays to Python native types
            frame_data = {
                "frame_number": int(frame_number),
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "objects": [
                    {
                        "id": int(tracker_id),
                        "bbox": [float(x) for x in bbox],
                        "class_id": int(class_id)
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    )
                ] if detections and detections.tracker_id is not None else []
            }
            tracking_data["frames"].append(frame_data)
            processed_frames += 1
            
            # Log progress every 100 frames
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        tracking_time = time.time() - tracking_start
        logger.info(f"Tracking processing completed in {tracking_time:.2f}s")
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        
        # Log detailed performance metrics
        logger.info("=== PERFORMANCE METRICS ===")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Frame loading: {load_time:.2f}s")
        logger.info(f"Pitch detection: {pitch_time:.2f}s")
        logger.info(f"Player detection: {player_time:.2f}s")
        logger.info(f"Tracking processing: {tracking_time:.2f}s")
        logger.info(f"Total processing: {processing_time:.2f}s")
        logger.info(f"Average FPS: {fps:.2f}")
        logger.info(f"Device: {model_manager.primary_device}")
        logger.info(f"GPU count: {model_manager.gpu_count}")
        logger.info("=== END METRICS ===")
        
        # Log GPU memory usage
        memory_usage = model_manager.get_memory_usage()
        if memory_usage:
            logger.info(f"GPU memory usage: {memory_usage}")
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_challenge_optimized(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: OptimizedModelManager = Depends(get_optimized_model_manager),
):
    """Process challenge with concurrent support and optimizations."""
    logger.info("Attempting to acquire challenge semaphore...")
    
    async with challenge_semaphore:
        logger.info(f"Challenge semaphore acquired ({challenge_semaphore._value} slots remaining)")
        
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            # Download video
            download_start = time.time()
            video_path = await download_video(video_url)
            download_time = time.time() - download_start
            logger.info(f"Video download completed in {download_time:.2f}s")
            
            try:
                # Process video with optimizations
                tracking_data = await process_soccer_video_optimized(
                    video_path,
                    model_manager
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"],
                    "download_time": download_time,
                    "total_time": tracking_data["processing_time"] + download_time
                }
                
                logger.info(
                    f"Completed challenge {challenge_id} - "
                    f"Processing: {tracking_data['processing_time']:.2f}s, "
                    f"Download: {download_time:.2f}s, "
                    f"Total: {response['total_time']:.2f}s"
                )
                
                return response
                
            finally:
                # Clean up video file
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing optimized soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing challenge semaphore...")

# Fallback to original processing for compatibility
async def process_challenge_fallback(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: OptimizedModelManager = Depends(get_optimized_model_manager),
):
    """Fallback to single-frame processing if batch processing fails."""
    from miner.endpoints.soccer import process_soccer_video
    from miner.utils.model_manager import ModelManager
    
    logger.warning("Using fallback single-frame processing")
    
    # Create traditional model manager for fallback
    fallback_manager = ModelManager(device=config.device)
    fallback_manager.load_all_models()
    
    try:
        challenge_data = await request.json()
        challenge_id = challenge_data.get("challenge_id")
        video_url = challenge_data.get("video_url")
        
        if not video_url:
            raise HTTPException(status_code=400, detail="No video URL provided")
        
        video_path = await download_video(video_url)
        
        try:
            tracking_data = await process_soccer_video(
                video_path,
                fallback_manager
            )
            
            response = {
                "challenge_id": challenge_id,
                "frames": tracking_data["frames"],
                "processing_time": tracking_data["processing_time"]
            }
            
            return response
            
        finally:
            try:
                os.unlink(video_path)
            except:
                pass
            
    except Exception as e:
        logger.error(f"Error in fallback processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fallback processing error: {str(e)}")

# Create router with optimized endpoint
router = APIRouter()

# Primary optimized endpoint
router.add_api_route(
    "/challenge",
    process_challenge_optimized,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)

# Fallback endpoint for debugging
router.add_api_route(
    "/challenge/fallback",
    process_challenge_fallback,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
