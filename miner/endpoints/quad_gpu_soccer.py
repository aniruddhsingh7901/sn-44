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
from miner.utils.quad_gpu_model_manager import QuadGPUModelManager
from miner.utils.optimized_video_processor import OptimizedVideoProcessor
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global quad GPU model manager instance
quad_gpu_model_manager = None

# Maximum concurrent processing for 4x4090 setup
MAX_QUAD_CONCURRENT_CHALLENGES = 6
quad_challenge_semaphore = asyncio.Semaphore(MAX_QUAD_CONCURRENT_CHALLENGES)

def get_quad_gpu_model_manager(config: Config = Depends(get_config)) -> QuadGPUModelManager:
    global quad_gpu_model_manager
    if quad_gpu_model_manager is None:
        quad_gpu_model_manager = QuadGPUModelManager(
            device=config.device,
            enable_multi_gpu=True
        )
        quad_gpu_model_manager.load_all_models_quad_gpu()
        quad_gpu_model_manager.warm_up_quad_gpus()
    return quad_gpu_model_manager

async def process_soccer_video_quad_gpu(
    video_path: str,
    model_manager: QuadGPUModelManager,
) -> Dict[str, Any]:
    """Process soccer video with 4x4090 GPU setup for top leaderboard performance."""
    start_time = time.time()
    
    try:
        # Get video info
        video_info = OptimizedVideoProcessor.get_video_info(video_path)
        logger.info(f"🚀 Quad GPU: Video {video_info.width}x{video_info.height}, {video_info.total_frames} frames")
        
        # Initialize video processor with quad GPU settings
        video_processor = OptimizedVideoProcessor(
            device=model_manager.primary_device,
            cuda_timeout=300.0,  # 5 minutes timeout
            max_batch_size=128,  # Larger batches for 4x4090s
            prefetch_frames=128,
            num_workers=8
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        # Load all frames for maximum speed
        load_start = time.time()
        all_frames, load_time = await video_processor.load_all_frames_batch(video_path)
        logger.info(f"🚀 Quad GPU: Loaded {len(all_frames)} frames in {load_time:.3f}s")
        
        # Quality preservation: Process ALL frames at FULL resolution
        logger.info(f"🎯 Quality Mode: Processing ALL {len(all_frames)} frames at 1280px resolution")
        
        # Initialize tracker
        tracker = sv.ByteTrack()
        
        tracking_data = {"frames": []}
        
        # Quad GPU parallel processing with quality preservation
        logger.info("🚀 Starting quad GPU parallel detection...")
        detection_start = time.time()
        
        # Run pitch and player detection in parallel across 4 GPUs
        pitch_task = model_manager.quad_gpu_batch_inference(
            "pitch", 
            all_frames, 
            imgsz=1280  # Full resolution
        )
        
        player_task = model_manager.quad_gpu_batch_inference(
            "player", 
            all_frames, 
            imgsz=1280  # Full resolution
        )
        
        # Wait for both to complete
        pitch_results, player_results = await asyncio.gather(pitch_task, player_task)
        
        detection_time = time.time() - detection_start
        logger.info(f"🚀 Quad GPU: Parallel detection completed in {detection_time:.3f}s")
        
        # Process results and apply tracking
        logger.info("🔧 Processing results and applying tracking...")
        tracking_start = time.time()
        
        for frame_number, (pitch_result, player_result) in enumerate(zip(pitch_results, player_results)):
            # Extract keypoints from pitch detection
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # Extract detections from player detection
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            # Create frame data
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
            
            # Log progress every 50 frames
            if frame_number % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"🚀 Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        tracking_time = time.time() - tracking_start
        logger.info(f"🔧 Tracking processing completed in {tracking_time:.3f}s")
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        
        # Quad GPU performance metrics
        logger.info("🏆 === QUAD GPU PERFORMANCE METRICS ===")
        logger.info(f"🎯 Total frames: {total_frames}")
        logger.info(f"🎯 Quality mode: Full resolution (1280px)")
        logger.info(f"🎯 Frame coverage: 100% (no skipping)")
        logger.info(f"⚡ Frame loading: {load_time:.3f}s")
        logger.info(f"🚀 Quad GPU detection: {detection_time:.3f}s")
        logger.info(f"🔧 Tracking processing: {tracking_time:.3f}s")
        logger.info(f"🏁 Total processing: {processing_time:.3f}s")
        logger.info(f"📈 Average FPS: {fps:.2f}")
        logger.info(f"🔥 GPU count: {model_manager.gpu_count}")
        logger.info(f"💾 Memory usage: {model_manager.get_quad_gpu_memory_usage()}")
        logger.info("🏆 === END QUAD GPU METRICS ===")
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"🚨 Quad GPU processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quad GPU video processing error: {str(e)}")

async def process_challenge_quad_gpu(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: QuadGPUModelManager = Depends(get_quad_gpu_model_manager),
):
    """Process challenge with 4x4090 GPU setup for top leaderboard performance."""
    logger.info("🚀 Quad GPU: Attempting to acquire challenge semaphore...")
    
    async with quad_challenge_semaphore:
        logger.info(f"🚀 Quad GPU: Semaphore acquired ({quad_challenge_semaphore._value} slots remaining)")
        
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"🚀 Quad GPU: Processing challenge {challenge_id}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            # Ultra-fast video download
            download_start = time.time()
            try:
                video_path = await asyncio.wait_for(
                    download_video(video_url), 
                    timeout=10.0  # 10 second download timeout
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Video download timeout")
            
            download_time = time.time() - download_start
            logger.info(f"⚡ Quad GPU: Video download completed in {download_time:.3f}s")
            
            try:
                # Process video with quad GPU setup
                tracking_data = await process_soccer_video_quad_gpu(
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
                
                total_time = response["total_time"]
                logger.info(
                    f"🏁 Quad GPU: Challenge {challenge_id} completed - "
                    f"Processing: {tracking_data['processing_time']:.3f}s, "
                    f"Download: {download_time:.3f}s, "
                    f"Total: {total_time:.3f}s"
                )
                
                # Log performance vs leaderboard targets
                if total_time <= 0.8:
                    logger.info(f"🏆 QUAD GPU: RANK #1 PERFORMANCE! ({total_time:.3f}s)")
                elif total_time <= 1.0:
                    logger.info(f"🥈 QUAD GPU: TOP 3 PERFORMANCE! ({total_time:.3f}s)")
                elif total_time <= 1.2:
                    logger.info(f"🥉 QUAD GPU: TOP 5 PERFORMANCE! ({total_time:.3f}s)")
                elif total_time <= 1.4:
                    logger.info(f"📈 QUAD GPU: TOP 10 PERFORMANCE! ({total_time:.3f}s)")
                else:
                    logger.warning(f"⚠️ QUAD GPU: Need more optimization ({total_time:.3f}s)")
                
                return response
                
            finally:
                # Ultra-fast cleanup
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"🚨 Quad GPU challenge processing error: {str(e)}")
            logger.exception("Quad GPU error traceback:")
            raise HTTPException(status_code=500, detail=f"Quad GPU challenge error: {str(e)}")
        finally:
            logger.info("🚀 Quad GPU: Releasing challenge semaphore...")

# Create quad GPU router
router = APIRouter()

# Quad GPU optimized endpoint
router.add_api_route(
    "/challenge",
    process_challenge_quad_gpu,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)

# Health check with quad GPU performance info
@router.get("/quad/health")
async def quad_gpu_health_check(
    model_manager: QuadGPUModelManager = Depends(get_quad_gpu_model_manager)
):
    """Quad GPU health check with performance metrics."""
    stats = model_manager.get_performance_stats()
    
    return {
        "status": "🚀 QUAD GPU READY",
        "mode": "quad_gpu_optimized",
        "gpu_count": stats["gpu_count"],
        "target_performance": stats["expected_performance"],
        "quality_preservation": stats["quality_preservation"],
        "resolution": f"{stats['target_resolution']}px",
        "batch_size": stats["batch_size"],
        "memory_usage": stats["memory_usage"],
        "optimizations": stats["optimizations"],
        "leaderboard_targets": {
            "rank_1": "0.8s",
            "top_3": "1.0s", 
            "top_5": "1.2s",
            "our_target": "0.8-1.0s"
        }
    }

# Performance benchmark endpoint
@router.post("/quad/benchmark")
async def quad_gpu_benchmark(
    model_manager: QuadGPUModelManager = Depends(get_quad_gpu_model_manager)
):
    """Run quad GPU performance benchmark."""
    try:
        # Download test video
        test_url = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"
        video_path = await download_video(test_url)
        
        try:
            # Run quad GPU processing
            start_time = time.time()
            result = await process_soccer_video_quad_gpu(video_path, model_manager)
            total_time = time.time() - start_time
            
            # Determine leaderboard position
            if total_time <= 0.8:
                position = "🏆 RANK #1 POTENTIAL"
                status = "CHAMPION"
            elif total_time <= 1.0:
                position = "🥈 TOP 3 POTENTIAL"
                status = "ELITE"
            elif total_time <= 1.2:
                position = "🥉 TOP 5 POTENTIAL"
                status = "COMPETITIVE"
            elif total_time <= 1.4:
                position = "📈 TOP 10 POTENTIAL"
                status = "GOOD"
            else:
                position = "📉 NEEDS OPTIMIZATION"
                status = "IMPROVEMENT_NEEDED"
            
            return {
                "benchmark_result": "success",
                "processing_time": result["processing_time"],
                "total_time": total_time,
                "frames_processed": len(result["frames"]),
                "quality_mode": "full_resolution_all_frames",
                "leaderboard_position": position,
                "status": status,
                "target_met": total_time <= 1.0,
                "gpu_count": model_manager.gpu_count,
                "memory_usage": model_manager.get_quad_gpu_memory_usage()
            }
            
        finally:
            try:
                os.unlink(video_path)
            except:
                pass
                
    except Exception as e:
        return {
            "benchmark_result": "error",
            "error": str(e)
        }

# Performance statistics endpoint
@router.get("/quad/stats")
async def quad_gpu_stats(
    model_manager: QuadGPUModelManager = Depends(get_quad_gpu_model_manager)
):
    """Get detailed quad GPU performance statistics."""
    return model_manager.get_performance_stats()
