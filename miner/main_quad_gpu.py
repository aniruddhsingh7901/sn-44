import os
from fastapi import FastAPI
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config
from miner.endpoints.quad_gpu_soccer import router as quad_gpu_soccer_router
from miner.endpoints.availability import router as availability_router

# Setup logging
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Score Vision QUAD GPU Miner",
    description="4x4090 GPU optimized miner for top leaderboard performance with quality preservation",
    version="4.0.0-QUAD"
)

# Add dependencies
app.dependency_overrides[Config] = get_config

# Include quad GPU routers
app.include_router(
    quad_gpu_soccer_router,
    prefix="/soccer",
    tags=["soccer"]
)
app.include_router(
    availability_router,
    tags=["availability"]
)

@app.on_event("startup")
async def startup_event():
    """Initialize quad GPU components on startup."""
    logger.info("🚀 Starting Score Vision QUAD GPU Miner...")
    logger.info("🎯 Target: RANK #1 leaderboard performance (0.8-1.0s)")
    logger.info("🎯 Quality: Full resolution + all frames preservation")
    
    # Log system information
    config = factory_config()
    logger.info(f"Device: {config.device}")
    
    # Import torch to check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            if gpu_count >= 4:
                logger.info(f"🚀 QUAD GPU MODE: {gpu_count} GPUs detected")
                logger.info("🏆 Expected performance: TOP 3 leaderboard")
            else:
                logger.warning(f"⚠️ Only {gpu_count} GPUs detected, expected 4 for optimal performance")
            
            # Set maximum performance mode for all GPUs
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.cuda.set_per_process_memory_fraction(0.95, i)
                
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                logger.info(f"🔥 GPU {i}: {props.name} ({memory_gb:.1f}GB) - QUAD MODE")
        else:
            logger.error("❌ CUDA not available - quad GPU mode requires CUDA")
    except ImportError:
        logger.error("❌ PyTorch not available")
    
    logger.info("✅ QUAD GPU miner startup complete - Ready for leaderboard domination!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down QUAD GPU miner...")
    
    # Clear GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("🧹 Cleared all GPU memory caches")
    except ImportError:
        pass
    
    logger.info("✅ QUAD GPU miner shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with quad GPU system information."""
    config = factory_config()
    
    system_info = {
        "service": "Score Vision QUAD GPU Miner",
        "version": "4.0.0-QUAD",
        "mode": "QUAD_GPU_OPTIMIZED",
        "target_performance": "0.8-1.0s (RANK #1 potential)",
        "leaderboard_target": "TOP 3",
        "quality_preservation": True,
        "device": config.device,
        "status": "🚀 QUAD GPU READY",
        "optimizations": [
            "🚀 4x GPU Parallel Processing",
            "🎯 Quality Preservation (1280px)",
            "⚡ All Frames Processing",
            "💨 FP16 Precision",
            "🎯 Torch Compile",
            "📦 Optimized Batching",
            "🔗 Model Fusion",
            "💾 Memory Pre-allocation",
            "🚀 6 Concurrent Challenges"
        ]
    }
    
    # Add GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            system_info["gpu_count"] = gpu_count
            system_info["gpu_memory"] = {}
            
            total_memory = 0
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                total_memory += memory_gb
                system_info["gpu_memory"][f"gpu_{i}"] = {
                    "name": props.name,
                    "total_memory_gb": memory_gb,
                    "quad_mode": True
                }
            
            system_info["total_gpu_memory_gb"] = total_memory
            
            if gpu_count >= 4:
                system_info["performance_estimate"] = "🏆 RANK #1 POTENTIAL"
            elif gpu_count >= 2:
                system_info["performance_estimate"] = "🥈 TOP 5 POTENTIAL"
            else:
                system_info["performance_estimate"] = "📈 COMPETITIVE"
                
    except ImportError:
        system_info["gpu_info"] = "PyTorch not available"
    
    return system_info

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "🚀 QUAD GPU HEALTHY", 
        "service": "quad_gpu_miner",
        "ready_for": "leaderboard_domination"
    }

@app.get("/leaderboard/targets")
async def leaderboard_targets():
    """Show current leaderboard targets and our capabilities."""
    return {
        "current_leaders": {
            "rank_1": {
                "time": "0.8s",
                "miner": "5DAegYk9W3u5GS2huM2z9pdx98FwqoPqWxkexJPBCj2XQ8pr",
                "score": "68.70"
            },
            "rank_2": {
                "time": "1.0s", 
                "miner": "5D4zEiSfiZqM7Cz8KzjfuS8c3oPuMdLHNU6YTn8TtJ6FTwja",
                "score": "73.70"
            },
            "rank_3": {
                "time": "1.2s",
                "miner": "5GRH5LxtqoVH4R1b7dPQstvgwx6oy4LzfwD6yLoAXzqsyKT7", 
                "score": "75.10"
            }
        },
        "our_capabilities": {
            "target_time": "0.8-1.0s",
            "expected_rank": "TOP 3",
            "quality_preservation": True,
            "competitive_advantage": "4x4090 + Quality Preservation",
            "gpu_count": 4,
            "concurrent_challenges": 6
        },
        "scoring_breakdown": {
            "quality_weight": "60%",
            "speed_weight": "30%", 
            "availability_weight": "10%",
            "our_strategy": "Maximize quality while achieving top speed"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7999)
