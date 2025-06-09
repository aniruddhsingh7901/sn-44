import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple, List
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

class OptimizedVideoProcessor:
    """Enhanced video processor with batch loading and async frame streaming."""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,
        mps_timeout: float = 1800.0,
        cpu_timeout: float = 10800.0,
        max_batch_size: int = 16,
        prefetch_frames: int = 32,
        num_workers: int = 4
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.prefetch_frames = prefetch_frames
        self.num_workers = num_workers
        
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:
            self.processing_timeout = cpu_timeout
            
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        logger.info(
            f"Optimized video processor initialized: device={device}, "
            f"timeout={self.processing_timeout:.1f}s, batch_size={max_batch_size}, "
            f"prefetch={prefetch_frames}, workers={num_workers}"
        )
    
    async def load_all_frames_batch(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """
        Load all video frames at once for batch processing.
        Optimized for speed over memory efficiency.
        """
        start_time = time.time()
        
        def _load_frames():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frames = []
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    
                    # Check timeout periodically
                    if len(frames) % 100 == 0:
                        elapsed = time.time() - start_time
                        if elapsed > self.processing_timeout:
                            logger.warning(f"Frame loading timeout after {elapsed:.1f}s")
                            break
                            
            finally:
                cap.release()
            
            return frames
        
        # Load frames in executor to avoid blocking
        frames = await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_frames
        )
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(frames)} frames in {load_time:.2f}s")
        
        return frames, load_time
    
    async def stream_frames_optimized(
        self,
        video_path: str,
        batch_size: Optional[int] = None
    ) -> AsyncGenerator[Tuple[List[int], List[np.ndarray]], None]:
        """
        Stream video frames in batches with prefetching for optimal performance.
        
        Args:
            video_path: Path to the video file
            batch_size: Number of frames per batch (defaults to max_batch_size)
            
        Yields:
            Tuple[List[int], List[np.ndarray]]: Frame numbers and frame data batches
        """
        if batch_size is None:
            batch_size = self.max_batch_size
            
        start_time = time.time()
        
        # Load all frames for maximum speed
        frames, load_time = await self.load_all_frames_batch(video_path)
        
        if not frames:
            logger.warning("No frames loaded from video")
            return
        
        # Yield frames in batches
        for i in range(0, len(frames), batch_size):
            elapsed_time = time.time() - start_time
            if elapsed_time > self.processing_timeout:
                logger.warning(
                    f"Video processing timeout reached after {elapsed_time:.1f}s "
                    f"on {self.device} device ({i} frames processed)"
                )
                break
            
            end_idx = min(i + batch_size, len(frames))
            batch_frames = frames[i:end_idx]
            batch_frame_numbers = list(range(i, end_idx))
            
            yield batch_frame_numbers, batch_frames
            
            # Small async yield to prevent blocking
            await asyncio.sleep(0)
    
    async def stream_frames_memory_efficient(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Memory-efficient frame streaming (fallback for very large videos).
        """
        start_time = time.time()
        
        def _frame_generator():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_count = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame_count, frame
                    frame_count += 1
            finally:
                cap.release()
        
        frame_gen = _frame_generator()
        
        for frame_number, frame in frame_gen:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.processing_timeout:
                logger.warning(
                    f"Video processing timeout reached after {elapsed_time:.1f}s "
                    f"on {self.device} device ({frame_number} frames processed)"
                )
                break
            
            yield frame_number, frame
            
            # Periodic async yield
            if frame_number % 10 == 0:
                await asyncio.sleep(0)
    
    async def preprocess_frames_batch(
        self, 
        frames: List[np.ndarray],
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Preprocess a batch of frames for optimal inference.
        
        Args:
            frames: List of frame arrays
            target_size: Optional resize target (width, height)
            
        Returns:
            List of preprocessed frames
        """
        def _preprocess_batch():
            processed_frames = []
            
            for frame in frames:
                # Resize if target size specified
                if target_size:
                    frame = cv2.resize(frame, target_size)
                
                # Convert BGR to RGB (YOLO expects RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                processed_frames.append(frame)
            
            return processed_frames
        
        # Process in executor to avoid blocking
        processed_frames = await asyncio.get_event_loop().run_in_executor(
            self.executor, _preprocess_batch
        )
        
        return processed_frames
    
    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if video is readable within timeout period.
        
        Args:
            video_path: Path to video file
            timeout: Maximum time to wait for video check
            
        Returns:
            bool: True if video is readable
        """
        try:
            async def _check_video():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return False
                ret, _ = cap.read()
                cap.release()
                return ret
            
            return await asyncio.wait_for(_check_video(), timeout)
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {str(e)}")
            return False
    
    def get_optimal_batch_size(self, video_info: sv.VideoInfo) -> int:
        """
        Calculate optimal batch size based on video properties and available memory.
        
        Args:
            video_info: Video information object
            
        Returns:
            Optimal batch size for processing
        """
        # Base batch size on resolution and available memory
        width, height = video_info.width, video_info.height
        pixels_per_frame = width * height
        
        # Estimate memory usage per frame (RGB + processing overhead)
        bytes_per_frame = pixels_per_frame * 3 * 1.5  # RGB + overhead
        
        # Target memory usage (adjust based on GPU memory)
        if self.device == "cuda":
            # Use ~2GB for batch processing on GPU
            target_memory_gb = 2.0
        else:
            # Use ~1GB for CPU processing
            target_memory_gb = 1.0
        
        target_memory_bytes = target_memory_gb * 1024**3
        optimal_batch_size = int(target_memory_bytes / bytes_per_frame)
        
        # Clamp to reasonable bounds
        optimal_batch_size = max(1, min(optimal_batch_size, self.max_batch_size))
        
        logger.info(
            f"Calculated optimal batch size: {optimal_batch_size} "
            f"(resolution: {width}x{height}, device: {self.device})"
        )
        
        return optimal_batch_size
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
