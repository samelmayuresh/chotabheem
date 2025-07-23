# utils/performance_optimizer.py - Performance optimization for music playback system
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from functools import lru_cache, wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = max_metrics
        self.lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric"""
        with self.lock:
            self.metrics.append(metric)
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def get_average_duration(self, operation: str, hours: int = 1) -> float:
        """Get average duration for an operation"""
        cutoff_time = time.time() - (hours * 3600)
        
        relevant_metrics = [
            m for m in self.metrics 
            if m.operation == operation and m.start_time > cutoff_time and m.success
        ]
        
        if not relevant_metrics:
            return 0.0
        
        return sum(m.duration for m in relevant_metrics) / len(relevant_metrics)
    
    def get_success_rate(self, operation: str, hours: int = 1) -> float:
        """Get success rate for an operation"""
        cutoff_time = time.time() - (hours * 3600)
        
        relevant_metrics = [
            m for m in self.metrics 
            if m.operation == operation and m.start_time > cutoff_time
        ]
        
        if not relevant_metrics:
            return 1.0
        
        success_count = sum(1 for m in relevant_metrics if m.success)
        return success_count / len(relevant_metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {"message": "No metrics recorded"}
        
        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "success_count": 0,
                    "avg_duration": 0.0,
                    "success_rate": 0.0
                }
            
            op = operations[metric.operation]
            op["count"] += 1
            op["total_duration"] += metric.duration
            if metric.success:
                op["success_count"] += 1
        
        # Calculate averages
        for op_data in operations.values():
            op_data["avg_duration"] = op_data["total_duration"] / op_data["count"]
            op_data["success_rate"] = op_data["success_count"] / op_data["count"]
        
        return operations

def performance_monitor(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                metric = PerformanceMetrics(
                    operation=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=success,
                    error_message=error_message
                )
                
                # Record metric if monitor exists
                if hasattr(wrapper, '_monitor'):
                    wrapper._monitor.record_metric(metric)
        
        return wrapper
    return decorator

class PlaylistCache:
    """Cache for playlist generation to improve performance"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def _generate_key(self, emotion: str, mode: str, count: int, preferences: Dict[str, Any]) -> str:
        """Generate cache key"""
        pref_str = json.dumps(preferences, sort_keys=True) if preferences else ""
        return f"{emotion}_{mode}_{count}_{hash(pref_str)}"
    
    def get(self, emotion: str, mode: str, count: int, preferences: Dict[str, Any] = None) -> Optional[List]:
        """Get cached playlist"""
        key = self._generate_key(emotion, mode, count, preferences)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                # Check if entry is still valid
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    return entry["playlist"]
                else:
                    # Remove expired entry
                    del self.cache[key]
        
        return None
    
    def put(self, emotion: str, mode: str, count: int, playlist: List, preferences: Dict[str, Any] = None):
        """Cache playlist"""
        key = self._generate_key(emotion, mode, count, preferences)
        
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]
            
            self.cache[key] = {
                "playlist": playlist,
                "timestamp": time.time()
            }
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            current_time = time.time()
            valid_entries = sum(
                1 for entry in self.cache.values()
                if current_time - entry["timestamp"] < self.ttl_seconds
            )
            
            return {
                "total_entries": len(self.cache),
                "valid_entries": valid_entries,
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_request_count', 1), 1)
            }

class AsyncPlaylistGenerator:
    """Asynchronous playlist generation for better performance"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = PlaylistCache()
    
    async def generate_playlist_async(self, 
                                    song_database,
                                    emotion: str,
                                    mode: str,
                                    count: int,
                                    preferences: Dict[str, Any] = None) -> List:
        """Generate playlist asynchronously"""
        
        # Check cache first
        cached_playlist = self.cache.get(emotion, mode, count, preferences)
        if cached_playlist:
            return cached_playlist
        
        # Generate playlist in thread pool
        loop = asyncio.get_event_loop()
        playlist = await loop.run_in_executor(
            self.executor,
            self._generate_playlist_sync,
            song_database,
            emotion,
            mode,
            count,
            preferences
        )
        
        # Cache result
        self.cache.put(emotion, mode, count, playlist, preferences)
        
        return playlist
    
    def _generate_playlist_sync(self, song_database, emotion, mode, count, preferences):
        """Synchronous playlist generation (runs in thread pool)"""
        if mode == "targeted":
            return song_database.get_vocal_songs_for_emotion(emotion, count)
        elif mode == "full_session":
            return song_database.get_mixed_playlist(emotion, 0.7, count)
        else:  # custom
            vocal_ratio = preferences.get("vocal_ratio", 0.7) if preferences else 0.7
            return song_database.get_mixed_playlist(emotion, vocal_ratio, count)
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.cache.clear()

class MemoryOptimizer:
    """Optimize memory usage for music playback"""
    
    def __init__(self):
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
        self.cleanup_callbacks: List[Callable] = []
    
    def register_cleanup_callback(self, callback: Callable):
        """Register cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is above threshold"""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            return memory_usage > self.memory_threshold
        except ImportError:
            # psutil not available, assume memory is fine
            return False
    
    def cleanup_memory(self):
        """Trigger memory cleanup"""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logging.warning(f"Memory cleanup callback failed: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()

class PerformanceOptimizer:
    """Main performance optimizer for music playback system"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.playlist_cache = PlaylistCache()
        self.async_generator = AsyncPlaylistGenerator()
        self.memory_optimizer = MemoryOptimizer()
        
        # Performance settings
        self.settings = {
            "enable_caching": True,
            "enable_async_generation": True,
            "enable_memory_optimization": True,
            "cache_ttl": 3600,  # 1 hour
            "max_cache_size": 100,
            "memory_check_interval": 300  # 5 minutes
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        if self.settings["enable_memory_optimization"]:
            self._start_memory_monitor()
    
    def _start_memory_monitor(self):
        """Start memory monitoring thread"""
        def memory_monitor():
            while True:
                try:
                    if self.memory_optimizer.check_memory_usage():
                        logging.info("High memory usage detected, triggering cleanup")
                        self.memory_optimizer.cleanup_memory()
                    
                    time.sleep(self.settings["memory_check_interval"])
                except Exception as e:
                    logging.error(f"Memory monitor error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        thread = threading.Thread(target=memory_monitor, daemon=True)
        thread.start()
    
    @performance_monitor("playlist_generation")
    def optimize_playlist_generation(self, 
                                   song_database,
                                   emotion: str,
                                   mode: str,
                                   count: int,
                                   preferences: Dict[str, Any] = None) -> List:
        """Optimized playlist generation"""
        
        if self.settings["enable_caching"]:
            # Check cache first
            cached_playlist = self.playlist_cache.get(emotion, mode, count, preferences)
            if cached_playlist:
                return cached_playlist
        
        # Generate playlist
        if mode == "targeted":
            playlist = song_database.get_vocal_songs_for_emotion(emotion, count)
        elif mode == "full_session":
            playlist = song_database.get_mixed_playlist(emotion, 0.7, count)
        else:  # custom
            vocal_ratio = preferences.get("vocal_ratio", 0.7) if preferences else 0.7
            playlist = song_database.get_mixed_playlist(emotion, vocal_ratio, count)
        
        # Cache result
        if self.settings["enable_caching"]:
            self.playlist_cache.put(emotion, mode, count, playlist, preferences)
        
        return playlist
    
    @performance_monitor("song_loading")
    def optimize_song_loading(self, song, preload_next: bool = True):
        """Optimize song loading with preloading"""
        
        # This would implement actual song loading optimization
        # For now, it's a placeholder that demonstrates the pattern
        
        start_time = time.time()
        
        try:
            # Simulate song loading
            time.sleep(0.1)  # Simulated loading time
            
            if preload_next:
                # Preload next song in background
                threading.Thread(
                    target=self._preload_next_song,
                    args=(song,),
                    daemon=True
                ).start()
            
            return True
            
        except Exception as e:
            logging.error(f"Song loading failed: {e}")
            return False
    
    def _preload_next_song(self, current_song):
        """Preload next song in background"""
        # This would implement actual preloading logic
        pass
    
    @performance_monitor("database_query")
    @lru_cache(maxsize=128)
    def optimize_database_queries(self, emotion: str, count: int, filters: str = ""):
        """Optimize database queries with caching"""
        
        # This would implement optimized database queries
        # The @lru_cache decorator provides automatic caching
        
        # Simulate database query
        time.sleep(0.05)
        return f"optimized_query_result_{emotion}_{count}_{filters}"
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            "monitor_summary": self.monitor.get_performance_summary(),
            "cache_stats": self.playlist_cache.get_stats(),
            "memory_stats": self._get_memory_stats(),
            "settings": self.settings,
            "recommendations": self._get_performance_recommendations()
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
                "available": psutil.virtual_memory().available
            }
        except ImportError:
            return {"message": "psutil not available for memory stats"}
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        
        recommendations = []
        
        # Check cache hit rate
        cache_stats = self.playlist_cache.get_stats()
        if cache_stats.get("hit_rate", 0) < 0.5:
            recommendations.append("Consider increasing cache size or TTL for better hit rate")
        
        # Check average operation times
        avg_playlist_time = self.monitor.get_average_duration("playlist_generation")
        if avg_playlist_time > 1.0:
            recommendations.append("Playlist generation is slow, consider optimizing song database queries")
        
        # Check memory usage
        if self.memory_optimizer.check_memory_usage():
            recommendations.append("High memory usage detected, consider reducing cache sizes")
        
        # Check success rates
        playlist_success_rate = self.monitor.get_success_rate("playlist_generation")
        if playlist_success_rate < 0.95:
            recommendations.append("Low playlist generation success rate, check error logs")
        
        return recommendations
    
    def optimize_settings(self, **kwargs):
        """Update optimization settings"""
        
        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value
                logging.info(f"Updated setting {key} to {value}")
        
        # Apply settings changes
        if "cache_ttl" in kwargs:
            self.playlist_cache.ttl_seconds = kwargs["cache_ttl"]
        
        if "max_cache_size" in kwargs:
            self.playlist_cache.max_size = kwargs["max_cache_size"]
    
    def cleanup(self):
        """Cleanup optimizer resources"""
        
        self.playlist_cache.clear()
        self.async_generator.cleanup()
        
        # Clear function caches if they exist
        try:
            if hasattr(self.optimize_database_queries, 'cache_clear'):
                self.optimize_database_queries.cache_clear()
        except AttributeError:
            pass
        
        logging.info("PerformanceOptimizer cleaned up")

# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

def optimize_function(operation_name: str):
    """Decorator to optimize function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            # Set monitor for performance tracking
            monitor_decorator = performance_monitor(operation_name)
            monitored_func = monitor_decorator(func)
            monitored_func._monitor = optimizer.monitor
            
            return monitored_func(*args, **kwargs)
        
        return wrapper
    return decorator