"""
Detection Repository Implementation
Handles persistence of detection results and related data
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .base_repository import FileBasedRepository
from src.utils.common import normalize_text


class DetectionMode(Enum):
    PATTERN = "pattern"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class DetectionResult:
    """Detection result data model"""
    text: str
    is_ai: bool
    confidence: float
    mode: DetectionMode
    indicators: List[str]
    metadata: Dict[str, Any]
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        # Normalize text for consistent storage
        self.text = normalize_text(self.text)
        
        # Ensure confidence is within valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create from dictionary"""
        # Handle enum conversion
        if isinstance(data.get('mode'), str):
            data['mode'] = DetectionMode(data['mode'])
        
        # Handle datetime conversion
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        return cls(**data)


class IDetectionRepository:
    """Interface for detection result repository"""
    
    async def save_result(self, result: DetectionResult) -> str:
        """Save detection result"""
        pass
    
    async def get_result(self, result_id: str) -> Optional[DetectionResult]:
        """Get detection result by ID"""
        pass
    
    async def get_results_by_user(self, user_id: str, 
                                 limit: Optional[int] = None) -> List[DetectionResult]:
        """Get results for specific user"""
        pass
    
    async def get_results_by_text_hash(self, text_hash: str) -> List[DetectionResult]:
        """Get results for same text (by hash)"""
        pass
    
    async def get_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detection statistics"""
        pass
    
    async def search_results(self, query: Dict[str, Any], 
                           limit: Optional[int] = None) -> List[DetectionResult]:
        """Search results with complex queries"""
        pass


class DetectionRepository(FileBasedRepository[DetectionResult], IDetectionRepository):
    """File-based detection result repository"""
    
    def __init__(self, storage_path: str = "data/results"):
        super().__init__(storage_path, "detection_result")
        self._text_hash_index = {}  # Cache for text hash lookups
    
    def _serialize_entity(self, entity: DetectionResult) -> Dict[str, Any]:
        """Serialize detection result"""
        data = super()._serialize_entity(entity)
        
        # Add searchable fields
        data["text_length"] = len(entity.text)
        data["text_hash"] = self._calculate_text_hash(entity.text)
        data["confidence_bucket"] = self._get_confidence_bucket(entity.confidence)
        data["date"] = entity.timestamp.date().isoformat() if entity.timestamp else None
        
        # Convert enum to string for storage
        if isinstance(entity.mode, DetectionMode):
            data["mode"] = entity.mode.value
        
        return data
    
    def _deserialize_entity(self, data: Dict[str, Any]) -> DetectionResult:
        """Deserialize detection result"""
        # Remove metadata fields
        entity_data = {k: v for k, v in data.items() if not k.startswith('_')}
        
        # Remove computed fields
        for field in ['text_length', 'text_hash', 'confidence_bucket', 'date']:
            entity_data.pop(field, None)
        
        return DetectionResult.from_dict(entity_data)
    
    def _calculate_text_hash(self, text: str) -> str:
        """Calculate hash of normalized text"""
        from src.utils.common import calculate_text_hash
        return calculate_text_hash(normalize_text(text))
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for aggregation"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    async def save_result(self, result: DetectionResult) -> str:
        """Save detection result"""
        result_id = await self.save(result)
        
        # Update text hash index
        text_hash = self._calculate_text_hash(result.text)
        if text_hash not in self._text_hash_index:
            self._text_hash_index[text_hash] = []
        self._text_hash_index[text_hash].append(result_id)
        
        return result_id
    
    async def get_result(self, result_id: str) -> Optional[DetectionResult]:
        """Get detection result by ID"""
        return await self.get_by_id(result_id)
    
    async def get_results_by_user(self, user_id: str, 
                                 limit: Optional[int] = None) -> List[DetectionResult]:
        """Get results for specific user"""
        filters = {"user_id": user_id}
        results = await self.get_all(filters)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda r: r.timestamp or datetime.min, reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    async def get_results_by_text_hash(self, text_hash: str) -> List[DetectionResult]:
        """Get results for same text (by hash)"""
        filters = {"text_hash": text_hash}
        return await self.get_all(filters)
    
    async def get_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detection statistics"""
        filters = {"user_id": user_id} if user_id else None
        results = await self.get_all(filters)
        
        if not results:
            return {
                "total_detections": 0,
                "ai_detected": 0,
                "human_detected": 0,
                "accuracy_by_mode": {},
                "confidence_distribution": {},
                "detection_trends": {}
            }
        
        # Basic statistics
        total = len(results)
        ai_count = sum(1 for r in results if r.is_ai)
        human_count = total - ai_count
        
        # Statistics by mode
        mode_stats = {}
        for result in results:
            mode = result.mode.value if isinstance(result.mode, DetectionMode) else str(result.mode)
            if mode not in mode_stats:
                mode_stats[mode] = {"total": 0, "ai": 0, "human": 0, "avg_confidence": 0}
            
            mode_stats[mode]["total"] += 1
            if result.is_ai:
                mode_stats[mode]["ai"] += 1
            else:
                mode_stats[mode]["human"] += 1
            mode_stats[mode]["avg_confidence"] += result.confidence
        
        # Calculate average confidence by mode
        for mode, stats in mode_stats.items():
            if stats["total"] > 0:
                stats["avg_confidence"] /= stats["total"]
        
        # Confidence distribution
        confidence_buckets = {"high": 0, "medium": 0, "low": 0, "very_low": 0}
        for result in results:
            bucket = self._get_confidence_bucket(result.confidence)
            confidence_buckets[bucket] += 1
        
        # Detection trends (by date)
        date_counts = {}
        for result in results:
            if result.timestamp:
                date = result.timestamp.date().isoformat()
                if date not in date_counts:
                    date_counts[date] = {"ai": 0, "human": 0}
                
                if result.is_ai:
                    date_counts[date]["ai"] += 1
                else:
                    date_counts[date]["human"] += 1
        
        return {
            "total_detections": total,
            "ai_detected": ai_count,
            "human_detected": human_count,
            "ai_percentage": ai_count / total * 100 if total > 0 else 0,
            "statistics_by_mode": mode_stats,
            "confidence_distribution": confidence_buckets,
            "detection_trends": date_counts,
            "average_confidence": sum(r.confidence for r in results) / total if total > 0 else 0,
            "average_processing_time": sum(r.processing_time or 0 for r in results) / total if total > 0 else 0
        }
    
    async def search_results(self, query: Dict[str, Any], 
                           limit: Optional[int] = None) -> List[DetectionResult]:
        """Search results with complex queries"""
        # Build filters from query
        filters = {}
        
        if "user_id" in query:
            filters["user_id"] = query["user_id"]
        
        if "mode" in query:
            filters["mode"] = query["mode"]
        
        if "is_ai" in query:
            filters["is_ai"] = query["is_ai"]
        
        if "min_confidence" in query:
            filters["confidence"] = {"$gt": query["min_confidence"]}
        
        if "max_confidence" in query:
            if "confidence" not in filters:
                filters["confidence"] = {}
            filters["confidence"]["$lt"] = query["max_confidence"]
        
        if "text_contains" in query:
            filters["text"] = {"$regex": query["text_contains"]}
        
        if "date_from" in query:
            filters["date"] = {"$gt": query["date_from"]}
        
        if "date_to" in query:
            if "date" not in filters:
                filters["date"] = {}
            filters["date"]["$lt"] = query["date_to"]
        
        results = await self.get_all(filters)
        
        # Sort results
        sort_field = query.get("sort_by", "timestamp")
        sort_desc = query.get("sort_desc", True)
        
        if sort_field == "timestamp":
            results.sort(key=lambda r: r.timestamp or datetime.min, reverse=sort_desc)
        elif sort_field == "confidence":
            results.sort(key=lambda r: r.confidence, reverse=sort_desc)
        elif sort_field == "text_length":
            results.sort(key=lambda r: len(r.text), reverse=sort_desc)
        
        if limit:
            results = results[:limit]
        
        return results
    
    async def get_duplicate_texts(self, threshold: int = 2) -> Dict[str, List[DetectionResult]]:
        """Find texts that have been analyzed multiple times"""
        text_groups = {}
        all_results = await self.get_all()
        
        for result in all_results:
            text_hash = self._calculate_text_hash(result.text)
            if text_hash not in text_groups:
                text_groups[text_hash] = []
            text_groups[text_hash].append(result)
        
        # Filter groups with multiple entries
        duplicates = {
            hash_val: results 
            for hash_val, results in text_groups.items() 
            if len(results) >= threshold
        }
        
        return duplicates
    
    async def cleanup_old_results(self, days: int = 30) -> int:
        """Remove results older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        all_results = await self.get_all()
        deleted_count = 0
        
        for result in all_results:
            if result.timestamp and result.timestamp < cutoff_date:
                await self.delete(result.id if hasattr(result, 'id') else '')
                deleted_count += 1
        
        return deleted_count
    
    async def export_results(self, filters: Optional[Dict[str, Any]] = None, 
                           format: str = "json") -> str:
        """Export results to file"""
        results = await self.get_all(filters)
        
        if format.lower() == "json":
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_results": len(results),
                "results": [result.to_dict() for result in results]
            }
            
            export_file = f"detection_results_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            from src.utils.common import save_json
            save_json(export_data, self.storage_path / export_file)
            return str(self.storage_path / export_file)
        
        elif format.lower() == "csv":
            import csv
            export_file = f"detection_results_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = self.storage_path / export_file
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                if results:
                    fieldnames = ['text', 'is_ai', 'confidence', 'mode', 'indicators', 
                                'timestamp', 'user_id', 'processing_time', 'model_version']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results:
                        row = {
                            'text': result.text[:500] + '...' if len(result.text) > 500 else result.text,
                            'is_ai': result.is_ai,
                            'confidence': result.confidence,
                            'mode': result.mode.value if isinstance(result.mode, DetectionMode) else str(result.mode),
                            'indicators': '; '.join(result.indicators),
                            'timestamp': result.timestamp.isoformat() if result.timestamp else '',
                            'user_id': result.user_id or '',
                            'processing_time': result.processing_time or '',
                            'model_version': result.model_version or ''
                        }
                        writer.writerow(row)
            
            return str(file_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


__all__ = [
    'DetectionResult', 'DetectionMode', 'IDetectionRepository', 'DetectionRepository'
]