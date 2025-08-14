"""
Data Access Layer Abstraction
Provides abstraction between business logic and data storage
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime, timedelta
from enum import Enum

# Import repository interfaces
from src.core.repositories.base_repository import IRepository
from src.core.repositories.detection_repository import DetectionResult, DetectionMode

T = TypeVar('T')


class QueryOperator(Enum):
    """Query operators for filtering"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt" 
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    IN = "in"
    NOT_IN = "nin"
    CONTAINS = "contains"
    REGEX = "regex"
    EXISTS = "exists"


class SortDirection(Enum):
    """Sort directions"""
    ASC = "asc"
    DESC = "desc"


class QueryFilter:
    """Query filter specification"""
    
    def __init__(self, field: str, operator: QueryOperator, value: Any):
        self.field = field
        self.operator = operator
        self.value = value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value
        }


class QuerySort:
    """Query sort specification"""
    
    def __init__(self, field: str, direction: SortDirection = SortDirection.ASC):
        self.field = field
        self.direction = direction
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "direction": self.direction.value
        }


class Query:
    """Query specification for data access"""
    
    def __init__(self):
        self.filters: List[QueryFilter] = []
        self.sorts: List[QuerySort] = []
        self.limit: Optional[int] = None
        self.offset: Optional[int] = None
        self.fields: Optional[List[str]] = None
    
    def filter(self, field: str, operator: QueryOperator, value: Any) -> 'Query':
        """Add filter to query"""
        self.filters.append(QueryFilter(field, operator, value))
        return self
    
    def equals(self, field: str, value: Any) -> 'Query':
        """Add equals filter"""
        return self.filter(field, QueryOperator.EQUALS, value)
    
    def greater_than(self, field: str, value: Any) -> 'Query':
        """Add greater than filter"""
        return self.filter(field, QueryOperator.GREATER_THAN, value)
    
    def less_than(self, field: str, value: Any) -> 'Query':
        """Add less than filter"""
        return self.filter(field, QueryOperator.LESS_THAN, value)
    
    def contains(self, field: str, value: Any) -> 'Query':
        """Add contains filter"""
        return self.filter(field, QueryOperator.CONTAINS, value)
    
    def in_values(self, field: str, values: List[Any]) -> 'Query':
        """Add 'in' filter"""
        return self.filter(field, QueryOperator.IN, values)
    
    def sort_by(self, field: str, direction: SortDirection = SortDirection.ASC) -> 'Query':
        """Add sort to query"""
        self.sorts.append(QuerySort(field, direction))
        return self
    
    def sort_asc(self, field: str) -> 'Query':
        """Sort ascending"""
        return self.sort_by(field, SortDirection.ASC)
    
    def sort_desc(self, field: str) -> 'Query':
        """Sort descending"""
        return self.sort_by(field, SortDirection.DESC)
    
    def take(self, limit: int) -> 'Query':
        """Set limit"""
        self.limit = limit
        return self
    
    def skip(self, offset: int) -> 'Query':
        """Set offset"""
        self.offset = offset
        return self
    
    def select(self, fields: List[str]) -> 'Query':
        """Select specific fields"""
        self.fields = fields
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary"""
        return {
            "filters": [f.to_dict() for f in self.filters],
            "sorts": [s.to_dict() for s in self.sorts],
            "limit": self.limit,
            "offset": self.offset,
            "fields": self.fields
        }


class PagedResult(Generic[T]):
    """Paged query result"""
    
    def __init__(self, items: List[T], total_count: int, page_size: int, page_number: int):
        self.items = items
        self.total_count = total_count
        self.page_size = page_size
        self.page_number = page_number
        self.total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
        self.has_next = page_number < self.total_pages
        self.has_previous = page_number > 1


class IDataAccessLayer(ABC, Generic[T]):
    """Generic data access layer interface"""
    
    @abstractmethod
    async def create(self, entity: T) -> str:
        """Create new entity"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def find(self, query: Query) -> List[T]:
        """Find entities matching query"""
        pass
    
    @abstractmethod
    async def find_one(self, query: Query) -> Optional[T]:
        """Find single entity matching query"""
        pass
    
    @abstractmethod
    async def find_paged(self, query: Query, page_size: int, page_number: int) -> PagedResult[T]:
        """Find entities with paging"""
        pass
    
    @abstractmethod
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        pass
    
    @abstractmethod
    async def count(self, query: Optional[Query] = None) -> int:
        """Count entities"""
        pass
    
    @abstractmethod
    async def exists(self, query: Query) -> bool:
        """Check if entities exist matching query"""
        pass


class DataAccessLayer(IDataAccessLayer[T]):
    """Generic data access layer implementation"""
    
    def __init__(self, repository: IRepository[T]):
        self.repository = repository
    
    def _convert_query_to_filters(self, query: Query) -> Dict[str, Any]:
        """Convert query to repository filters"""
        filters = {}
        
        for query_filter in query.filters:
            field = query_filter.field
            operator = query_filter.operator
            value = query_filter.value
            
            if operator == QueryOperator.EQUALS:
                filters[field] = value
            elif operator == QueryOperator.GREATER_THAN:
                filters[field] = {"$gt": value}
            elif operator == QueryOperator.GREATER_EQUAL:
                filters[field] = {"$gte": value}
            elif operator == QueryOperator.LESS_THAN:
                filters[field] = {"$lt": value}
            elif operator == QueryOperator.LESS_EQUAL:
                filters[field] = {"$lte": value}
            elif operator == QueryOperator.IN:
                filters[field] = {"$in": value}
            elif operator == QueryOperator.NOT_IN:
                filters[field] = {"$nin": value}
            elif operator == QueryOperator.CONTAINS:
                filters[field] = {"$regex": str(value)}
            elif operator == QueryOperator.REGEX:
                filters[field] = {"$regex": value}
            elif operator == QueryOperator.EXISTS:
                filters[field] = {"$exists": bool(value)}
        
        return filters
    
    def _apply_sorting(self, items: List[T], query: Query) -> List[T]:
        """Apply sorting to results"""
        if not query.sorts:
            return items
        
        # Apply sorts in reverse order (last sort has highest priority)
        for sort_spec in reversed(query.sorts):
            reverse = sort_spec.direction == SortDirection.DESC
            
            # Get sort key function based on field
            def get_sort_key(item):
                if hasattr(item, sort_spec.field):
                    return getattr(item, sort_spec.field)
                elif hasattr(item, '__dict__'):
                    return item.__dict__.get(sort_spec.field)
                else:
                    return None
            
            items.sort(key=get_sort_key, reverse=reverse)
        
        return items
    
    def _apply_pagination(self, items: List[T], query: Query) -> List[T]:
        """Apply pagination to results"""
        start = query.offset or 0
        if query.limit:
            end = start + query.limit
            return items[start:end]
        elif start > 0:
            return items[start:]
        return items
    
    async def create(self, entity: T) -> str:
        """Create new entity"""
        return await self.repository.save(entity)
    
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        return await self.repository.get_by_id(id)
    
    async def find(self, query: Query) -> List[T]:
        """Find entities matching query"""
        filters = self._convert_query_to_filters(query)
        items = await self.repository.get_all(filters)
        
        # Apply sorting and pagination in memory
        # (In a real database, this would be done at the DB level)
        items = self._apply_sorting(items, query)
        items = self._apply_pagination(items, query)
        
        return items
    
    async def find_one(self, query: Query) -> Optional[T]:
        """Find single entity matching query"""
        query.take(1)  # Limit to 1 result
        results = await self.find(query)
        return results[0] if results else None
    
    async def find_paged(self, query: Query, page_size: int, page_number: int) -> PagedResult[T]:
        """Find entities with paging"""
        # Get total count first
        count_query = Query()
        count_query.filters = query.filters  # Same filters for count
        total_count = await self.count(count_query)
        
        # Apply pagination to query
        offset = (page_number - 1) * page_size
        query.skip(offset).take(page_size)
        
        items = await self.find(query)
        
        return PagedResult(
            items=items,
            total_count=total_count,
            page_size=page_size,
            page_number=page_number
        )
    
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        return await self.repository.update(id, updates)
    
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        return await self.repository.delete(id)
    
    async def count(self, query: Optional[Query] = None) -> int:
        """Count entities"""
        if query:
            filters = self._convert_query_to_filters(query)
            return await self.repository.count(filters)
        return await self.repository.count()
    
    async def exists(self, query: Query) -> bool:
        """Check if entities exist matching query"""
        count = await self.count(query)
        return count > 0


class IDetectionDataAccess(IDataAccessLayer[DetectionResult]):
    """Specialized data access interface for detection results"""
    
    @abstractmethod
    async def get_by_user(self, user_id: str, limit: Optional[int] = None) -> List[DetectionResult]:
        """Get detection results by user"""
        pass
    
    @abstractmethod
    async def get_by_text_hash(self, text_hash: str) -> List[DetectionResult]:
        """Get results for same text"""
        pass
    
    @abstractmethod
    async def get_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detection statistics"""
        pass
    
    @abstractmethod
    async def get_recent(self, hours: int = 24, limit: Optional[int] = None) -> List[DetectionResult]:
        """Get recent detection results"""
        pass


class DetectionDataAccess(DataAccessLayer[DetectionResult], IDetectionDataAccess):
    """Data access layer for detection results"""
    
    def __init__(self, detection_repository):
        super().__init__(detection_repository)
        # Cast to specific repository type for additional methods
        self.detection_repo = detection_repository
    
    async def get_by_user(self, user_id: str, limit: Optional[int] = None) -> List[DetectionResult]:
        """Get detection results by user"""
        if hasattr(self.detection_repo, 'get_results_by_user'):
            return await self.detection_repo.get_results_by_user(user_id, limit)
        
        # Fallback to generic query
        query = Query().equals("user_id", user_id).sort_desc("timestamp")
        if limit:
            query.take(limit)
        return await self.find(query)
    
    async def get_by_text_hash(self, text_hash: str) -> List[DetectionResult]:
        """Get results for same text"""
        if hasattr(self.detection_repo, 'get_results_by_text_hash'):
            return await self.detection_repo.get_results_by_text_hash(text_hash)
        
        # Fallback to generic query
        query = Query().equals("text_hash", text_hash)
        return await self.find(query)
    
    async def get_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detection statistics"""
        if hasattr(self.detection_repo, 'get_statistics'):
            return await self.detection_repo.get_statistics(user_id)
        
        # Fallback to basic statistics
        query = Query()
        if user_id:
            query.equals("user_id", user_id)
        
        results = await self.find(query)
        total = len(results)
        ai_count = sum(1 for r in results if r.is_ai)
        
        return {
            "total_detections": total,
            "ai_detected": ai_count,
            "human_detected": total - ai_count,
            "ai_percentage": (ai_count / total * 100) if total > 0 else 0
        }
    
    async def get_recent(self, hours: int = 24, limit: Optional[int] = None) -> List[DetectionResult]:
        """Get recent detection results"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = (Query()
                .greater_than("timestamp", cutoff_time)
                .sort_desc("timestamp"))
        
        if limit:
            query.take(limit)
        
        return await self.find(query)
    
    async def get_by_confidence_range(self, min_confidence: float, max_confidence: float) -> List[DetectionResult]:
        """Get results by confidence range"""
        query = (Query()
                .greater_equal("confidence", min_confidence)
                .less_equal("confidence", max_confidence)
                .sort_desc("confidence"))
        
        return await self.find(query)
    
    async def get_by_mode(self, mode: DetectionMode) -> List[DetectionResult]:
        """Get results by detection mode"""
        query = Query().equals("mode", mode.value)
        return await self.find(query)


# Query Builder Helper Functions

def query() -> Query:
    """Create new query"""
    return Query()


def detection_query() -> Query:
    """Create query for detection results"""
    return Query()


# Common query patterns
class CommonQueries:
    """Common query patterns for detection results"""
    
    @staticmethod
    def recent_ai_detections(hours: int = 24) -> Query:
        """Query for recent AI detections"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return (query()
                .equals("is_ai", True)
                .greater_than("timestamp", cutoff_time)
                .sort_desc("timestamp"))
    
    @staticmethod
    def high_confidence_detections(threshold: float = 0.8) -> Query:
        """Query for high confidence detections"""
        return (query()
                .greater_equal("confidence", threshold)
                .sort_desc("confidence"))
    
    @staticmethod
    def user_detections(user_id: str, limit: int = 50) -> Query:
        """Query for user's detections"""
        return (query()
                .equals("user_id", user_id)
                .sort_desc("timestamp")
                .take(limit))
    
    @staticmethod
    def failed_detections() -> Query:
        """Query for failed/low confidence detections"""
        return (query()
                .less_than("confidence", 0.5)
                .sort_asc("confidence"))


__all__ = [
    'QueryOperator', 'SortDirection', 'QueryFilter', 'QuerySort', 'Query',
    'PagedResult', 'IDataAccessLayer', 'DataAccessLayer',
    'IDetectionDataAccess', 'DetectionDataAccess',
    'query', 'detection_query', 'CommonQueries'
]