"""
Base Repository Pattern Implementation
Provides abstract base class and common functionality for all repositories
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
import json
import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.utils.common import load_json, save_json, ensure_directory, calculate_text_hash

T = TypeVar('T')
logger = logging.getLogger(__name__)


class IRepository(ABC, Generic[T]):
    """Generic repository interface"""
    
    @abstractmethod
    async def save(self, entity: T) -> str:
        """Save entity and return ID"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Get all entities with optional filters"""
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
    async def exists(self, id: str) -> bool:
        """Check if entity exists"""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters"""
        pass


class FileBasedRepository(IRepository[T]):
    """Base file-based repository implementation"""
    
    def __init__(self, storage_path: str, entity_type: str):
        self.storage_path = Path(storage_path)
        self.entity_type = entity_type
        self.data_file = self.storage_path / f"{entity_type}.json"
        self._data = None
        self._lock = asyncio.Lock()
        
        # Ensure storage directory exists
        ensure_directory(self.storage_path)
    
    async def _load_data(self) -> Dict[str, Any]:
        """Load data from file"""
        if self._data is None:
            self._data = load_json(self.data_file, {"entities": {}, "metadata": {}})
        return self._data
    
    async def _save_data(self) -> None:
        """Save data to file"""
        if self._data is not None:
            save_json(self._data, self.data_file)
    
    def _generate_id(self, entity: Any) -> str:
        """Generate unique ID for entity"""
        timestamp = datetime.utcnow().isoformat()
        if hasattr(entity, '__dict__'):
            content = json.dumps(entity.__dict__, sort_keys=True)
        else:
            content = str(entity)
        content_hash = calculate_text_hash(content)[:8]
        return f"{self.entity_type}_{timestamp}_{content_hash}"
    
    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """Serialize entity to dictionary"""
        if hasattr(entity, '__dict__'):
            data = entity.__dict__.copy()
        elif hasattr(entity, 'to_dict'):
            data = entity.to_dict()
        else:
            data = {"value": entity}
        
        data["_type"] = self.entity_type
        data["_created_at"] = datetime.utcnow().isoformat()
        return data
    
    def _deserialize_entity(self, data: Dict[str, Any]) -> T:
        """Deserialize dictionary to entity"""
        # This is a basic implementation - override in specific repositories
        return data
    
    def _matches_filters(self, entity_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if entity matches filters"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key.startswith('_'):
                continue  # Skip metadata fields
                
            entity_value = entity_data.get(key)
            
            if isinstance(value, dict):
                # Complex filter (e.g., {"$gt": 0.5})
                if "$gt" in value and (entity_value is None or entity_value <= value["$gt"]):
                    return False
                if "$lt" in value and (entity_value is None or entity_value >= value["$lt"]):
                    return False
                if "$in" in value and entity_value not in value["$in"]:
                    return False
                if "$regex" in value:
                    import re
                    if not re.search(value["$regex"], str(entity_value)):
                        return False
            else:
                # Simple equality filter
                if entity_value != value:
                    return False
        
        return True
    
    async def save(self, entity: T) -> str:
        """Save entity and return ID"""
        async with self._lock:
            data = await self._load_data()
            entity_id = self._generate_id(entity)
            entity_data = self._serialize_entity(entity)
            entity_data["_id"] = entity_id
            
            data["entities"][entity_id] = entity_data
            data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
            data["metadata"]["total_count"] = len(data["entities"])
            
            await self._save_data()
            logger.debug(f"Saved entity {entity_id} to {self.entity_type} repository")
            return entity_id
    
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        data = await self._load_data()
        entity_data = data["entities"].get(id)
        
        if entity_data:
            return self._deserialize_entity(entity_data)
        return None
    
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Get all entities with optional filters"""
        data = await self._load_data()
        entities = []
        
        for entity_data in data["entities"].values():
            if self._matches_filters(entity_data, filters):
                entities.append(self._deserialize_entity(entity_data))
        
        return entities
    
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        async with self._lock:
            data = await self._load_data()
            
            if id not in data["entities"]:
                return False
            
            data["entities"][id].update(updates)
            data["entities"][id]["_updated_at"] = datetime.utcnow().isoformat()
            data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
            
            await self._save_data()
            logger.debug(f"Updated entity {id} in {self.entity_type} repository")
            return True
    
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        async with self._lock:
            data = await self._load_data()
            
            if id not in data["entities"]:
                return False
            
            del data["entities"][id]
            data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
            data["metadata"]["total_count"] = len(data["entities"])
            
            await self._save_data()
            logger.debug(f"Deleted entity {id} from {self.entity_type} repository")
            return True
    
    async def exists(self, id: str) -> bool:
        """Check if entity exists"""
        data = await self._load_data()
        return id in data["entities"]
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters"""
        data = await self._load_data()
        
        if not filters:
            return len(data["entities"])
        
        count = 0
        for entity_data in data["entities"].values():
            if self._matches_filters(entity_data, filters):
                count += 1
        
        return count
    
    async def get_metadata(self) -> Dict[str, Any]:
        """Get repository metadata"""
        data = await self._load_data()
        return data.get("metadata", {})
    
    async def backup(self, backup_path: str) -> bool:
        """Create backup of repository data"""
        try:
            data = await self._load_data()
            backup_file = Path(backup_path) / f"{self.entity_type}_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            save_json(data, backup_file)
            logger.info(f"Created backup at {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    async def restore(self, backup_file: str) -> bool:
        """Restore repository from backup"""
        try:
            async with self._lock:
                backup_data = load_json(backup_file)
                if backup_data and "entities" in backup_data:
                    self._data = backup_data
                    await self._save_data()
                    logger.info(f"Restored repository from {backup_file}")
                    return True
                else:
                    logger.error(f"Invalid backup file format: {backup_file}")
                    return False
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False


class InMemoryRepository(IRepository[T]):
    """In-memory repository implementation for testing"""
    
    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        self._data = {}
        self._counter = 0
        self._lock = asyncio.Lock()
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        self._counter += 1
        return f"{self.entity_type}_{self._counter}"
    
    async def save(self, entity: T) -> str:
        """Save entity and return ID"""
        async with self._lock:
            entity_id = self._generate_id()
            self._data[entity_id] = {
                "entity": entity,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            return entity_id
    
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        data = self._data.get(id)
        return data["entity"] if data else None
    
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Get all entities with optional filters"""
        entities = []
        for data in self._data.values():
            # Simple filter implementation - extend as needed
            entities.append(data["entity"])
        return entities
    
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        async with self._lock:
            if id not in self._data:
                return False
            
            entity = self._data[id]["entity"]
            if hasattr(entity, '__dict__'):
                entity.__dict__.update(updates)
            
            self._data[id]["updated_at"] = datetime.utcnow()
            return True
    
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        async with self._lock:
            if id in self._data:
                del self._data[id]
                return True
            return False
    
    async def exists(self, id: str) -> bool:
        """Check if entity exists"""
        return id in self._data
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters"""
        return len(self._data)
    
    def clear(self) -> None:
        """Clear all data (for testing)"""
        self._data.clear()
        self._counter = 0


# Unit of Work pattern for transaction management
class UnitOfWork:
    """Unit of Work pattern for managing transactions"""
    
    def __init__(self):
        self._repositories = {}
        self._changes = []
        self._committed = False
    
    def register_repository(self, name: str, repository: IRepository) -> None:
        """Register repository for transaction management"""
        self._repositories[name] = repository
    
    def get_repository(self, name: str) -> Optional[IRepository]:
        """Get registered repository"""
        return self._repositories.get(name)
    
    async def __aenter__(self):
        """Enter async context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager"""
        if exc_type is None and not self._committed:
            await self.commit()
        elif exc_type is not None:
            await self.rollback()
    
    def add_change(self, operation: str, repository: str, 
                   entity_id: str, data: Any = None) -> None:
        """Record change for transaction"""
        self._changes.append({
            "operation": operation,
            "repository": repository,
            "entity_id": entity_id,
            "data": data,
            "timestamp": datetime.utcnow()
        })
    
    async def commit(self) -> None:
        """Commit all changes"""
        try:
            # In a real implementation, you would apply all changes atomically
            logger.info(f"Committing {len(self._changes)} changes")
            self._committed = True
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            await self.rollback()
            raise
    
    async def rollback(self) -> None:
        """Rollback all changes"""
        logger.info(f"Rolling back {len(self._changes)} changes")
        self._changes.clear()


__all__ = [
    'IRepository', 'FileBasedRepository', 'InMemoryRepository', 'UnitOfWork'
]