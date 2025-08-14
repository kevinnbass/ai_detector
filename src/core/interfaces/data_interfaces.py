"""
Data Interface Definitions
Interfaces for data processing, collection, and management
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Iterator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from .base_interfaces import IInitializable, IConfigurable, IValidatable, IDisposable


class DataType(Enum):
    """Data type enumeration"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    BINARY = "binary"
    IMAGE = "image"


class DataSource(Enum):
    """Data source enumeration"""
    TWITTER = "twitter"
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"


class DataQuality(Enum):
    """Data quality level"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataSample:
    """Standardized data sample"""
    id: str
    content: Any
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[DataSource] = None
    timestamp: Optional[datetime] = None
    quality: Optional[DataQuality] = None


@dataclass
class DataBatch:
    """Collection of data samples"""
    samples: List[DataSample]
    batch_id: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class IDataSource(ABC):
    """Interface for data sources"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status"""
        pass
    
    @abstractmethod
    async def read(self, query: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DataSample, None]:
        """Read data from source"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get data schema"""
        pass


class IDataSink(ABC):
    """Interface for data sinks"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data sink"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data sink"""
        pass
    
    @abstractmethod
    async def write(self, sample: DataSample) -> bool:
        """Write single data sample"""
        pass
    
    @abstractmethod
    async def write_batch(self, batch: DataBatch) -> bool:
        """Write batch of data samples"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        pass


class IDataCollector(IInitializable, IConfigurable, ABC):
    """Interface for data collectors"""
    
    @abstractmethod
    async def collect(self, count: Optional[int] = None, 
                     filters: Optional[Dict[str, Any]] = None) -> List[DataSample]:
        """
        Collect data samples
        
        Args:
            count: Maximum number of samples to collect
            filters: Collection filters
            
        Returns:
            List of collected data samples
        """
        pass
    
    @abstractmethod
    async def collect_stream(self, filters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[DataSample, None]:
        """Collect data as stream"""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass
    
    @abstractmethod
    def set_collection_rate(self, rate: float) -> None:
        """Set collection rate (samples per second)"""
        pass
    
    @abstractmethod
    def get_supported_sources(self) -> List[DataSource]:
        """Get supported data sources"""
        pass


class IDataProcessor(IInitializable, IConfigurable, ABC):
    """Interface for data processors"""
    
    @abstractmethod
    async def process(self, sample: DataSample) -> DataSample:
        """Process single data sample"""
        pass
    
    @abstractmethod
    async def process_batch(self, batch: DataBatch) -> DataBatch:
        """Process batch of data samples"""
        pass
    
    @abstractmethod
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        pass
    
    @abstractmethod
    def can_process_type(self, data_type: DataType) -> bool:
        """Check if can process given data type"""
        pass


class IDataValidator(ABC):
    """Interface for data validators"""
    
    @abstractmethod
    def validate(self, sample: DataSample) -> tuple[bool, List[str]]:
        """
        Validate data sample
        
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        pass
    
    @abstractmethod
    def validate_batch(self, batch: DataBatch) -> Dict[str, tuple[bool, List[str]]]:
        """Validate batch of samples"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules"""
        pass
    
    @abstractmethod
    def add_validation_rule(self, name: str, rule: callable) -> None:
        """Add custom validation rule"""
        pass


class IDataTransformer(ABC):
    """Interface for data transformers"""
    
    @abstractmethod
    def transform(self, sample: DataSample) -> DataSample:
        """Transform single data sample"""
        pass
    
    @abstractmethod
    def inverse_transform(self, sample: DataSample) -> DataSample:
        """Inverse transform sample"""
        pass
    
    @abstractmethod
    def can_transform(self, from_type: DataType, to_type: DataType) -> bool:
        """Check if can transform between types"""
        pass
    
    @abstractmethod
    def get_transform_schema(self) -> Dict[str, Any]:
        """Get transformation schema"""
        pass


class IDataExporter(ABC):
    """Interface for data exporters"""
    
    @abstractmethod
    async def export(self, samples: List[DataSample], 
                    destination: str, format: DataType) -> bool:
        """Export data samples"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[DataType]:
        """Get supported export formats"""
        pass
    
    @abstractmethod
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics"""
        pass


class IDataImporter(ABC):
    """Interface for data importers"""
    
    @abstractmethod
    async def import_data(self, source: str, format: DataType) -> List[DataSample]:
        """Import data samples"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[DataType]:
        """Get supported import formats"""
        pass
    
    @abstractmethod
    def preview_import(self, source: str, format: DataType, limit: int = 10) -> List[DataSample]:
        """Preview import data"""
        pass


class IDataLabeler(ABC):
    """Interface for data labeling"""
    
    @abstractmethod
    async def label_sample(self, sample: DataSample, label: str, 
                          confidence: Optional[float] = None) -> DataSample:
        """Label a data sample"""
        pass
    
    @abstractmethod
    def get_label_stats(self) -> Dict[str, Any]:
        """Get labeling statistics"""
        pass
    
    @abstractmethod
    def get_available_labels(self) -> List[str]:
        """Get available labels"""
        pass
    
    @abstractmethod
    def suggest_label(self, sample: DataSample) -> tuple[str, float]:
        """Suggest label for sample"""
        pass


class IDataSplitter(ABC):
    """Interface for data splitting"""
    
    @abstractmethod
    def split(self, samples: List[DataSample], 
             ratios: Dict[str, float]) -> Dict[str, List[DataSample]]:
        """Split data into sets"""
        pass
    
    @abstractmethod
    def stratified_split(self, samples: List[DataSample], 
                        ratios: Dict[str, float], 
                        stratify_by: str) -> Dict[str, List[DataSample]]:
        """Stratified data split"""
        pass
    
    @abstractmethod
    def temporal_split(self, samples: List[DataSample], 
                      split_date: datetime) -> Dict[str, List[DataSample]]:
        """Temporal data split"""
        pass


class IDataAugmenter(ABC):
    """Interface for data augmentation"""
    
    @abstractmethod
    async def augment(self, sample: DataSample, 
                     augmentation_type: str) -> List[DataSample]:
        """Augment data sample"""
        pass
    
    @abstractmethod
    def get_augmentation_types(self) -> List[str]:
        """Get available augmentation types"""
        pass
    
    @abstractmethod
    def estimate_augmentation_count(self, sample: DataSample, 
                                  augmentation_type: str) -> int:
        """Estimate number of augmented samples"""
        pass


class IDataQualityAssessor(ABC):
    """Interface for data quality assessment"""
    
    @abstractmethod
    def assess_quality(self, sample: DataSample) -> DataQuality:
        """Assess data quality"""
        pass
    
    @abstractmethod
    def assess_batch_quality(self, batch: DataBatch) -> Dict[str, Any]:
        """Assess quality of data batch"""
        pass
    
    @abstractmethod
    def get_quality_metrics(self) -> List[str]:
        """Get available quality metrics"""
        pass
    
    @abstractmethod
    def get_quality_report(self, samples: List[DataSample]) -> Dict[str, Any]:
        """Generate quality report"""
        pass


class IDataCache(ABC):
    """Interface for data caching"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[DataSample]:
        """Get cached data sample"""
        pass
    
    @abstractmethod
    async def set(self, key: str, sample: DataSample, ttl: Optional[int] = None) -> None:
        """Cache data sample"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached sample"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data"""
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class IDataRepository(ABC):
    """Interface for data repositories"""
    
    @abstractmethod
    async def save(self, sample: DataSample) -> str:
        """Save data sample, return ID"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[DataSample]:
        """Get sample by ID"""
        pass
    
    @abstractmethod
    async def find(self, query: Dict[str, Any]) -> List[DataSample]:
        """Find samples matching query"""
        pass
    
    @abstractmethod
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update sample"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete sample"""
        pass
    
    @abstractmethod
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        pass