"""
Unified Data Collector Module
Consolidates functionality from 6 different collector implementations
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
from abc import ABC, abstractmethod


class CollectorMode(Enum):
    """Collection modes"""
    INTERACTIVE = "interactive"
    BATCH = "batch"
    STREAMING = "streaming"
    FILE_BASED = "file"


class DataSource(Enum):
    """Data source types"""
    MANUAL = "manual"
    TWITTER_API = "twitter_api"
    FILE = "file"
    WEB_SCRAPING = "web_scraping"
    DEMO = "demo"


@dataclass
class Sample:
    """Unified data sample"""
    id: str
    text: str
    label: str
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Sample':
        """Create from dictionary"""
        return cls(**data)


class DataRepository:
    """Repository pattern for data persistence"""
    
    def __init__(self, base_path: str = "../../data"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save(self, filename: str, data: Any) -> None:
        """Save data to file"""
        filepath = os.path.join(self.base_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filename: str) -> Any:
        """Load data from file"""
        filepath = os.path.join(self.base_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def exists(self, filename: str) -> bool:
        """Check if file exists"""
        return os.path.exists(os.path.join(self.base_path, filename))


class CollectorStrategy(ABC):
    """Abstract strategy for different collection methods"""
    
    @abstractmethod
    async def collect(self, config: Dict[str, Any]) -> List[Sample]:
        """Collect data based on configuration"""
        pass


class InteractiveCollector(CollectorStrategy):
    """Interactive command-line collection"""
    
    async def collect(self, config: Dict[str, Any]) -> List[Sample]:
        """Collect data interactively"""
        samples = []
        print("\n" + "="*60)
        print("INTERACTIVE DATA COLLECTION")
        print("="*60)
        print("Format: <text> | <label> | <confidence> (optional)")
        print("Commands: 'save', 'stats', 'quit'")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nEnter data: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'stats':
                    self._print_stats(samples)
                    continue
                elif user_input.lower() == 'save':
                    return samples
                elif not user_input:
                    continue
                
                # Parse input
                parts = user_input.split('|')
                if len(parts) < 2:
                    print("ERROR: Invalid format")
                    continue
                
                text = parts[0].strip()
                label = parts[1].strip()
                confidence = float(parts[2].strip()) if len(parts) > 2 else 1.0
                
                sample = Sample(
                    id=f"sample_{int(time.time())}_{len(samples)}",
                    text=text,
                    label=label,
                    source="interactive",
                    confidence=confidence
                )
                samples.append(sample)
                print(f"SUCCESS: Added {label} sample (total: {len(samples)})")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"ERROR: {e}")
        
        return samples
    
    def _print_stats(self, samples: List[Sample]) -> None:
        """Print collection statistics"""
        if not samples:
            print("No samples collected yet")
            return
        
        labels = {}
        for sample in samples:
            labels[sample.label] = labels.get(sample.label, 0) + 1
        
        print(f"\nStatistics:")
        print(f"  Total samples: {len(samples)}")
        for label, count in labels.items():
            print(f"  {label}: {count} ({count/len(samples)*100:.1f}%)")


class BatchCollector(CollectorStrategy):
    """Batch collection from multiple sources"""
    
    async def collect(self, config: Dict[str, Any]) -> List[Sample]:
        """Collect data in batch"""
        samples = []
        
        # Collect from file if specified
        if 'file_path' in config:
            file_samples = await self._collect_from_file(config['file_path'])
            samples.extend(file_samples)
        
        # Collect from API if configured
        if 'api_config' in config:
            api_samples = await self._collect_from_api(config['api_config'])
            samples.extend(api_samples)
        
        # Add demo data if requested
        if config.get('include_demo', False):
            demo_samples = self._get_demo_samples()
            samples.extend(demo_samples)
        
        return samples
    
    async def _collect_from_file(self, file_path: str) -> List[Sample]:
        """Load samples from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Sample.from_dict(item) for item in data]
        except Exception as e:
            print(f"ERROR loading file: {e}")
            return []
    
    async def _collect_from_api(self, api_config: Dict) -> List[Sample]:
        """Collect from API (placeholder for actual implementation)"""
        # This would implement actual API calls
        print("API collection not yet implemented")
        return []
    
    def _get_demo_samples(self) -> List[Sample]:
        """Get demo samples for testing"""
        return [
            Sample(
                id="demo_1",
                text="While AI continues to evolve, it's important to note both advantages and disadvantages.",
                label="ai",
                source="demo",
                confidence=0.95
            ),
            Sample(
                id="demo_2", 
                text="AI is moving way too fast tbh, can't keep up anymore lol",
                label="human",
                source="demo",
                confidence=0.92
            )
        ]


class UnifiedDataCollector:
    """
    Unified data collector that consolidates all collection functionality
    Replaces: data_collector.py, simple_collector.py, tweet_data_collector.py,
              test_collector.py, demo_optimization.py, run_optimization.py
    """
    
    def __init__(self, mode: CollectorMode = CollectorMode.INTERACTIVE,
                 source: DataSource = DataSource.MANUAL,
                 data_file: str = "unified_dataset.json"):
        self.mode = mode
        self.source = source
        self.data_file = data_file
        self.repository = DataRepository()
        self.samples: List[Sample] = []
        
        # Load existing data
        self._load_existing_data()
        
        # Set strategy based on mode
        self.strategy = self._get_strategy()
    
    def _get_strategy(self) -> CollectorStrategy:
        """Get appropriate collection strategy"""
        if self.mode == CollectorMode.INTERACTIVE:
            return InteractiveCollector()
        elif self.mode == CollectorMode.BATCH:
            return BatchCollector()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _load_existing_data(self) -> None:
        """Load existing data from repository"""
        data = self.repository.load(self.data_file)
        if data:
            self.samples = [Sample.from_dict(item) for item in data.get('samples', [])]
            print(f"Loaded {len(self.samples)} existing samples")
        else:
            print("Starting with empty dataset")
    
    async def collect(self, config: Optional[Dict[str, Any]] = None) -> List[Sample]:
        """
        Main collection method - delegates to strategy
        
        Args:
            config: Configuration dictionary for collection
            
        Returns:
            List of collected samples
        """
        config = config or {}
        new_samples = await self.strategy.collect(config)
        self.samples.extend(new_samples)
        return new_samples
    
    def add_sample(self, text: str, label: str, source: Optional[str] = None,
                   confidence: float = 1.0, metadata: Optional[Dict] = None) -> bool:
        """
        Add a single sample
        
        Args:
            text: Sample text
            label: Classification label
            source: Data source
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if not text or len(text.strip()) < 10:
            print("ERROR: Text too short")
            return False
        
        if not label:
            print("ERROR: Label required")
            return False
        
        # Check for duplicates
        for existing in self.samples:
            if existing.text.strip() == text.strip():
                print("WARNING: Duplicate detected, skipping")
                return False
        
        sample = Sample(
            id=f"sample_{int(time.time())}_{len(self.samples)}",
            text=text.strip(),
            label=label,
            source=source or self.source.value,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.samples.append(sample)
        print(f"SUCCESS: Added {label} sample (total: {len(self.samples)})")
        return True
    
    def bulk_add(self, samples_data: List[Dict[str, Any]]) -> int:
        """
        Add multiple samples at once
        
        Args:
            samples_data: List of sample dictionaries
            
        Returns:
            Number of samples added
        """
        added = 0
        for data in samples_data:
            if self.add_sample(**data):
                added += 1
        
        print(f"Added {added}/{len(samples_data)} samples")
        self.save()
        return added
    
    def save(self) -> None:
        """Save current samples to repository"""
        data = {
            'samples': [s.to_dict() for s in self.samples],
            'metadata': {
                'total_samples': len(self.samples),
                'labels': self._get_label_counts(),
                'sources': self._get_source_counts(),
                'last_updated': datetime.now().isoformat(),
                'version': '2.1.0'
            }
        }
        self.repository.save(self.data_file, data)
        print(f"Saved {len(self.samples)} samples")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.samples:
            return {'total_samples': 0, 'message': 'No samples collected'}
        
        return {
            'total_samples': len(self.samples),
            'labels': self._get_label_counts(),
            'sources': self._get_source_counts(),
            'avg_confidence': sum(s.confidence for s in self.samples) / len(self.samples),
            'date_range': self._get_date_range()
        }
    
    def get_balanced_sample(self, max_samples: int = 50) -> Dict[str, List[Sample]]:
        """Get balanced samples by label"""
        label_groups = {}
        for sample in self.samples:
            if sample.label not in label_groups:
                label_groups[sample.label] = []
            label_groups[sample.label].append(sample)
        
        # Balance the samples
        min_count = min(len(samples) for samples in label_groups.values())
        sample_size = min(max_samples // len(label_groups), min_count)
        
        balanced = {}
        for label, samples in label_groups.items():
            balanced[label] = samples[:sample_size]
        
        return balanced
    
    def export(self, format: str = 'json', filename: Optional[str] = None) -> str:
        """
        Export data in specified format
        
        Args:
            format: Export format (json, csv, etc.)
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"export_{timestamp}.{format}"
        
        if format == 'json':
            self.repository.save(filename, {
                'samples': [s.to_dict() for s in self.samples],
                'export_info': {
                    'total': len(self.samples),
                    'timestamp': datetime.now().isoformat(),
                    'format': format
                }
            })
        elif format == 'csv':
            # CSV export implementation
            pass
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        filepath = os.path.join(self.repository.base_path, filename)
        print(f"Exported to {filepath}")
        return filepath
    
    def filter_samples(self, **criteria) -> List[Sample]:
        """
        Filter samples by criteria
        
        Args:
            **criteria: Filter criteria (label, source, confidence_min, etc.)
            
        Returns:
            Filtered samples
        """
        filtered = self.samples
        
        if 'label' in criteria:
            filtered = [s for s in filtered if s.label == criteria['label']]
        
        if 'source' in criteria:
            filtered = [s for s in filtered if s.source == criteria['source']]
        
        if 'confidence_min' in criteria:
            filtered = [s for s in filtered if s.confidence >= criteria['confidence_min']]
        
        if 'confidence_max' in criteria:
            filtered = [s for s in filtered if s.confidence <= criteria['confidence_max']]
        
        return filtered
    
    def _get_label_counts(self) -> Dict[str, int]:
        """Get count by label"""
        counts = {}
        for sample in self.samples:
            counts[sample.label] = counts.get(sample.label, 0) + 1
        return counts
    
    def _get_source_counts(self) -> Dict[str, int]:
        """Get count by source"""
        counts = {}
        for sample in self.samples:
            counts[sample.source] = counts.get(sample.source, 0) + 1
        return counts
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get date range of samples"""
        if not self.samples:
            return {}
        
        dates = [s.timestamp for s in self.samples]
        return {
            'earliest': min(dates),
            'latest': max(dates)
        }
    
    def clear(self) -> None:
        """Clear all samples"""
        self.samples = []
        print("Cleared all samples")
    
    def validate(self) -> Dict[str, Any]:
        """Validate dataset quality"""
        issues = []
        
        # Check for duplicates
        texts = [s.text for s in self.samples]
        if len(texts) != len(set(texts)):
            issues.append("Duplicate texts found")
        
        # Check label balance
        label_counts = self._get_label_counts()
        if label_counts:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            if max_count > min_count * 2:
                issues.append("Imbalanced labels")
        
        # Check for empty fields
        for i, sample in enumerate(self.samples):
            if not sample.text.strip():
                issues.append(f"Empty text in sample {i}")
            if not sample.label:
                issues.append(f"Missing label in sample {i}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_samples': len(self.samples),
            'quality_score': (1 - len(issues) / max(len(self.samples), 1)) * 100
        }


# Convenience functions for backward compatibility
def create_collector(mode: str = 'interactive', source: str = 'manual') -> UnifiedDataCollector:
    """Factory function to create collector"""
    collector_mode = CollectorMode(mode)
    data_source = DataSource(source)
    return UnifiedDataCollector(collector_mode, data_source)


async def quick_collect(num_samples: int = 10, mode: str = 'batch') -> List[Sample]:
    """Quick collection helper"""
    collector = create_collector(mode)
    config = {'include_demo': True} if mode == 'batch' else {}
    return await collector.collect(config)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create unified collector
        collector = UnifiedDataCollector(
            mode=CollectorMode.INTERACTIVE,
            source=DataSource.MANUAL
        )
        
        # Collect data
        await collector.collect()
        
        # Print statistics
        stats = collector.get_statistics()
        print(f"\nFinal statistics: {stats}")
        
        # Validate dataset
        validation = collector.validate()
        print(f"Validation: {validation}")
    
    asyncio.run(main())