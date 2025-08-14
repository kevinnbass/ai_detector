import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import hashlib
import os
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some statistics features disabled.")

class DataCollector:
    def __init__(self, data_file: str = "../data/labeled_dataset.json"):
        self.data_file = data_file
        self.dataset = []
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing labeled data if file exists"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.dataset = data.get('samples', [])
                    print(f"Loaded {len(self.dataset)} existing samples")
            except Exception as e:
                print(f"Error loading existing data: {e}")
                self.dataset = []
        else:
            print("No existing dataset found, starting fresh")
            self.dataset = []
    
    def add_sample(self, text: str, label: str, confidence: float = 1.0, 
                   source: str = "manual", metadata: Dict[str, Any] = None) -> str:
        """
        Add a labeled sample to the dataset
        
        Args:
            text: The text content
            label: 'gpt4o' or 'human'
            confidence: How confident you are in the label (0.0-1.0)
            source: Where the sample came from ('manual', 'twitter', 'generated', etc.)
            metadata: Additional information about the sample
        
        Returns:
            sample_id: Unique identifier for the sample
        """
        if label not in ['gpt4o', 'human']:
            raise ValueError("Label must be 'gpt4o' or 'human'")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate unique ID based on text content
        sample_id = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        
        # Check if sample already exists
        for sample in self.dataset:
            if sample['id'] == sample_id:
                print(f"Sample already exists: {sample_id}")
                return sample_id
        
        sample = {
            'id': sample_id,
            'text': text.strip(),
            'label': label,
            'confidence': confidence,
            'source': source,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'character_count': len(text.strip()),
            'word_count': len(text.strip().split()),
            'metadata': metadata or {}
        }
        
        self.dataset.append(sample)
        print(f"Added sample {sample_id}: {label} ({len(text)} chars)")
        return sample_id
    
    def add_batch_from_file(self, file_path: str, label: str, source: str = "file_import"):
        """
        Add multiple samples from a text file (one per line)
        
        Args:
            file_path: Path to text file with one sample per line
            label: 'gpt4o' or 'human' for all samples in file
            source: Source description
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            added = 0
            for line in lines:
                text = line.strip()
                if text and len(text) > 20:  # Minimum length filter
                    self.add_sample(text, label, source=source)
                    added += 1
            
            print(f"Added {added} samples from {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    def add_twitter_samples(self, tweets: List[Dict[str, Any]], label: str):
        """
        Add samples from Twitter data
        
        Args:
            tweets: List of tweet objects with 'text' field
            label: 'gpt4o' or 'human'
        """
        added = 0
        for tweet in tweets:
            if 'text' in tweet:
                metadata = {
                    'platform': 'twitter',
                    'tweet_id': tweet.get('id'),
                    'user': tweet.get('user', {}).get('screen_name'),
                    'created_at': tweet.get('created_at'),
                    'retweet_count': tweet.get('retweet_count', 0),
                    'favorite_count': tweet.get('favorite_count', 0)
                }
                
                self.add_sample(
                    text=tweet['text'],
                    label=label,
                    source='twitter',
                    metadata=metadata
                )
                added += 1
        
        print(f"Added {added} Twitter samples")
    
    def save_dataset(self):
        """Save the current dataset to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        data = {
            'metadata': {
                'created': datetime.now(timezone.utc).isoformat(),
                'total_samples': len(self.dataset),
                'gpt4o_samples': len([s for s in self.dataset if s['label'] == 'gpt4o']),
                'human_samples': len([s for s in self.dataset if s['label'] == 'human']),
                'sources': list(set(s['source'] for s in self.dataset)),
                'version': '1.0'
            },
            'samples': self.dataset
        }
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.dataset)} samples to {self.data_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.dataset:
            return {
                'total_samples': 0,
                'gpt4o_samples': 0,
                'human_samples': 0,
                'balance_ratio': 0,
                'avg_length': 0,
                'avg_words': 0,
                'sources': [],
                'confidence_avg': 0
            }
        
        # Calculate statistics manually (works with or without pandas)
        gpt4o_samples = [s for s in self.dataset if s['label'] == 'gpt4o']
        human_samples = [s for s in self.dataset if s['label'] == 'human']
        
        # Source counting
        sources = {}
        for sample in self.dataset:
            source = sample.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        # Calculate averages
        char_counts = [s.get('character_count', len(s['text'])) for s in self.dataset]
        word_counts = [s.get('word_count', len(s['text'].split())) for s in self.dataset]
        confidences = [s.get('confidence', 1.0) for s in self.dataset]
        timestamps = [s.get('timestamp', '') for s in self.dataset]
        
        stats = {
            'total_samples': len(self.dataset),
            'gpt4o_samples': len(gpt4o_samples),
            'human_samples': len(human_samples),
            'balance_ratio': len(gpt4o_samples) / len(human_samples) if len(human_samples) > 0 else float('inf'),
            'avg_length': sum(char_counts) / len(char_counts) if char_counts else 0,
            'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            'sources': sources,
            'confidence_avg': sum(confidences) / len(confidences) if confidences else 0,
            'date_range': {
                'earliest': min(timestamps) if timestamps else '',
                'latest': max(timestamps) if timestamps else ''
            }
        }
        
        return stats
    
    def get_samples_by_label(self, label: str) -> List[Dict[str, Any]]:
        """Get all samples with specific label"""
        return [s for s in self.dataset if s['label'] == label]
    
    def remove_sample(self, sample_id: str) -> bool:
        """Remove a sample by ID"""
        for i, sample in enumerate(self.dataset):
            if sample['id'] == sample_id:
                removed = self.dataset.pop(i)
                print(f"Removed sample {sample_id}: {removed['label']}")
                return True
        
        print(f"Sample {sample_id} not found")
        return False
    
    def update_sample_label(self, sample_id: str, new_label: str, confidence: float = 1.0):
        """Update the label of an existing sample"""
        if new_label not in ['gpt4o', 'human']:
            raise ValueError("Label must be 'gpt4o' or 'human'")
        
        for sample in self.dataset:
            if sample['id'] == sample_id:
                old_label = sample['label']
                sample['label'] = new_label
                sample['confidence'] = confidence
                sample['updated'] = datetime.now(timezone.utc).isoformat()
                print(f"Updated sample {sample_id}: {old_label} → {new_label}")
                return True
        
        print(f"Sample {sample_id} not found")
        return False
    
    def export_for_training(self, train_ratio: float = 0.8) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export dataset split for training and validation
        
        Args:
            train_ratio: Fraction of data to use for training
        
        Returns:
            Dictionary with 'train' and 'val' splits
        """
        if not self.dataset:
            return {'train': [], 'val': []}
        
        # Separate samples by label
        gpt4o_samples = [s for s in self.dataset if s['label'] == 'gpt4o']
        human_samples = [s for s in self.dataset if s['label'] == 'human']
        
        # Shuffle the samples
        import random
        gpt4o_samples = gpt4o_samples.copy()
        human_samples = human_samples.copy()
        random.shuffle(gpt4o_samples)
        random.shuffle(human_samples)
        
        gpt4o_train_size = int(len(gpt4o_samples) * train_ratio)
        human_train_size = int(len(human_samples) * train_ratio)
        
        train_samples = (gpt4o_samples[:gpt4o_train_size] + 
                        human_samples[:human_train_size])
        
        val_samples = (gpt4o_samples[gpt4o_train_size:] + 
                      human_samples[human_train_size:])
        
        # Shuffle the splits
        import random
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        
        return {
            'train': train_samples,
            'val': val_samples
        }
    
    def interactive_labeling_session(self):
        """Interactive command-line session for labeling data"""
        print("\n" + "="*60)
        print("INTERACTIVE LABELING SESSION")
        print("="*60)
        print("Commands:")
        print("  add <text> <label>  - Add sample (label: gpt4o/human)")
        print("  stats               - Show dataset statistics")
        print("  save                - Save current dataset")
        print("  quit                - Exit session")
        print("="*60)
        
        while True:
            try:
                try:
                    command = input("\n> ").strip()
                except EOFError:
                    # Handle Ctrl+C or EOF gracefully
                    print("\nExiting...")
                    break
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'stats':
                    stats = self.get_statistics()
                    print(f"\nDataset Statistics:")
                    print(f"  Total samples: {stats['total_samples']}")
                    print(f"  GPT-4o samples: {stats['gpt4o_samples']}")
                    print(f"  Human samples: {stats['human_samples']}")
                    if stats['total_samples'] > 0:
                        print(f"  Balance ratio: {stats['balance_ratio']:.2f}")
                        print(f"  Average length: {stats['avg_length']:.0f} chars")
                elif command.lower() == 'save':
                    self.save_dataset()
                elif command.startswith('add '):
                    parts = command.split(' ', 2)
                    if len(parts) >= 3:
                        _, label, text = parts
                        if label in ['gpt4o', 'human']:
                            sample_id = self.add_sample(text, label, source='interactive')
                            print(f"Added: {sample_id}")
                        else:
                            print("Label must be 'gpt4o' or 'human'")
                    else:
                        print("Usage: add <label> <text>")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nSaving dataset before exit...")
        self.save_dataset()

# Usage example and CLI interface
def main():
    collector = DataCollector()
    
    # Show current stats
    stats = collector.get_statistics()
    print(f"Current dataset: {stats['total_samples']} samples")
    
    # Interactive session
    collector.interactive_labeling_session()

if __name__ == "__main__":
    main()