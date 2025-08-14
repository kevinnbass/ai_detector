#!/usr/bin/env python3
"""
Simple data collector - no external dependencies
Just collects your labeled examples into a JSON file
"""

import json
import os
from datetime import datetime

def simple_data_collector():
    """Simple interactive data collection"""
    
    # Load existing data
    data_file = "../data/labeled_dataset.json" 
    dataset = []
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                dataset = data.get('samples', [])
            print(f"Loaded {len(dataset)} existing samples")
        except:
            print("Starting fresh dataset")
    else:
        print("Starting fresh dataset")
        
    print("\n" + "="*50)
    print("SIMPLE DATA COLLECTOR")
    print("="*50)
    print("Commands:")
    print("  add gpt4o <text>   - Add GPT-4o sample")  
    print("  add human <text>   - Add human sample")
    print("  list               - Show all samples")
    print("  stats              - Show statistics")
    print("  save               - Save dataset")
    print("  quit               - Exit")
    print("="*50)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
                
            elif command.lower() == 'stats':
                gpt4o_count = len([s for s in dataset if s['label'] == 'gpt4o'])
                human_count = len([s for s in dataset if s['label'] == 'human'])
                print(f"\nDataset Statistics:")
                print(f"  Total samples: {len(dataset)}")
                print(f"  GPT-4o samples: {gpt4o_count}")
                print(f"  Human samples: {human_count}")
                if len(dataset) > 0:
                    balance = gpt4o_count / human_count if human_count > 0 else float('inf')
                    print(f"  Balance ratio: {balance:.2f}")
                
            elif command.lower() == 'list':
                print(f"\nAll {len(dataset)} samples:")
                for i, sample in enumerate(dataset, 1):
                    label = sample['label'].upper()
                    text = sample['text'][:60] + "..." if len(sample['text']) > 60 else sample['text']
                    print(f"  {i}. [{label}] {text}")
                    
            elif command.lower() == 'save':
                save_dataset(dataset, data_file)
                
            elif command.startswith('add '):
                parts = command.split(' ', 2)
                if len(parts) >= 3:
                    _, label, text = parts
                    if label.lower() in ['gpt4o', 'human']:
                        # Add sample
                        sample = {
                            'id': f"sample_{len(dataset)+1:03d}",
                            'text': text.strip(),
                            'label': label.lower(),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'manual'
                        }
                        dataset.append(sample)
                        print(f"✅ Added {label.lower()} sample ({len(text)} chars)")
                    else:
                        print("Label must be 'gpt4o' or 'human'")
                else:
                    print("Usage: add <gpt4o|human> <text>")
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nSaving dataset...")
    save_dataset(dataset, data_file)
    print("Goodbye!")

def save_dataset(dataset, filename):
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    gpt4o_count = len([s for s in dataset if s['label'] == 'gpt4o'])
    human_count = len([s for s in dataset if s['label'] == 'human'])
    
    data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'total_samples': len(dataset),
            'gpt4o_samples': gpt4o_count,
            'human_samples': human_count,
            'version': '1.0'
        },
        'samples': dataset
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved {len(dataset)} samples to {filename}")

if __name__ == "__main__":
    simple_data_collector()