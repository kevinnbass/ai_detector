#!/usr/bin/env python3
"""
Test script to verify data collector works correctly
"""

import sys
import os
sys.path.append('mining')

from mining.data_collector import DataCollector

def test_data_collector():
    """Test all functionality of data collector"""
    print("Testing Data Collector...")
    print("=" * 40)
    
    # Initialize
    collector = DataCollector("data/test_dataset.json")
    
    # Test 1: Initial stats
    print("Test 1: Initial statistics")
    stats = collector.get_statistics()
    print(f"  Total samples: {stats['total_samples']}")
    assert stats['total_samples'] >= 0
    print("  [PASS]")
    
    # Test 2: Add GPT-4o sample
    print("\nTest 2: Add GPT-4o sample")
    gpt_text = "While artificial intelligence continues to evolve, it's important to note both advantages and disadvantages. On one hand, productivity increases. On the other hand, concerns persist."
    sample_id = collector.add_sample(gpt_text, 'gpt4o', source='test')
    print(f"  Added sample: {sample_id}")
    assert sample_id is not None
    print("  [PASS]")
    
    # Test 3: Add human sample  
    print("\nTest 3: Add human sample")
    human_text = "AI is moving way too fast tbh can barely keep up anymore 😅"
    sample_id2 = collector.add_sample(human_text, 'human', source='test')
    print(f"  Added sample: {sample_id2}")
    assert sample_id2 is not None
    print("  [PASS]")
    
    # Test 4: Updated statistics
    print("\nTest 4: Updated statistics")
    stats = collector.get_statistics()
    print(f"  Total: {stats['total_samples']}")
    print(f"  GPT-4o: {stats['gpt4o_samples']}")
    print(f"  Human: {stats['human_samples']}")
    print(f"  Balance ratio: {stats['balance_ratio']:.2f}")
    assert stats['total_samples'] >= 2  # At least 2 samples
    assert stats['gpt4o_samples'] >= 1  # At least 1 GPT-4o
    assert stats['human_samples'] >= 1  # At least 1 human
    print("  [PASS]")
    
    # Test 5: Save dataset
    print("\nTest 5: Save dataset")
    collector.save_dataset()
    print("  Dataset saved")
    assert os.path.exists("data/test_dataset.json")
    print("  [PASS]")
    
    # Test 6: Load existing data
    print("\nTest 6: Load existing data")
    collector2 = DataCollector("data/test_dataset.json")
    stats2 = collector2.get_statistics()
    print(f"  Loaded samples: {stats2['total_samples']}")
    assert stats2['total_samples'] >= 2
    print("  [PASS]")
    
    # Test 7: Export for training
    print("\nTest 7: Export for training")
    export = collector2.export_for_training()
    total_exported = len(export['train']) + len(export['val'])
    print(f"  Train: {len(export['train'])}, Val: {len(export['val'])}")
    print(f"  Total exported: {total_exported}")
    assert total_exported >= 2
    print("  [PASS]")
    
    print("\n" + "=" * 40)
    print("*** ALL TESTS PASSED! ***")
    print("Data collector is working correctly!")
    print("\nTo start using it:")
    print("  python mining/data_collector.py")

if __name__ == "__main__":
    test_data_collector()