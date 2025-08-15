"""
Simple validation of throughput processing components.
"""

import time
import asyncio
from src.data.processing.high_throughput_processor import HighThroughputProcessor, ProcessingConfig

def test_basic_throughput():
    """Test basic throughput processing."""
    print("Testing High-Throughput Processing...")
    
    # Create processor
    config = ProcessingConfig(
        max_workers=4,
        batch_size=50,
        vectorization_enabled=True,
        stream_processing=False,  # Simplified for testing
        processing_timeout=10.0
    )
    
    processor = HighThroughputProcessor(config)
    
    # Generate simple test data
    tweets = []
    for i in range(100):
        tweets.append({
            'id': f'test_{i}',
            'text': f'Test tweet {i} with sample content for processing.'
        })
    
    # Process synchronously for simplicity
    start_time = time.time()
    results = processor.process_tweets_sync(tweets)
    duration = time.time() - start_time
    
    # Calculate throughput
    throughput = (len(tweets) / duration) * 60
    
    print(f"Processed: {len(results)} tweets")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Throughput: {throughput:.0f} tweets/min")
    
    # Validate results
    assert len(results) == len(tweets), "Result count mismatch"
    assert all('is_ai_generated' in result for result in results), "Missing classification"
    
    # Get processing stats
    stats = processor.get_processing_stats()
    print(f"Stats: {stats.tweets_processed} processed, {stats.tweets_per_minute:.0f} tpm")
    
    # Get throughput report
    report = processor.get_throughput_report()
    current_throughput = report['current_throughput']['tweets_per_minute']
    target_achieved = report['current_throughput']['target_achieved']
    
    print(f"Report throughput: {current_throughput:.0f} tpm")
    print(f"Target achieved: {target_achieved}")
    
    processor.cleanup()
    print("✅ Basic throughput test completed!")
    
    return throughput >= 1000

if __name__ == "__main__":
    success = test_basic_throughput()
    if success:
        print("🎯 TARGET ACHIEVED: >1000 tweets/min")
    else:
        print("⚠️  Target not achieved in basic test, but optimizations available")