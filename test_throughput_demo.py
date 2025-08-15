"""
Quick demonstration of high-throughput processing achieving >1000 tweets/min.
"""

import asyncio
import time
from src.data.processing.throughput_optimizer import create_throughput_optimizer

def generate_sample_tweets(count: int):
    """Generate sample tweets for testing."""
    tweets = []
    for i in range(count):
        tweets.append({
            'id': f'demo_tweet_{i}',
            'text': f'This is demo tweet number {i}. It contains enough text to be processed effectively by the AI detection system. The content includes various patterns that can be analyzed for classification purposes.',
            'timestamp': time.time(),
            'user_id': f'user_{i % 50}'
        })
    return tweets

async def test_throughput_optimization():
    """Test throughput optimization with sample data."""
    print("🚀 Starting High-Throughput Processing Demo")
    print("=" * 50)
    
    # Create optimizer
    optimizer = create_throughput_optimizer(target_tpm=1000, enable_gpu=False)
    
    # Generate test data
    test_sizes = [500, 1000, 1500]
    
    for test_size in test_sizes:
        print(f"\n📊 Testing with {test_size} tweets...")
        
        tweets = generate_sample_tweets(test_size)
        
        # Process with optimization
        start_time = time.time()
        results = await optimizer.optimize_throughput(tweets)
        duration = time.time() - start_time
        
        # Calculate metrics
        throughput = (len(tweets) / duration) * 60
        target_achieved = throughput >= 1000
        
        print(f"   📈 Processed: {len(results)} tweets")
        print(f"   ⏱️  Duration: {duration:.3f} seconds")
        print(f"   🏃 Throughput: {throughput:.0f} tweets/min")
        print(f"   🎯 Target (1000 tpm): {'✅ ACHIEVED' if target_achieved else '❌ MISSED'}")
        
        if target_achieved:
            print(f"   💪 Exceeds target by: {throughput - 1000:.0f} tweets/min")
    
    # Get optimization report
    print(f"\n📋 Optimization Report:")
    print("=" * 30)
    
    report = optimizer.get_optimization_report()
    if 'error' not in report:
        perf = report['current_performance']
        print(f"   Average Throughput: {perf['avg_throughput']:.0f} tpm")
        print(f"   Max Throughput: {perf['max_throughput']:.0f} tpm")
        print(f"   Target Achievement Rate: {perf['target_achievement_rate']:.1%}")
        
        print(f"\n🔧 Optimization Strategy:")
        strategy = report['optimization_strategy']
        for key, value in strategy.items():
            print(f"   {key}: {value}")
    
    # Cleanup
    optimizer.cleanup()
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_throughput_optimization())