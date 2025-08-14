#!/usr/bin/env python3
"""
Run the complete tweet pattern analysis and prompt optimization workflow
"""

import os
import asyncio
from tweet_data_collector import TweetDataCollector, LLMPatternAnalyzer, PromptOptimizer

async def run_complete_workflow():
    """Run the complete workflow with sample data"""
    
    print("TWEET PATTERN ANALYSIS & PROMPT OPTIMIZATION")
    print("="*70)
    
    # Step 1: Create collector and add sample data
    print("\n1. DATA COLLECTION")
    collector = TweetDataCollector()
    
    # Add comprehensive sample data
    sample_tweets = [
        # AI-generated samples (characteristic patterns)
        {"text": "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity across various sectors. On the other hand, concerns about job displacement and ethical implications require careful consideration.", "label": "ai", "source": "sample"},
        {"text": "It's worth considering that machine learning algorithms, while powerful, have certain limitations that we should carefully evaluate before implementation. Furthermore, the ethical implications merit thorough examination.", "label": "ai", "source": "sample"},
        {"text": "The intersection of artificial intelligence and healthcare presents both promising opportunities and significant challenges that merit thorough examination. It's important to note that while AI can enhance diagnostic accuracy, we must also consider the potential risks and limitations.", "label": "ai", "source": "sample"},
        {"text": "In today's rapidly evolving technological landscape, it's crucial to understand that blockchain technology offers both significant advantages and notable challenges. On one hand, decentralization provides enhanced security. On the other hand, scalability concerns require careful consideration.", "label": "ai", "source": "sample"},
        {"text": "When examining the current state of renewable energy, it's important to recognize that while solar and wind technologies have made substantial progress, there are still certain limitations that we should carefully evaluate before widespread implementation.", "label": "ai", "source": "sample"},
        {"text": "It's worth noting that remote work policies present both opportunities and challenges for modern organizations. While flexibility can enhance employee satisfaction, it's important to consider potential impacts on collaboration and company culture.", "label": "ai", "source": "sample"},
        {"text": "The field of quantum computing, while promising, faces several significant hurdles that merit careful consideration. It's important to understand that while quantum supremacy offers theoretical advantages, practical implementation remains challenging.", "label": "ai", "source": "sample"},
        {"text": "In considering the future of autonomous vehicles, we must acknowledge both the potential benefits and inherent risks. While self-driving technology promises enhanced safety, it's crucial to examine the ethical and regulatory challenges that lie ahead.", "label": "ai", "source": "sample"},
        
        # Human-generated samples (natural, casual, personal)
        {"text": "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore. feels like we're heading straight for skynet territory lol 😅", "label": "human", "source": "sample"},
        {"text": "just tried the new chatgpt update and wow it's actually pretty good now. still makes mistakes but way better than before", "label": "human", "source": "sample"},
        {"text": "why does every ai company think they need to release their model with some dramatic name like 'titan' or 'apex' lmao just call it what it is", "label": "human", "source": "sample"},
        {"text": "been using claude for coding help and honestly it's insane how much faster i can debug now. still double-check everything but damn", "label": "human", "source": "sample"},
        {"text": "my manager just said we need to 'leverage AI synergies for optimal workflow optimization' and I literally cannot even", "label": "human", "source": "sample"},
        {"text": "twitter's algorithm is so broken rn. showing me the same 3 accounts over and over while hiding actually interesting stuff", "label": "human", "source": "sample"},
        {"text": "unpopular opinion but I think most AI hype is just marketing bs. like yeah it's cool but we're not getting AGI next year calm down", "label": "human", "source": "sample"},
        {"text": "spent 2 hours trying to get this neural network to work and turns out i had a typo in one variable name. coding is pain", "label": "human", "source": "sample"},
    ]
    
    # Add all sample tweets
    collector.bulk_add_tweets(sample_tweets)
    
    if len(collector.tweets) < 10:
        print("ERROR: Need at least 10 tweets for analysis.")
        return
    
    collector.print_statistics()
    
    # Step 2: Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\nERROR: Please set GEMINI_API_KEY environment variable")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        print("Example: set GEMINI_API_KEY=your_key_here")
        return
    
    try:
        # Step 3: Pattern Analysis
        print("\n2. PATTERN ANALYSIS")
        analyzer = LLMPatternAnalyzer(api_key)
        
        ai_tweets = [t for t in collector.tweets if t.label == 'ai']
        human_tweets = [t for t in collector.tweets if t.label == 'human']
        
        print(f"Analyzing {len(ai_tweets)} AI tweets and {len(human_tweets)} human tweets...")
        
        ai_analyses = await analyzer.analyze_tweet_patterns(ai_tweets)
        human_analyses = await analyzer.analyze_tweet_patterns(human_tweets)
        
        # Step 4: Generate Pattern Summary
        print("\n3. PATTERN SUMMARY GENERATION")
        pattern_summary = await analyzer.generate_pattern_summary(ai_analyses, human_analyses)
        
        # Save pattern analysis
        os.makedirs("../data", exist_ok=True)
        with open("../data/pattern_analysis.json", 'w') as f:
            import json
            json.dump(pattern_summary, f, indent=2)
        print("SAVED: Pattern analysis saved to ../data/pattern_analysis.json")
        
        # Step 5: Generate Optimized Prompt
        print("\n4. PROMPT OPTIMIZATION")
        optimizer = PromptOptimizer(api_key)
        optimized_prompt = await optimizer.generate_optimized_prompt(pattern_summary)
        
        print("SUCCESS: Generated optimized prompt:")
        print("-" * 50)
        print(optimized_prompt)
        print("-" * 50)
        
        # Step 6: Test Prompt
        print("\n5. PROMPT TESTING")
        test_result = await optimizer.test_prompt_accuracy(optimized_prompt, collector.tweets)
        
        # Save optimized prompt and results
        from dataclasses import asdict
        final_result = {
            'optimized_prompt': optimized_prompt,
            'test_results': asdict(test_result),
            'pattern_summary': pattern_summary,
            'generation_info': {
                'total_tweets_analyzed': len(collector.tweets),
                'ai_tweets': len(ai_tweets),
                'human_tweets': len(human_tweets),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open("../data/optimized_prompt.json", 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\nFINAL RESULTS:")
        print(f"   Accuracy: {test_result.accuracy_score:.1%}")
        print(f"   Precision: {test_result.precision:.1%}")  
        print(f"   Recall: {test_result.recall:.1%}")
        print(f"   Saved to: ../data/optimized_prompt.json")
        
        print(f"\nSUCCESS: Optimization complete! Use the optimized prompt in your Chrome extension.")
        
        return optimized_prompt, test_result.accuracy_score
        
    except Exception as e:
        print(f"ERROR: Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

if __name__ == "__main__":
    try:
        import google.generativeai as genai
        from datetime import datetime
        import json
        
        print("INFO: Starting optimization workflow...")
        result = asyncio.run(run_complete_workflow())
        
        if result[0]:  # optimized_prompt exists
            print(f"\nWorkflow completed successfully with {result[1]:.1%} accuracy!")
        else:
            print("\nWorkflow failed. Please check the error messages above.")
            
    except ImportError:
        print("ERROR: Please install required dependencies:")
        print("   pip install google-generativeai")