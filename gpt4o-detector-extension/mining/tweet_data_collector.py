#!/usr/bin/env python3
"""
TWEET DATA COLLECTOR FOR OPTIMIZED PROMPT GENERATION
===================================================

This module collects labeled AI vs human tweets, analyzes them with LLM,
and generates optimized prompts based on discovered patterns.

The workflow:
1. Collect labeled tweets (AI vs human)
2. Analyze both sets with LLM to identify distinguishing patterns
3. Generate optimized detection prompts from pattern analysis
4. Test and refine prompts for maximum accuracy
"""

import json
import os
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")

@dataclass
class TweetSample:
    """Individual tweet sample"""
    text: str
    label: str  # 'ai' or 'human'
    source: str  # Where it came from
    confidence: float  # How confident we are in the label
    metadata: Dict[str, Any]  # Additional info
    timestamp: str
    id: str

@dataclass
class PatternAnalysis:
    """Analysis result from LLM"""
    tweet_id: str
    label: str
    detected_patterns: List[str]
    confidence_markers: List[str]
    linguistic_features: Dict[str, float]
    reasoning: str
    analysis_timestamp: str

@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization"""
    optimized_prompt: str
    accuracy_score: float
    precision: float
    recall: float
    test_results: List[Dict[str, Any]]
    pattern_weights: Dict[str, float]
    generation_timestamp: str

class TweetDataCollector:
    """
    Collect and manage labeled tweet data for prompt optimization
    """
    
    def __init__(self, data_file: str = "../data/tweet_dataset.json"):
        self.data_file = data_file
        self.tweets = []
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing tweet data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tweets = [TweetSample(**tweet) for tweet in data.get('tweets', [])]
                print(f"SUCCESS: Loaded {len(self.tweets)} existing tweet samples")
            except Exception as e:
                print(f"ERROR: Error loading data: {e}")
                self.tweets = []
        else:
            print("INFO: No existing data found, starting fresh")
            self.tweets = []
    
    def save_data(self):
        """Save tweet data to file"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        data = {
            'tweets': [asdict(tweet) for tweet in self.tweets],
            'metadata': {
                'total_samples': len(self.tweets),
                'ai_samples': len([t for t in self.tweets if t.label == 'ai']),
                'human_samples': len([t for t in self.tweets if t.label == 'human']),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"SAVED: Saved {len(self.tweets)} tweet samples")
    
    def add_tweet(self, text: str, label: str, source: str = "manual", 
                  confidence: float = 1.0, metadata: Dict[str, Any] = None):
        """Add a labeled tweet sample"""
        
        if not text or len(text.strip()) < 10:
            print("ERROR: Tweet text too short")
            return False
            
        if label not in ['ai', 'human']:
            print("ERROR: Label must be 'ai' or 'human'")
            return False
            
        # Check for duplicates
        for existing in self.tweets:
            if existing.text.strip() == text.strip():
                print("WARNING: Duplicate tweet detected, skipping")
                return False
        
        tweet = TweetSample(
            text=text.strip(),
            label=label,
            source=source,
            confidence=confidence,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            id=f"tweet_{int(time.time())}_{len(self.tweets)}"
        )
        
        self.tweets.append(tweet)
        print(f"SUCCESS: Added {label} tweet (total: {len(self.tweets)})")
        return True
    
    def bulk_add_tweets(self, tweet_list: List[Dict[str, Any]]):
        """Add multiple tweets at once"""
        added = 0
        for tweet_data in tweet_list:
            if self.add_tweet(**tweet_data):
                added += 1
        
        print(f"SUCCESS: Added {added}/{len(tweet_list)} tweets")
        self.save_data()
        return added
    
    def interactive_collection(self):
        """Interactive tweet collection session"""
        print("\n" + "="*60)
        print(" INTERACTIVE TWEET COLLECTION")
        print("="*60)
        print("Instructions:")
        print("- Paste tweet text, then specify 'ai' or 'human'")
        print("- Format: <tweet text> | <ai/human> | <confidence 0-1> (optional)")
        print("- Type 'stats' to see current statistics")
        print("- Type 'save' to save progress")
        print("- Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nINFO: Enter tweet data: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'stats':
                    self.print_statistics()
                    continue
                elif user_input.lower() == 'save':
                    self.save_data()
                    continue
                elif not user_input:
                    continue
                
                # Parse input
                parts = user_input.split('|')
                if len(parts) < 2:
                    print("ERROR: Format: <tweet text> | <ai/human> | <confidence> (optional)")
                    continue
                
                text = parts[0].strip()
                label = parts[1].strip().lower()
                confidence = float(parts[2].strip()) if len(parts) > 2 else 1.0
                
                if self.add_tweet(text, label, "interactive", confidence):
                    if len(self.tweets) % 10 == 0:
                        self.save_data()  # Auto-save every 10 tweets
                
            except KeyboardInterrupt:
                print("\n\n Exiting...")
                break
            except Exception as e:
                print(f"ERROR: Error: {e}")
        
        self.save_data()
        self.print_statistics()
    
    def print_statistics(self):
        """Print current dataset statistics"""
        if not self.tweets:
            print(" No tweets collected yet")
            return
            
        ai_count = len([t for t in self.tweets if t.label == 'ai'])
        human_count = len([t for t in self.tweets if t.label == 'human'])
        
        print(f"\nDataset Statistics:")
        print(f"   Total tweets: {len(self.tweets)}")
        print(f"   AI tweets: {ai_count}")
        print(f"   Human tweets: {human_count}")
        print(f"   Balance: {ai_count/len(self.tweets)*100:.1f}% AI, {human_count/len(self.tweets)*100:.1f}% Human")
        
        if len(self.tweets) >= 20:
            print("SUCCESS: Good sample size for analysis")
        elif len(self.tweets) >= 10:
            print("WARNING: Minimum sample size reached, more data recommended")
        else:
            print("ERROR: Need more samples (minimum 10 recommended)")
    
    def get_balanced_sample(self, max_samples: int = 50) -> Tuple[List[TweetSample], List[TweetSample]]:
        """Get balanced samples of AI and human tweets"""
        ai_tweets = [t for t in self.tweets if t.label == 'ai']
        human_tweets = [t for t in self.tweets if t.label == 'human']
        
        sample_size = min(max_samples // 2, len(ai_tweets), len(human_tweets))
        
        return ai_tweets[:sample_size], human_tweets[:sample_size]
    
    def export_for_analysis(self, filename: str = None) -> str:
        """Export data in format suitable for LLM analysis"""
        if not filename:
            filename = f"../data/tweets_for_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        ai_tweets, human_tweets = self.get_balanced_sample()
        
        export_data = {
            'ai_tweets': [{'text': t.text, 'id': t.id, 'confidence': t.confidence} for t in ai_tweets],
            'human_tweets': [{'text': t.text, 'id': t.id, 'confidence': t.confidence} for t in human_tweets],
            'export_info': {
                'total_ai': len(ai_tweets),
                'total_human': len(human_tweets),
                'export_timestamp': datetime.now().isoformat(),
                'purpose': 'LLM pattern analysis for prompt optimization'
            }
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"EXPORTED: Exported {len(ai_tweets)} AI + {len(human_tweets)} human tweets to {filename}")
        return filename

class LLMPatternAnalyzer:
    """
    Analyze collected tweets with LLM to identify patterns
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=4000,
                response_mime_type="application/json"
            )
        )
        self.analysis_results = []
    
    async def analyze_tweet_patterns(self, tweets: List[TweetSample]) -> List[PatternAnalysis]:
        """Analyze individual tweets to identify patterns"""
        
        print(f" Analyzing {len(tweets)} tweets for patterns...")
        results = []
        
        for i, tweet in enumerate(tweets):
            print(f"  Analyzing tweet {i+1}/{len(tweets)}...")
            
            prompt = f"""Analyze this tweet for AI generation patterns. Return detailed JSON analysis.

TWEET: "{tweet.text}"
KNOWN LABEL: {tweet.label}

Analyze and identify specific patterns that indicate AI vs human authorship:

{{
    "tweet_id": "{tweet.id}",
    "predicted_label": "ai" or "human",
    "confidence": 0.0-1.0,
    "detected_patterns": [
        "specific patterns found (e.g., 'excessive_hedging', 'balanced_presentation')"
    ],
    "linguistic_features": {{
        "hedging_frequency": 0.0-1.0,
        "contrast_rhetoric": 0.0-1.0,
        "formal_language": 0.0-1.0,
        "meta_commentary": 0.0-1.0,
        "structured_presentation": 0.0-1.0,
        "qualifier_usage": 0.0-1.0,
        "emotional_expression": 0.0-1.0,
        "spontaneity_markers": 0.0-1.0,
        "personal_voice": 0.0-1.0,
        "natural_errors": 0.0-1.0
    }},
    "confidence_markers": [
        "specific phrases or structures that increase/decrease confidence"
    ],
    "twitter_specific_analysis": {{
        "platform_appropriateness": 0.0-1.0,
        "casual_authenticity": 0.0-1.0,
        "emoji_usage": "natural/systematic/absent",
        "abbreviation_patterns": "authentic/formal/mixed",
        "engagement_style": "natural/optimized/neutral"
    }},
    "reasoning": "detailed explanation of why this is AI or human",
    "key_discriminators": [
        "most important features that distinguish AI from human"
    ]
}}

Focus on finding reliable patterns that consistently differentiate AI from human tweets."""
            
            try:
                response = await self.model.generate_content_async(prompt)
                response_text = response.candidates[0].content.parts[0].text
                
                # Parse JSON response
                analysis_data = json.loads(response_text)
                
                result = PatternAnalysis(
                    tweet_id=tweet.id,
                    label=tweet.label,
                    detected_patterns=analysis_data.get('detected_patterns', []),
                    confidence_markers=analysis_data.get('confidence_markers', []),
                    linguistic_features=analysis_data.get('linguistic_features', {}),
                    reasoning=analysis_data.get('reasoning', ''),
                    analysis_timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"ERROR: Error analyzing tweet {tweet.id}: {e}")
                continue
        
        self.analysis_results = results
        print(f"SUCCESS: Analyzed {len(results)}/{len(tweets)} tweets")
        return results
    
    async def generate_pattern_summary(self, ai_analyses: List[PatternAnalysis], 
                                      human_analyses: List[PatternAnalysis]) -> Dict[str, Any]:
        """Generate summary of patterns found in AI vs human tweets"""
        
        prompt = f"""Analyze these tweet analysis results to identify the most reliable patterns for distinguishing AI from human tweets.

AI TWEET ANALYSES ({len(ai_analyses)} tweets):
{json.dumps([asdict(a) for a in ai_analyses], indent=2)}

HUMAN TWEET ANALYSES ({len(human_analyses)} tweets):
{json.dumps([asdict(h) for h in human_analyses], indent=2)}

Generate a comprehensive pattern summary:

{{
    "most_reliable_ai_indicators": [
        {{
            "pattern": "pattern_name",
            "description": "what this pattern looks like",
            "frequency_in_ai": 0.0-1.0,
            "frequency_in_human": 0.0-1.0,
            "reliability_score": 0.0-1.0,
            "example_phrases": ["examples from the data"]
        }}
    ],
    "most_reliable_human_indicators": [
        {{
            "pattern": "pattern_name", 
            "description": "what this pattern looks like",
            "frequency_in_human": 0.0-1.0,
            "frequency_in_ai": 0.0-1.0,
            "reliability_score": 0.0-1.0,
            "example_phrases": ["examples from the data"]
        }}
    ],
    "twitter_specific_patterns": {{
        "ai_twitter_markers": ["patterns specific to AI on Twitter"],
        "human_twitter_markers": ["patterns specific to humans on Twitter"],
        "platform_context_importance": 0.0-1.0
    }},
    "linguistic_feature_weights": {{
        "hedging_frequency": 0.0-1.0,
        "contrast_rhetoric": 0.0-1.0,
        "formal_language": 0.0-1.0,
        "meta_commentary": 0.0-1.0,
        "emotional_expression": 0.0-1.0,
        "spontaneity_markers": 0.0-1.0,
        "personal_voice": 0.0-1.0
    }},
    "detection_strategy": {{
        "primary_indicators": ["most important patterns to check first"],
        "secondary_indicators": ["supporting patterns"],
        "context_modifiers": ["patterns that depend on context"],
        "false_positive_warnings": ["patterns that might mislead"]
    }},
    "confidence_calibration": {{
        "high_confidence_patterns": ["patterns that strongly indicate AI/human"],
        "medium_confidence_patterns": ["patterns that suggest AI/human"],
        "low_confidence_patterns": ["patterns that weakly indicate AI/human"],
        "uncertainty_indicators": ["when to express uncertainty"]
    }}
}}

Focus on finding the most reliable, consistent patterns that work specifically for Twitter/X content."""
        
        response = await self.model.generate_content_async(prompt)
        response_text = response.candidates[0].content.parts[0].text
        
        return json.loads(response_text)

class PromptOptimizer:
    """
    Generate optimized prompts based on pattern analysis
    """
    
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    async def generate_optimized_prompt(self, pattern_summary: Dict[str, Any]) -> str:
        """Generate an optimized detection prompt based on discovered patterns"""
        
        prompt = f"""Based on this comprehensive pattern analysis of AI vs human tweets, create the most effective prompt for detecting AI-generated tweets.

PATTERN ANALYSIS DATA:
{json.dumps(pattern_summary, indent=2)}

Create an optimized prompt that:
1. Focuses on the most reliable discriminating patterns
2. Is specifically tuned for Twitter/X content 
3. Provides structured JSON output with confidence scores
4. Minimizes false positives and false negatives
5. Works efficiently in real-time detection

Return the optimized prompt as a string that can be used directly with an LLM for tweet analysis. The prompt should be concise but comprehensive, focusing on the proven patterns from your analysis.

The prompt should instruct the LLM to analyze a tweet and return JSON with:
- ai_probability (0.0-1.0)
- prediction ("ai" or "human") 
- confidence level
- key_indicators found
- reasoning

Make it as accurate as possible based on the patterns you discovered."""
        
        response = await self.model.generate_content_async(prompt)
        return response.candidates[0].content.parts[0].text
    
    async def test_prompt_accuracy(self, optimized_prompt: str, test_tweets: List[TweetSample]) -> PromptOptimizationResult:
        """Test the optimized prompt against known labeled data"""
        
        print(f"TESTING: Testing optimized prompt against {len(test_tweets)} tweets...")
        
        correct_predictions = 0
        test_results = []
        
        for tweet in test_tweets:
            full_prompt = f"{optimized_prompt}\n\nTWEET: \"{tweet.text}\""
            
            try:
                response = await self.model.generate_content_async(full_prompt)
                response_text = response.candidates[0].content.parts[0].text
                
                # Parse response 
                try:
                    result = json.loads(response_text)
                    predicted = result.get('prediction', 'unknown')
                    ai_prob = result.get('ai_probability', 0.5)
                    
                    is_correct = predicted == tweet.label
                    if is_correct:
                        correct_predictions += 1
                    
                    test_results.append({
                        'tweet_id': tweet.id,
                        'actual_label': tweet.label,
                        'predicted_label': predicted,
                        'ai_probability': ai_prob,
                        'correct': is_correct,
                        'confidence': result.get('confidence', {}),
                        'key_indicators': result.get('key_indicators', [])
                    })
                    
                except json.JSONDecodeError:
                    test_results.append({
                        'tweet_id': tweet.id,
                        'actual_label': tweet.label,
                        'predicted_label': 'error',
                        'correct': False,
                        'error': 'JSON parse failed'
                    })
                
            except Exception as e:
                test_results.append({
                    'tweet_id': tweet.id,
                    'actual_label': tweet.label,
                    'predicted_label': 'error',
                    'correct': False,
                    'error': str(e)
                })
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_tweets)
        
        # Calculate precision and recall
        true_positives = len([r for r in test_results if r.get('actual_label') == 'ai' and r.get('predicted_label') == 'ai'])
        false_positives = len([r for r in test_results if r.get('actual_label') == 'human' and r.get('predicted_label') == 'ai'])  
        false_negatives = len([r for r in test_results if r.get('actual_label') == 'ai' and r.get('predicted_label') == 'human'])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"SUCCESS: Prompt testing complete:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall: {recall:.1%}")
        
        return PromptOptimizationResult(
            optimized_prompt=optimized_prompt,
            accuracy_score=accuracy,
            precision=precision,
            recall=recall,
            test_results=test_results,
            pattern_weights={},  # Could extract from pattern analysis
            generation_timestamp=datetime.now().isoformat()
        )

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

async def main_optimization_workflow():
    """
    Complete workflow: collect → analyze → optimize → test
    """
    print(" TWEET PATTERN ANALYSIS & PROMPT OPTIMIZATION")
    print("="*70)
    
    # Step 1: Data Collection
    print("\n1. DATA COLLECTION")
    collector = TweetDataCollector()
    
    if len(collector.tweets) < 20:
        print("Need more tweet samples for analysis")
        print("Choose an option:")
        print("  1. Interactive collection session")
        print("  2. Load sample data")
        print("  3. Skip to demo with existing data")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            collector.interactive_collection()
        elif choice == "2":
            # Add some sample data for demo
            sample_tweets = [
                {"text": "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity across various sectors. On the other hand, concerns about job displacement and ethical implications require careful consideration.", "label": "ai", "source": "sample"},
                {"text": "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore. feels like we're heading straight for skynet territory lol 😅", "label": "human", "source": "sample"},
                {"text": "It's worth considering that machine learning algorithms, while powerful, have certain limitations that we should carefully evaluate before implementation.", "label": "ai", "source": "sample"},
                {"text": "just tried the new chatgpt update and wow it's actually pretty good now. still makes mistakes but way better than before", "label": "human", "source": "sample"},
                {"text": "The intersection of artificial intelligence and healthcare presents both promising opportunities and significant challenges that merit thorough examination.", "label": "ai", "source": "sample"},
                {"text": "why does every ai company think they need to release their model with some dramatic name like 'titan' or 'apex' lmao just call it what it is", "label": "human", "source": "sample"}
            ]
            collector.bulk_add_tweets(sample_tweets)
    
    if len(collector.tweets) < 10:
        print("ERROR: Need at least 10 tweets for analysis. Run interactive collection first.")
        return
    
    # Step 2: Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: Please set GEMINI_API_KEY environment variable")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        return
    
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
    
    print(f"\n FINAL RESULTS:")
    print(f"    Accuracy: {test_result.accuracy_score:.1%}")
    print(f"    Precision: {test_result.precision:.1%}")  
    print(f"    Recall: {test_result.recall:.1%}")
    print(f"   SAVED: Saved to: ../data/optimized_prompt.json")
    
    print(f"\nSUCCESS: Optimization complete! Use the optimized prompt in your Chrome extension.")

if __name__ == "__main__":
    if not GEMINI_AVAILABLE:
        print("ERROR: Please install required dependencies:")
        print("   pip install google-generativeai")
    else:
        # Run the workflow
        collector = TweetDataCollector()
        if len(collector.tweets) == 0:
            print("Starting interactive data collection...")
            collector.interactive_collection()
        else:
            print("Data exists. Running full optimization workflow...")
            asyncio.run(main_optimization_workflow())