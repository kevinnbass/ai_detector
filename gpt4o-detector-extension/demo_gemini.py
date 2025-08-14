#!/usr/bin/env python3
"""
GEMINI STRUCTURED AI DETECTOR - DEMO SCRIPT
==========================================

Interactive demo showcasing the comprehensive Gemini-powered AI detection system
with structured JSON output and quantified analysis across 10 dimensions.

Usage:
    python demo_gemini.py

Requirements:
    pip install google-generativeai numpy dataclasses-json
    
API Key Setup:
    set GEMINI_API_KEY=your-api-key-here  (Windows)
    export GEMINI_API_KEY=your-api-key-here  (Linux/Mac)
    
Get your API key: https://makersuite.google.com/app/apikey
"""

import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

# Add mining directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mining'))

try:
    from gemini_structured_analyzer import GeminiStructuredAnalyzer, GEMINI_AVAILABLE
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're in the correct directory and have installed requirements:")
    print("  pip install google-generativeai numpy dataclasses-json")
    sys.exit(1)

def print_banner():
    """Print demo banner"""
    print("🧠" + "=" * 68 + "🧠")
    print("    GEMINI STRUCTURED AI DETECTOR - INTERACTIVE DEMO")
    print("         Comprehensive Analysis with Quantified Scores")
    print("🧠" + "=" * 68 + "🧠")

def check_api_key():
    """Check if API key is available"""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not found!")
        print("\n🔑 To get your API key:")
        print("   1. Go to: https://makersuite.google.com/app/apikey")
        print("   2. Click 'Create API Key'")
        print("   3. Copy your key")
        print("\n💻 To set your API key:")
        print("   Windows CMD:    set GEMINI_API_KEY=your-key-here")
        print("   Windows PS:     $env:GEMINI_API_KEY=\"your-key-here\"")
        print("   Linux/Mac:      export GEMINI_API_KEY=\"your-key-here\"")
        print("\n📄 Or create .env file with: GEMINI_API_KEY=your-key-here")
        return None
    
    # Mask API key for display
    masked_key = api_key[:8] + "..." + api_key[-8:] if len(api_key) > 16 else "***"
    print(f"✅ API Key found: {masked_key}")
    return api_key

def get_demo_texts():
    """Get demo texts for analysis"""
    return {
        "gpt4o_sample": {
            "text": "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity across various sectors. On the other hand, concerns about job displacement and ethical implications require careful consideration.",
            "expected": "ai",
            "description": "Classic GPT-4o pattern with hedging and balanced presentation"
        },
        "human_casual": {
            "text": "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore. feels like we're heading straight for skynet territory lol 😅",
            "expected": "human", 
            "description": "Casual human writing with natural errors and emotions"
        },
        "human_technical": {
            "text": "Been debugging this React component for 3 hours now. The useEffect hook keeps triggering infinite re-renders because I forgot to add the dependency array. Classic mistake that still gets me every time. Coffee definitely needed.",
            "expected": "human",
            "description": "Technical human writing with personal experience"
        },
        "gpt4o_academic": {
            "text": "The integration of artificial intelligence into educational frameworks presents a compelling dichotomy. While proponents argue for enhanced personalized learning experiences, critics raise legitimate concerns regarding academic integrity and the potential erosion of critical thinking skills. It is essential to carefully balance these considerations moving forward.",
            "expected": "ai",
            "description": "Academic-style GPT-4o text with formal hedging"
        }
    }

def print_analysis_summary(result, expected=None):
    """Print formatted analysis summary"""
    
    print(f"\n🎯 **PREDICTION RESULTS**")
    print(f"   Prediction: {result.prediction.upper()} ({result.ai_probability:.1%})")
    print(f"   Confidence: {result.overall_confidence.certainty.upper()} ({result.overall_confidence.value:.1%})")
    print(f"   Reliability: {result.overall_confidence.reliability:.1%}")
    
    if expected:
        correct = "✅ CORRECT" if result.prediction == expected else "❌ INCORRECT"
        print(f"   Expected: {expected.upper()} - {correct}")
    
    print(f"\n📊 **DIMENSION SCORES** (0.0=Human, 1.0=AI)")
    
    # Extract dimension scores
    dimensions = [
        ("Cognitive Load", result.cognitive_load.overall_load.score),
        ("Emotional Intel", result.emotional_intelligence.overall_eq.score), 
        ("Creativity", result.creativity.overall_creativity.score),
        ("Linguistic", result.linguistic.overall_linguistic.score),
        ("Domain Expertise", result.domain_expertise.overall_expertise.score),
        ("Personality", result.personality.overall_personality.score),
        ("Temporal", result.temporal.overall_temporal.score),
        ("Cultural", result.cultural.overall_cultural.score),
        ("Deception", result.deception.overall_deception.score),
        ("Metacognitive", result.metacognitive.overall_metacognitive.score)
    ]
    
    for name, score in dimensions:
        bar_length = 20
        filled = int(score * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   {name:15} [{bar}] {score:.2f}")
    
    print(f"\n🔍 **KEY INSIGHTS**")
    
    # Top AI indicators
    if result.ai_markers:
        print(f"   AI Markers:")
        for marker in result.ai_markers[:3]:
            print(f"     • {marker}")
    
    # Top human indicators  
    if result.human_markers:
        print(f"   Human Markers:")
        for marker in result.human_markers[:3]:
            print(f"     • {marker}")
    
    # Contradictions
    if result.contradiction_indicators:
        print(f"   ⚠️  Contradictions:")
        for contradiction in result.contradiction_indicators[:2]:
            print(f"     • {contradiction}")
    
    print(f"\n💡 **RECOMMENDATION**")
    print(f"   {result.recommendation}")
    
    print(f"\n📈 **ENSEMBLE METRICS**")
    print(f"   Agreement: {result.ensemble_agreement:.1%}")
    print(f"   Stability: {result.prediction_stability:.1%}") 
    print(f"   Processing Time: {result.processing_time:.2f}s")

async def run_demo():
    """Run the interactive demo"""
    
    print_banner()
    
    # Check prerequisites
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai not installed!")
        print("   Install with: pip install google-generativeai")
        return
    
    api_key = check_api_key()
    if not api_key:
        return
    
    print(f"🔧 Initializing Gemini analyzer...")
    
    try:
        analyzer = GeminiStructuredAnalyzer(api_key, model_name="gemini-1.5-flash")
        print(f"✅ Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    demo_texts = get_demo_texts()
    
    print(f"\n🎮 **INTERACTIVE DEMO**")
    print(f"   Available samples:")
    for i, (key, data) in enumerate(demo_texts.items(), 1):
        print(f"   {i}. {key.replace('_', ' ').title()}: {data['description']}")
    print(f"   5. Enter custom text")
    print(f"   6. Run all samples")
    print(f"   0. Exit")
    
    results = {}
    
    while True:
        print(f"\n" + "-" * 70)
        choice = input("Select option (0-6): ").strip()
        
        if choice == "0":
            print("👋 Exiting demo. Thanks for trying the Gemini AI Detector!")
            break
            
        elif choice == "6":
            print(f"\n🚀 **RUNNING ALL SAMPLES**")
            
            for i, (key, data) in enumerate(demo_texts.items(), 1):
                print(f"\n" + "=" * 70)
                print(f"📝 SAMPLE {i}: {key.replace('_', ' ').title()}")
                print(f"Text: \"{data['text'][:100]}...\"")
                print(f"Expected: {data['expected'].upper()}")
                print(f"-" * 70)
                
                try:
                    result = await analyzer.comprehensive_analysis(data['text'])
                    results[key] = result
                    print_analysis_summary(result, data['expected'])
                    
                except Exception as e:
                    print(f"❌ Analysis failed: {e}")
                
                # Rate limiting
                if i < len(demo_texts):
                    print(f"\n⏳ Rate limiting... (2s)")
                    await asyncio.sleep(2)
            
            # Summary
            if results:
                correct = sum(1 for key, result in results.items() 
                            if result.prediction == demo_texts[key]['expected'])
                total = len(results)
                accuracy = correct / total * 100
                
                print(f"\n" + "=" * 70)
                print(f"🎯 **BATCH ANALYSIS SUMMARY**")
                print(f"   Samples analyzed: {total}")
                print(f"   Correct predictions: {correct}/{total}")
                print(f"   Accuracy: {accuracy:.1f}%")
                print(f"   Average processing time: {sum(r.processing_time for r in results.values()) / len(results):.2f}s")
        
        elif choice in ["1", "2", "3", "4"]:
            idx = int(choice) - 1
            key = list(demo_texts.keys())[idx]
            data = demo_texts[key]
            
            print(f"\n📝 **ANALYZING: {key.replace('_', ' ').title()}**")
            print(f"Text: \"{data['text']}\"")
            print(f"Expected: {data['expected'].upper()}")
            print(f"\n🧠 Running comprehensive analysis...")
            
            try:
                result = await analyzer.comprehensive_analysis(data['text'])
                results[key] = result
                print_analysis_summary(result, data['expected'])
                
                # Offer to save JSON
                save = input(f"\n💾 Save detailed JSON result? (y/n): ").strip().lower()
                if save == 'y':
                    os.makedirs("results", exist_ok=True)
                    filename = f"results/{key}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(asdict(result), f, indent=2)
                    
                    print(f"✅ Saved to {filename}")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == "5":
            print(f"\n✍️  **CUSTOM TEXT ANALYSIS**")
            custom_text = input("Enter your text to analyze: ").strip()
            
            if not custom_text:
                print("❌ No text entered")
                continue
                
            if len(custom_text) < 10:
                print("⚠️  Text is very short, results may be unreliable")
            
            print(f"\n🧠 Analyzing custom text...")
            
            try:
                result = await analyzer.comprehensive_analysis(custom_text)
                print_analysis_summary(result)
                
                # Offer to save
                save = input(f"\n💾 Save result? (y/n): ").strip().lower()
                if save == 'y':
                    os.makedirs("results", exist_ok=True)
                    filename = f"results/custom_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(asdict(result), f, indent=2)
                    
                    print(f"✅ Saved to {filename}")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
        
        else:
            print("❌ Invalid choice. Please select 0-6.")

def main():
    """Main entry point"""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()