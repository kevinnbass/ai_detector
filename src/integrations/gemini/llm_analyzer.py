import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import os
from data_collector import DataCollector

class LLMAnalyzer:
    """
    Advanced LLM-powered analysis using SOTA models like Gemini 2.5 Flash
    """
    
    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-username/gpt4o-detector",
            "X-Title": "GPT-4o Detector"
        }
    
    def analyze_text_patterns(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze text using LLM for deep pattern recognition
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis ('comprehensive', 'stylistic', 'rhetorical', 'comparative')
        """
        
        prompts = {
            "comprehensive": self._get_comprehensive_prompt(text),
            "stylistic": self._get_stylistic_prompt(text),
            "rhetorical": self._get_rhetorical_prompt(text),
            "comparative": self._get_comparative_prompt(text),
            "semantic": self._get_semantic_prompt(text)
        }
        
        if analysis_type not in prompts:
            analysis_type = "comprehensive"
        
        try:
            response = self._call_llm(prompts[analysis_type])
            
            # Parse structured response
            analysis = self._parse_llm_response(response, analysis_type)
            
            return {
                'analysis_type': analysis_type,
                'model': self.model,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'analysis': analysis,
                'raw_response': response
            }
            
        except Exception as e:
            return {
                'analysis_type': analysis_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_comprehensive_prompt(self, text: str) -> str:
        return f"""Analyze this text for AI generation patterns, specifically GPT-4o characteristics. Be extremely detailed and precise.

TEXT: "{text}"

Provide analysis in this EXACT JSON format:
{{
    "gpt4o_probability": 0.XX,
    "confidence": 0.XX,
    "detected_patterns": [
        {{
            "pattern": "pattern_name",
            "description": "what you detected",
            "evidence": "specific text evidence",
            "strength": 0.XX
        }}
    ],
    "linguistic_features": {{
        "hedging_frequency": "analysis",
        "contrast_rhetoric": "analysis", 
        "formal_register": "analysis",
        "structured_presentation": "analysis",
        "qualification_patterns": "analysis"
    }},
    "human_indicators": [
        "any patterns suggesting human authorship"
    ],
    "ai_indicators": [
        "any patterns suggesting AI authorship"
    ],
    "reasoning": "detailed explanation of classification",
    "key_phrases": [
        "phrases most indicative of AI/human"
    ]
}}

Focus specifically on GPT-4o patterns like:
- Excessive hedging ("perhaps", "might", "could", "seems")
- Contrast constructions ("not X, but Y", "while X, however Y")
- Qualifier phrases ("it's important to note", "keep in mind")
- Balanced presentations (always showing pros/cons)
- Formal language in casual contexts
- Structured enumeration ("firstly", "secondly")

RESPOND ONLY WITH VALID JSON."""
    
    def _get_stylistic_prompt(self, text: str) -> str:
        return f"""Analyze the writing style of this text for AI detection. Focus on stylistic patterns unique to GPT-4o.

TEXT: "{text}"

Return JSON analysis:
{{
    "style_score": {{
        "formality": 0.XX,
        "consistency": 0.XX,
        "predictability": 0.XX,
        "authenticity": 0.XX
    }},
    "vocabulary_analysis": {{
        "complexity": "assessment",
        "variety": "assessment", 
        "register_appropriateness": "assessment"
    }},
    "sentence_patterns": {{
        "length_consistency": "analysis",
        "structure_variety": "analysis",
        "flow_naturalness": "analysis"
    }},
    "emotional_markers": {{
        "authenticity": "analysis",
        "spontaneity": "analysis",
        "personal_voice": "analysis"
    }},
    "ai_style_indicators": [
        "specific stylistic markers suggesting AI"
    ],
    "human_style_indicators": [
        "specific stylistic markers suggesting human"
    ]
}}"""
    
    def _get_rhetorical_prompt(self, text: str) -> str:
        return f"""Analyze the rhetorical patterns in this text for AI detection. Focus on argumentative and persuasive structures.

TEXT: "{text}"

Return JSON analysis:
{{
    "rhetorical_patterns": {{
        "argument_structure": "analysis",
        "evidence_presentation": "analysis",
        "logical_flow": "analysis",
        "persuasive_techniques": "analysis"
    }},
    "gpt4o_rhetorical_markers": [
        "specific rhetorical patterns typical of GPT-4o"
    ],
    "balance_analysis": {{
        "pros_cons_presentation": "analysis",
        "perspective_neutrality": "analysis",
        "opinion_hedging": "analysis"
    }},
    "discourse_markers": {{
        "transition_usage": "analysis",
        "logical_connectors": "analysis",
        "meta_commentary": "analysis"
    }}
}}"""
    
    def _get_comparative_prompt(self, text: str) -> str:
        return f"""Compare this text against typical GPT-4o vs human writing patterns. Be specific about differences.

TEXT: "{text}"

Analyze and return:
{{
    "comparison_analysis": {{
        "vs_gpt4o": {{
            "similarity_score": 0.XX,
            "matching_patterns": ["list patterns that match GPT-4o"],
            "differences": ["how it differs from typical GPT-4o"]
        }},
        "vs_human": {{
            "similarity_score": 0.XX,
            "matching_patterns": ["list patterns that match human writing"],
            "differences": ["how it differs from typical human writing"]
        }}
    }},
    "distinctive_features": [
        "features that strongly suggest one or the other"
    ],
    "ambiguous_elements": [
        "elements that could go either way"
    ],
    "final_classification": {{
        "prediction": "gpt4o or human",
        "confidence": 0.XX,
        "primary_evidence": "main reason for classification"
    }}
}}"""
    
    def _get_semantic_prompt(self, text: str) -> str:
        return f"""Analyze the semantic patterns and conceptual structures in this text for AI detection.

TEXT: "{text}"

Return semantic analysis:
{{
    "conceptual_patterns": {{
        "abstraction_level": "analysis",
        "concept_relationships": "analysis", 
        "semantic_coherence": "analysis"
    }},
    "knowledge_representation": {{
        "encyclopedic_vs_experiential": "analysis",
        "factual_vs_personal": "analysis",
        "generic_vs_specific": "analysis"
    }},
    "semantic_markers": {{
        "ai_knowledge_patterns": ["patterns suggesting AI knowledge base"],
        "human_experience_patterns": ["patterns suggesting human experience"],
        "conceptual_gaps": ["areas where understanding seems shallow/deep"]
    }},
    "topic_treatment": {{
        "depth": "analysis",
        "perspective": "analysis",
        "authenticity": "analysis"
    }}
}}"""
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to OpenRouter"""
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent analysis
            "max_tokens": 2000,
            "top_p": 0.9
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if 'choices' not in result or not result['choices']:
            raise Exception(f"Unexpected response format: {result}")
        
        return result['choices'][0]['message']['content'].strip()
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Try to parse as JSON
            if response.startswith('```json'):
                response = response.split('```json')[1].split('```')[0]
            elif response.startswith('```'):
                response = response.split('```')[1].split('```')[0]
            
            return json.loads(response)
            
        except json.JSONDecodeError:
            # Fallback: extract key information using regex
            return {
                'parsing_error': True,
                'raw_text': response,
                'extracted_probability': self._extract_probability(response),
                'extracted_patterns': self._extract_patterns(response)
            }
    
    def _extract_probability(self, text: str) -> Optional[float]:
        """Extract probability from text response"""
        import re
        
        # Look for probability patterns
        patterns = [
            r'probability["\']?\s*:\s*([0-9.]+)',
            r'([0-9.]+)\s*probability',
            r'confidence["\']?\s*:\s*([0-9.]+)',
            r'([0-9]+)%\s*confident'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    return value if value <= 1.0 else value / 100.0
                except ValueError:
                    continue
        
        return None
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract detected patterns from text response"""
        patterns = []
        
        # Common pattern indicators
        indicators = [
            'hedging', 'contrast', 'formal', 'qualifier', 'balanced',
            'structured', 'enumeration', 'systematic', 'consistent'
        ]
        
        for indicator in indicators:
            if indicator in text.lower():
                patterns.append(indicator)
        
        return patterns
    
    def batch_analyze(self, texts: List[Dict[str, Any]], 
                     analysis_types: List[str] = None,
                     rate_limit: float = 1.0) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts with rate limiting
        
        Args:
            texts: List of text dictionaries with 'text' and 'label' keys
            analysis_types: Types of analysis to perform
            rate_limit: Seconds to wait between API calls
        """
        
        if analysis_types is None:
            analysis_types = ['comprehensive']
        
        results = []
        
        for i, text_data in enumerate(texts):
            print(f"Analyzing text {i+1}/{len(texts)}...")
            
            text_results = {
                'original_text': text_data['text'],
                'true_label': text_data.get('label'),
                'analyses': {}
            }
            
            for analysis_type in analysis_types:
                try:
                    analysis = self.analyze_text_patterns(
                        text_data['text'], 
                        analysis_type
                    )
                    text_results['analyses'][analysis_type] = analysis
                    
                    # Rate limiting
                    time.sleep(rate_limit)
                    
                except Exception as e:
                    print(f"Error analyzing text {i+1} with {analysis_type}: {e}")
                    text_results['analyses'][analysis_type] = {'error': str(e)}
            
            results.append(text_results)
        
        return results
    
    def create_ensemble_prediction(self, text: str) -> Dict[str, Any]:
        """
        Run multiple analysis types and create ensemble prediction
        """
        
        analysis_types = ['comprehensive', 'stylistic', 'rhetorical', 'semantic']
        
        analyses = {}
        probabilities = []
        all_patterns = []
        
        for analysis_type in analysis_types:
            try:
                result = self.analyze_text_patterns(text, analysis_type)
                analyses[analysis_type] = result
                
                # Extract probability if available
                if 'analysis' in result and isinstance(result['analysis'], dict):
                    prob = result['analysis'].get('gpt4o_probability')
                    if prob is not None:
                        probabilities.append(float(prob))
                
                # Extract patterns
                if 'analysis' in result and 'detected_patterns' in result['analysis']:
                    patterns = result['analysis']['detected_patterns']
                    if isinstance(patterns, list):
                        all_patterns.extend([p.get('pattern', '') for p in patterns])
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error in {analysis_type} analysis: {e}")
                analyses[analysis_type] = {'error': str(e)}
        
        # Calculate ensemble prediction
        ensemble_prob = np.mean(probabilities) if probabilities else 0.5
        
        # Count pattern occurrences
        pattern_counts = Counter(all_patterns)
        
        return {
            'ensemble_probability': ensemble_prob,
            'prediction': 'gpt4o' if ensemble_prob > 0.5 else 'human',
            'confidence': abs(ensemble_prob - 0.5) * 2,
            'individual_analyses': analyses,
            'pattern_consensus': dict(pattern_counts.most_common(10)),
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_on_dataset(self, data_collector: DataCollector,
                           sample_size: int = 50) -> Dict[str, Any]:
        """
        Evaluate LLM analyzer on labeled dataset
        """
        
        dataset = data_collector.dataset
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        
        # Sample balanced dataset
        gpt4o_samples = [s for s in dataset if s['label'] == 'gpt4o'][:sample_size//2]
        human_samples = [s for s in dataset if s['label'] == 'human'][:sample_size//2]
        
        test_samples = gpt4o_samples + human_samples
        
        results = []
        correct = 0
        
        for i, sample in enumerate(test_samples):
            print(f"Evaluating sample {i+1}/{len(test_samples)}...")
            
            try:
                prediction = self.create_ensemble_prediction(sample['text'])
                
                predicted_label = prediction['prediction']
                true_label = sample['label']
                
                is_correct = predicted_label == true_label
                if is_correct:
                    correct += 1
                
                results.append({
                    'text': sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text'],
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'probability': prediction['ensemble_probability'],
                    'confidence': prediction['confidence'],
                    'correct': is_correct,
                    'patterns': prediction['pattern_consensus']
                })
                
            except Exception as e:
                print(f"Error evaluating sample {i+1}: {e}")
                results.append({
                    'text': sample['text'][:100],
                    'error': str(e)
                })
        
        accuracy = correct / len([r for r in results if 'error' not in r])
        
        return {
            'accuracy': accuracy,
            'total_samples': len(test_samples),
            'correct_predictions': correct,
            'detailed_results': results,
            'model_used': self.model,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Demo LLM analysis capabilities"""
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("Get your key from: https://openrouter.ai/")
        return
    
    analyzer = LLMAnalyzer(api_key)
    
    # Demo texts
    gpt4o_text = "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity across various sectors. On the other hand, concerns about job displacement and ethical implications require careful consideration."
    
    human_text = "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore. feels like we're heading straight for skynet territory lol 😅"
    
    print("🤖 LLM-Powered AI Detection Analysis")
    print("=" * 60)
    
    # Analyze GPT-4o sample
    print("\n📝 Analyzing GPT-4o sample...")
    result1 = analyzer.create_ensemble_prediction(gpt4o_text)
    print(f"Prediction: {result1['prediction']} ({result1['ensemble_probability']:.2%} confidence)")
    print(f"Top patterns: {list(result1['pattern_consensus'].keys())[:3]}")
    
    # Analyze human sample  
    print("\n📝 Analyzing human sample...")
    result2 = analyzer.create_ensemble_prediction(human_text)
    print(f"Prediction: {result2['prediction']} ({result2['ensemble_probability']:.2%} confidence)")
    print(f"Top patterns: {list(result2['pattern_consensus'].keys())[:3]}")
    
    print("\n✅ LLM analysis complete!")
    print("\nTo use with your dataset:")
    print("  1. Set OPENROUTER_API_KEY environment variable")
    print("  2. Run: python mining/llm_analyzer.py")

if __name__ == "__main__":
    main()