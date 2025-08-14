import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import re
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import spacy

class GPT4oPatternMiner:
    def __init__(self):
        self.patterns = []
        self.gpt_data = []
        self.human_data = []
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Spacy model not found. Using basic analysis.")
            self.nlp = None
    
    def generate_prompts(self) -> List[str]:
        prompts = [
            "Explain quantum computing in 280 characters",
            "What's your take on the latest AI developments?",
            "React to this: Technology is moving too fast for ethics to keep up",
            "Write a tweet about climate change solutions",
            "Describe the future of work in a casual tweet",
            "Share thoughts on social media's impact on society",
            "Quick take on cryptocurrency",
            "Explain machine learning simply",
            "React to: AI will replace all jobs",
            "Tweet about privacy in the digital age",
            "What's the deal with NFTs?",
            "Thoughts on remote work culture",
            "Explain blockchain like I'm five",
            "React to recent tech layoffs",
            "Tweet about mental health awareness",
            "Quick opinion on electric vehicles",
            "Describe your perfect productivity system",
            "React to: Social media is destroying democracy",
            "Tweet about the importance of coding",
            "Thoughts on the metaverse",
        ]
        
        casual_prompts = [
            "Just had coffee, what's everyone up to?",
            "Weekend plans anyone?",
            "Can't believe it's already December",
            "Who else is tired of meetings?",
            "Best productivity tip, go!",
            "Movie recommendations?",
            "Anyone else struggling with work-life balance?",
            "Thoughts on the new iPhone?",
            "Is it just me or is time flying?",
            "Coffee or tea debate, let's go",
        ]
        
        technical_prompts = [
            "Explain REST APIs briefly",
            "Python vs JavaScript for beginners?",
            "Best practices for code reviews",
            "How to optimize database queries",
            "React vs Vue in 2024",
            "Microservices pros and cons",
            "DevOps explained simply",
            "Git workflow tips",
            "Docker basics in a tweet",
            "Cloud vs on-premise hosting",
        ]
        
        return prompts + casual_prompts + technical_prompts
    
    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        features = {}
        
        sentences = text.split('.')
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        words = text.split()
        features['total_words'] = len(words)
        features['unique_words'] = len(set(words))
        features['lexical_diversity'] = features['unique_words'] / features['total_words'] if features['total_words'] > 0 else 0
        
        hedge_words = ['perhaps', 'maybe', 'possibly', 'might', 'could', 'seems', 'appears', 'likely', 'probably', 'generally', 'typically', 'often', 'sometimes']
        features['hedge_frequency'] = sum(1 for word in words if word.lower() in hedge_words) / len(words) if words else 0
        
        formal_indicators = ['furthermore', 'moreover', 'consequently', 'therefore', 'thus', 'hence', 'accordingly', 'nevertheless', 'nonetheless']
        features['formality_score'] = sum(1 for word in words if word.lower() in formal_indicators) / len(words) if words else 0
        
        contrast_patterns = [
            r'not\s+\w+,?\s+but\s+\w+',
            r'while\s+\w+.*,\s+\w+',
            r'although\s+.*,\s+',
            r'however,?\s+',
            r'on\s+the\s+other\s+hand',
            r'in\s+contrast',
        ]
        features['contrast_rhetoric'] = sum(1 for pattern in contrast_patterns if re.search(pattern, text.lower())) / len(sentences) if sentences else 0
        
        list_indicators = [r'firstly', r'secondly', r'thirdly', r'first,', r'second,', r'third,', r'1\.', r'2\.', r'3\.', r'•', r'-\s+']
        features['list_structure'] = any(re.search(pattern, text.lower()) for pattern in list_indicators)
        
        qualifier_phrases = ['it\'s important to note', 'it\'s worth noting', 'it should be noted', 'keep in mind', 'bear in mind', 'consider that', 'remember that']
        features['qualifier_frequency'] = sum(1 for phrase in qualifier_phrases if phrase in text.lower()) / len(sentences) if sentences else 0
        
        return features
    
    def identify_gpt4o_patterns(self, texts: List[str]) -> List[Dict[str, Any]]:
        identified_patterns = []
        
        for text in texts:
            features = self.extract_linguistic_features(text)
            
            if features['hedge_frequency'] > 0.05:
                identified_patterns.append({
                    'pattern': 'excessive_hedging',
                    'description': 'High frequency of hedging words (perhaps, maybe, possibly)',
                    'frequency': features['hedge_frequency'],
                    'example': text[:100]
                })
            
            if features['contrast_rhetoric'] > 0.3:
                identified_patterns.append({
                    'pattern': 'contrast_rhetoric',
                    'description': 'Frequent use of "not X, but Y" constructions',
                    'frequency': features['contrast_rhetoric'],
                    'example': text[:100]
                })
            
            if features['formality_score'] > 0.02:
                identified_patterns.append({
                    'pattern': 'formal_in_casual',
                    'description': 'Formal language in casual contexts',
                    'frequency': features['formality_score'],
                    'example': text[:100]
                })
            
            if features['list_structure']:
                identified_patterns.append({
                    'pattern': 'structured_lists',
                    'description': 'Tendency to create numbered or bulleted lists',
                    'frequency': 1.0,
                    'example': text[:100]
                })
            
            if features['qualifier_frequency'] > 0.1:
                identified_patterns.append({
                    'pattern': 'excessive_qualifiers',
                    'description': 'Overuse of qualifier phrases like "it\'s important to note"',
                    'frequency': features['qualifier_frequency'],
                    'example': text[:100]
                })
        
        balanced_phrases = [
            r'on\s+one\s+hand.*on\s+the\s+other',
            r'advantages.*disadvantages',
            r'benefits.*drawbacks',
            r'pros.*cons',
            r'positive.*negative',
        ]
        
        for text in texts:
            if any(re.search(pattern, text.lower()) for pattern in balanced_phrases):
                identified_patterns.append({
                    'pattern': 'balanced_presentation',
                    'description': 'Presenting both sides equally (pros/cons, advantages/disadvantages)',
                    'frequency': 0.4,
                    'example': text[:100]
                })
        
        pattern_summary = {}
        for pattern in identified_patterns:
            key = pattern['pattern']
            if key not in pattern_summary:
                pattern_summary[key] = {
                    'pattern': key,
                    'description': pattern['description'],
                    'occurrences': 0,
                    'avg_frequency': 0,
                    'examples': []
                }
            pattern_summary[key]['occurrences'] += 1
            pattern_summary[key]['avg_frequency'] += pattern['frequency']
            if len(pattern_summary[key]['examples']) < 3:
                pattern_summary[key]['examples'].append(pattern['example'])
        
        for key in pattern_summary:
            if pattern_summary[key]['occurrences'] > 0:
                pattern_summary[key]['avg_frequency'] /= pattern_summary[key]['occurrences']
        
        return list(pattern_summary.values())
    
    def cluster_patterns(self, texts: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
        if not texts:
            return {}
        
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        clustered_texts = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_texts:
                clustered_texts[cluster_id] = []
            clustered_texts[cluster_id].append(texts[idx])
        
        feature_names = vectorizer.get_feature_names_out()
        cluster_characteristics = {}
        
        for cluster_id in clustered_texts:
            cluster_texts = clustered_texts[cluster_id]
            cluster_tfidf = vectorizer.transform(cluster_texts)
            avg_tfidf = cluster_tfidf.mean(axis=0).A1
            top_indices = avg_tfidf.argsort()[-10:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            cluster_characteristics[cluster_id] = {
                'top_features': top_features,
                'sample_texts': cluster_texts[:3]
            }
        
        return cluster_characteristics
    
    def generate_pattern_report(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_patterns_identified': len(patterns),
            'patterns': patterns,
            'detection_rules': self.generate_detection_rules(patterns),
            'confidence_threshold': 0.7,
            'recommended_features': [
                'hedge_frequency',
                'contrast_rhetoric',
                'formality_score',
                'list_structure',
                'qualifier_frequency',
                'balanced_presentation'
            ]
        }
        return report
    
    def generate_detection_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rules = []
        
        for pattern in patterns:
            if pattern['pattern'] == 'excessive_hedging':
                rules.append({
                    'rule_id': 'HEDGE_01',
                    'pattern': 'excessive_hedging',
                    'regex': r'\b(perhaps|maybe|possibly|might|could|seems|appears|likely|probably)\b',
                    'threshold': 3,
                    'weight': 0.15
                })
            elif pattern['pattern'] == 'contrast_rhetoric':
                rules.append({
                    'rule_id': 'CONTRAST_01',
                    'pattern': 'contrast_rhetoric',
                    'regex': r'not\s+\w+,?\s+but\s+\w+',
                    'threshold': 1,
                    'weight': 0.25
                })
            elif pattern['pattern'] == 'formal_in_casual':
                rules.append({
                    'rule_id': 'FORMAL_01',
                    'pattern': 'formal_in_casual',
                    'regex': r'\b(furthermore|moreover|consequently|therefore|thus|hence|accordingly)\b',
                    'threshold': 1,
                    'weight': 0.20
                })
            elif pattern['pattern'] == 'structured_lists':
                rules.append({
                    'rule_id': 'LIST_01',
                    'pattern': 'structured_lists',
                    'regex': r'(firstly|secondly|thirdly|\d\.|\•|-\s+)',
                    'threshold': 2,
                    'weight': 0.15
                })
            elif pattern['pattern'] == 'excessive_qualifiers':
                rules.append({
                    'rule_id': 'QUAL_01',
                    'pattern': 'excessive_qualifiers',
                    'regex': r"it's (important|worth) (to note|noting)|keep in mind|bear in mind",
                    'threshold': 1,
                    'weight': 0.20
                })
            elif pattern['pattern'] == 'balanced_presentation':
                rules.append({
                    'rule_id': 'BALANCE_01',
                    'pattern': 'balanced_presentation',
                    'regex': r'(advantages.*disadvantages|pros.*cons|benefits.*drawbacks)',
                    'threshold': 1,
                    'weight': 0.25
                })
        
        return rules
    
    def save_patterns(self, patterns: Dict[str, Any], filename: str = 'gpt4o_patterns.json'):
        with open(filename, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"Patterns saved to {filename}")
    
    def validate_patterns(self, test_texts: List[Dict[str, Any]]) -> Dict[str, float]:
        correct_predictions = 0
        total_predictions = len(test_texts)
        
        pattern_report = self.generate_pattern_report(self.identify_gpt4o_patterns([t['text'] for t in test_texts]))
        rules = pattern_report['detection_rules']
        
        for item in test_texts:
            text = item['text']
            is_gpt4o = item['is_gpt4o']
            
            score = 0
            for rule in rules:
                matches = len(re.findall(rule['regex'], text.lower()))
                if matches >= rule['threshold']:
                    score += rule['weight']
            
            predicted_gpt4o = score >= 0.5
            
            if predicted_gpt4o == is_gpt4o:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_tested': total_predictions,
            'correct': correct_predictions,
            'threshold_used': 0.5
        }

def main():
    miner = GPT4oPatternMiner()
    
    print("GPT-4o Pattern Mining System")
    print("=" * 50)
    
    prompts = miner.generate_prompts()
    print(f"Generated {len(prompts)} diverse prompts for analysis")
    
    sample_gpt4o_responses = [
        "While quantum computing is complex, it's essentially about using quantum mechanics to process information. Not traditional binary, but quantum states that can be both 0 and 1 simultaneously. This could revolutionize cryptography and drug discovery.",
        "The latest AI developments are fascinating. On one hand, we're seeing incredible advances in capability. On the other hand, there are valid concerns about safety and alignment. It's important to note that balanced regulation will be key.",
        "Technology advancement presents both opportunities and challenges. Firstly, innovation drives progress. Secondly, ethical frameworks need time to develop. However, this gap isn't necessarily problematic - it's a natural part of evolution.",
        "Climate change solutions require multifaceted approaches: 1) Renewable energy adoption, 2) Carbon capture technology, 3) Policy changes. Each has advantages and disadvantages, but collectively they offer hope.",
        "The future of work is likely hybrid. Not fully remote, but flexible. Companies are realizing that productivity isn't tied to location. However, collaboration and culture remain important considerations.",
    ]
    
    sample_human_tweets = [
        "quantum computing is wild. basically magic computers that break encryption. we're not ready for this tbh",
        "AI is moving so fast I can barely keep up anymore. feels like every week there's something new",
        "tech companies really said 'ethics? never heard of her' and just went full speed ahead lol",
        "we need more solar panels. like way more. why is this even a debate still??",
        "wfh forever please. commuting was such a waste of time and money",
    ]
    
    print("\nAnalyzing GPT-4o response patterns...")
    gpt_patterns = miner.identify_gpt4o_patterns(sample_gpt4o_responses)
    
    print(f"\nIdentified {len(gpt_patterns)} unique patterns:")
    for pattern in gpt_patterns:
        print(f"  - {pattern['pattern']}: {pattern['description']}")
        print(f"    Average frequency: {pattern['avg_frequency']:.2f}")
    
    print("\nClustering patterns...")
    clusters = miner.cluster_patterns(sample_gpt4o_responses + sample_human_tweets, n_clusters=3)
    
    for cluster_id, characteristics in clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Top features: {', '.join(characteristics['top_features'][:5])}")
    
    pattern_report = miner.generate_pattern_report(gpt_patterns)
    miner.save_patterns(pattern_report, '../data/gpt4o_patterns.json')
    
    print("\nValidating patterns on test set...")
    test_data = [
        {'text': sample_gpt4o_responses[0], 'is_gpt4o': True},
        {'text': sample_gpt4o_responses[1], 'is_gpt4o': True},
        {'text': sample_human_tweets[0], 'is_gpt4o': False},
        {'text': sample_human_tweets[1], 'is_gpt4o': False},
    ]
    
    validation_results = miner.validate_patterns(test_data)
    print(f"Validation accuracy: {validation_results['accuracy']:.2%}")
    print(f"Correct predictions: {validation_results['correct']}/{validation_results['total_tested']}")
    
    print("\n" + "=" * 50)
    print("Mining complete! Patterns saved to ../data/gpt4o_patterns.json")

if __name__ == "__main__":
    main()