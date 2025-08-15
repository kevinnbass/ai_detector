import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import openai
from datetime import datetime
import os
import re
from collections import Counter

from data_collector import DataCollector
from trainer import GPT4oTrainer
from detector import GPT4oDetector

class ActiveLearner:
    """
    Active learning system that identifies the most valuable samples to label
    """
    
    def __init__(self, data_file: str = "../data/labeled_dataset.json", 
                 openai_api_key: Optional[str] = None):
        self.data_collector = DataCollector(data_file)
        self.trainer = GPT4oTrainer(data_file)
        self.openai_client = None
        self.current_model = None
        self.uncertainty_threshold = 0.3  # Confidence range for uncertain samples
        
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
    
    def generate_gpt4o_samples(self, num_samples: int = 50, 
                              topics: List[str] = None) -> List[str]:
        """
        Generate samples using GPT-4o for positive examples
        
        Args:
            num_samples: Number of samples to generate
            topics: List of topics to generate content about
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key required for sample generation")
        
        if not topics:
            topics = [
                "artificial intelligence and ethics",
                "climate change solutions", 
                "cryptocurrency and blockchain",
                "remote work and productivity",
                "social media impact on society",
                "quantum computing basics",
                "renewable energy technologies",
                "mental health awareness",
                "digital privacy concerns",
                "future of education technology"
            ]
        
        # Prompts that encourage GPT-4o style responses
        prompt_templates = [
            "Write a thoughtful tweet about {topic}. Consider both the advantages and disadvantages.",
            "Explain {topic} in a balanced way, noting both benefits and concerns.",
            "Share your perspective on {topic}. It's important to note the various viewpoints.",
            "Discuss {topic}. Firstly, consider the positive aspects. Secondly, address potential challenges.",
            "Analyze {topic}. While there are benefits, we should also consider the drawbacks.",
            "Comment on {topic}. Not simply good or bad, but nuanced with multiple considerations.",
            "Reflect on {topic}. Perhaps most importantly, we should examine both sides.",
            "Evaluate {topic}. Generally speaking, there are both opportunities and risks to consider."
        ]
        
        generated_samples = []
        
        for i in range(num_samples):
            topic = np.random.choice(topics)
            template = np.random.choice(prompt_templates)
            prompt = template.format(topic=topic)
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=280,  # Tweet-like length
                    temperature=0.8
                )
                
                generated_text = response.choices[0].message.content.strip()
                generated_samples.append(generated_text)
                
                # Add to dataset automatically
                self.data_collector.add_sample(
                    text=generated_text,
                    label='gpt4o',
                    confidence=1.0,
                    source='gpt4o_generated',
                    metadata={'topic': topic, 'prompt': prompt}
                )
                
                print(f"Generated {i+1}/{num_samples}: {generated_text[:60]}...")
                
            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")
                continue
        
        print(f"Generated {len(generated_samples)} GPT-4o samples")
        return generated_samples
    
    def find_uncertain_samples(self, candidate_texts: List[str], 
                             num_samples: int = 20) -> List[Dict[str, Any]]:
        """
        Find samples where the current model is most uncertain
        
        Args:
            candidate_texts: Pool of unlabeled texts to choose from
            num_samples: Number of uncertain samples to return
        """
        if not self.current_model:
            self.train_initial_model()
        
        # Get predictions for all candidates
        X = self.trainer.extract_features(candidate_texts)
        probabilities = self.current_model.predict_proba(X)
        
        # Calculate uncertainty (distance from 0.5 decision boundary)
        uncertainties = []
        for i, prob in enumerate(probabilities):
            gpt4o_prob = prob[1]
            uncertainty = abs(0.5 - gpt4o_prob)  # Lower is more uncertain
            uncertainties.append({
                'text': candidate_texts[i],
                'gpt4o_probability': gpt4o_prob,
                'uncertainty': uncertainty,
                'index': i
            })
        
        # Sort by uncertainty (most uncertain first)
        uncertainties.sort(key=lambda x: x['uncertainty'])
        
        return uncertainties[:num_samples]
    
    def find_diverse_samples(self, candidate_texts: List[str], 
                           num_samples: int = 20) -> List[str]:
        """
        Find diverse samples using feature-based clustering
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Extract features
        X = self.trainer.extract_features(candidate_texts)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster to find diverse samples
        n_clusters = min(num_samples, len(candidate_texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Select one sample from each cluster (closest to centroid)
        diverse_samples = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Find sample closest to cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = [np.linalg.norm(X_scaled[idx] - centroid) 
                           for idx in cluster_indices]
                closest_idx = cluster_indices[np.argmin(distances)]
                diverse_samples.append(candidate_texts[closest_idx])
        
        return diverse_samples
    
    def train_initial_model(self):
        """Train initial model with current labeled data"""
        stats = self.data_collector.get_statistics()
        
        if stats['total_samples'] < 20:
            raise ValueError("Need at least 20 labeled samples to train initial model")
        
        print("Training initial model with labeled data...")
        X, y, texts = self.trainer.prepare_training_data(min_samples=20)
        
        # Use simple model for active learning
        self.current_model = RandomForestClassifier(
            n_estimators=50, 
            random_state=42,
            class_weight='balanced'
        )
        
        self.current_model.fit(X, y)
        
        # Evaluate
        predictions = self.current_model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Initial model accuracy: {accuracy:.3f}")
    
    def suggest_next_samples(self, candidate_pool: List[str], 
                           strategy: str = 'uncertainty',
                           num_suggestions: int = 10) -> List[Dict[str, Any]]:
        """
        Suggest next samples to label based on active learning strategy
        
        Args:
            candidate_pool: Pool of unlabeled texts
            strategy: 'uncertainty', 'diversity', or 'mixed'
            num_suggestions: Number of samples to suggest
        """
        suggestions = []
        
        if strategy == 'uncertainty':
            uncertain_samples = self.find_uncertain_samples(candidate_pool, num_suggestions)
            for sample in uncertain_samples:
                suggestions.append({
                    'text': sample['text'],
                    'strategy': 'uncertainty',
                    'score': sample['uncertainty'],
                    'gpt4o_prob': sample['gpt4o_probability'],
                    'reason': f"Model uncertain (prob: {sample['gpt4o_probability']:.2f})"
                })
        
        elif strategy == 'diversity':
            diverse_samples = self.find_diverse_samples(candidate_pool, num_suggestions)
            for sample in diverse_samples:
                suggestions.append({
                    'text': sample,
                    'strategy': 'diversity', 
                    'score': 1.0,
                    'reason': "Diverse feature representation"
                })
        
        elif strategy == 'mixed':
            # Half uncertainty, half diversity
            uncertain_count = num_suggestions // 2
            diverse_count = num_suggestions - uncertain_count
            
            uncertain_samples = self.find_uncertain_samples(candidate_pool, uncertain_count)
            diverse_samples = self.find_diverse_samples(candidate_pool, diverse_count)
            
            for sample in uncertain_samples:
                suggestions.append({
                    'text': sample['text'],
                    'strategy': 'uncertainty',
                    'score': sample['uncertainty'],
                    'gpt4o_prob': sample['gpt4o_probability'],
                    'reason': f"Model uncertain (prob: {sample['gpt4o_probability']:.2f})"
                })
            
            for sample in diverse_samples:
                suggestions.append({
                    'text': sample,
                    'strategy': 'diversity',
                    'score': 1.0, 
                    'reason': "Diverse feature representation"
                })
        
        return suggestions
    
    def interactive_labeling_session(self, candidate_pool: List[str] = None):
        """
        Interactive session for active learning with suggestions
        """
        self._print_session_header()
        
        if not candidate_pool:
            candidate_pool = self._get_candidate_pool()
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if self._should_exit(command):
                    break
                
                self._process_command(command, candidate_pool)
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_session_header(self):
        """Print interactive session header"""
        print("\n" + "="*80)
        print("ACTIVE LEARNING SESSION")
        print("="*80)
        print("I'll suggest the most valuable samples for you to label!")
        print("\nCommands:")
        print("  suggest [n]     - Get n sample suggestions (default: 5)")
        print("  label <text>    - Label a specific text")
        print("  stats           - Show dataset statistics")
        print("  generate [n]    - Generate GPT-4o samples (requires API key)")
        print("  retrain         - Retrain model with new data")
        print("  save            - Save current dataset")
        print("  quit            - Exit session")
        print("="*80)
    
    def _should_exit(self, command: str) -> bool:
        """Check if command indicates session should exit"""
        return command.lower() in ['quit', 'exit', 'q']
    
    def _process_command(self, command: str, candidate_pool: List[str]):
        """Process a single command"""
        if command.startswith('suggest'):
            self._handle_suggest_command(command, candidate_pool)
        elif command.startswith('label'):
            self._handle_label_command(command)
        elif command == 'stats':
            self._handle_stats_command()
        elif command.startswith('generate'):
            self._handle_generate_command(command)
        elif command == 'retrain':
            self._handle_retrain_command()
        elif command == 'save':
            self._handle_save_command()
        else:
            print("Unknown command. Type 'quit' to exit or see available commands above.")
    
    def _handle_suggest_command(self, command: str, candidate_pool: List[str]):
        """Handle suggest command"""
        parts = command.split()
        n = int(parts[1]) if len(parts) > 1 else 5
        
        if not self._ensure_model_ready():
            return
        
        suggestions = self.suggest_next_samples(candidate_pool, 'mixed', n)
        self._process_suggestions(suggestions)
    
    def _ensure_model_ready(self) -> bool:
        """Ensure model is ready for predictions"""
        if not self.current_model:
            try:
                self.train_initial_model()
                return True
            except ValueError as e:
                print(f"Error: {e}")
                print("Need more labeled data to make suggestions.")
                return False
        return True
    
    def _process_suggestions(self, suggestions: List[Dict]):
        """Process and display suggestions for labeling"""
        print(f"\nTop {len(suggestions)} suggestions to label:")
        for i, sug in enumerate(suggestions, 1):
            print(f"\n{i}. [{sug['strategy'].upper()}] {sug['reason']}")
            print(f"   Text: {sug['text'][:100]}...")
            
            label = self._get_user_label()
            if label:
                self._add_labeled_sample(sug['text'], label)
    
    def _get_user_label(self) -> Optional[str]:
        """Get label from user input"""
        label = input(f"   Label as (g)pt4o, (h)uman, or (s)kip? ").strip().lower()
        if label in ['g', 'gpt4o']:
            return 'gpt4o'
        elif label in ['h', 'human']:
            return 'human'
        return None
    
    def _add_labeled_sample(self, text: str, label: str):
        """Add labeled sample to dataset"""
        self.data_collector.add_sample(text, label, source='active_learning')
        print(f"   ✅ Labeled as {label.title()}")
    
    def _handle_label_command(self, command: str):
        """Handle manual label command"""
        text = command[6:].strip()
        if not text:
            print("Please provide text to label: label <text>")
            return
        
        label = input(f"Label '{text[:50]}...' as (g)pt4o or (h)uman? ").strip().lower()
        if label in ['g', 'gpt4o']:
            self.data_collector.add_sample(text, 'gpt4o', source='manual')
            print("✅ Labeled as GPT-4o")
        elif label in ['h', 'human']:
            self.data_collector.add_sample(text, 'human', source='manual')
            print("✅ Labeled as Human")
    
    def _handle_stats_command(self):
        """Handle stats command"""
        stats = self.data_collector.get_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  GPT-4o samples: {stats['gpt4o_samples']}")
        print(f"  Human samples: {stats['human_samples']}")
        if stats['total_samples'] > 0:
            print(f"  Balance ratio: {stats['balance_ratio']:.2f}")
            print(f"  Sources: {', '.join(stats['sources'])}")
    
    def _handle_generate_command(self, command: str):
        """Handle generate command"""
        parts = command.split()
        n = int(parts[1]) if len(parts) > 1 else 10
        
        try:
            self.generate_gpt4o_samples(n)
            print(f"✅ Generated {n} GPT-4o samples")
        except ValueError as e:
            print(f"Error: {e}")
            print("Set OPENAI_API_KEY environment variable to generate samples")
    
    def _handle_retrain_command(self):
        """Handle retrain command"""
        try:
            self.train_initial_model()
            print("✅ Model retrained with latest data")
        except ValueError as e:
            print(f"Error: {e}")
    
    def _handle_save_command(self):
        """Handle save command"""
        self.data_collector.save_dataset()
        print("✅ Dataset saved")
    
    def _get_candidate_pool(self) -> List[str]:
        """Get a pool of candidate texts for labeling suggestions"""
        # For demo purposes, return some example texts
        # In practice, you'd load from Twitter API, web scraping, etc.
        
        candidate_texts = [
            "AI development is accelerating at an unprecedented pace. While this brings exciting opportunities, it also raises important questions about safety and alignment.",
            "just saw the new AI announcement and wow this tech is moving FAST. kinda scary tbh but also super cool 🤖",
            "The implications of quantum computing are profound. Firstly, it will revolutionize cryptography. Secondly, it could accelerate drug discovery significantly.",
            "quantum computers gonna break all our passwords eventually lol. wild times ahead for cybersecurity folks",
            "Climate change mitigation requires multifaceted approaches. Not simple solutions, but complex interventions across multiple sectors and scales.",
            "we need to fix climate change NOW. stop talking and start doing something about it!!",
            "Remote work has fundamentally transformed the employment landscape. It's important to note both the benefits and challenges this presents.",
            "working from home is the best thing ever. never going back to the office if i can help it 💯",
            "Cryptocurrency adoption presents interesting possibilities. However, volatility remains a significant concern that cannot be ignored.",
            "crypto is just gambling with extra steps. change my mind 🎰",
        ]
        
        return candidate_texts

def main():
    """Main active learning interface"""
    print("Active Learning System for GPT-4o Detection")
    print("="*50)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found - can generate GPT-4o samples")
    else:
        print("⚠️  No OpenAI API key - limited to manual labeling")
    
    # Initialize active learner
    learner = ActiveLearner(openai_api_key=api_key)
    
    # Show current dataset stats
    stats = learner.data_collector.get_statistics()
    print(f"\nCurrent dataset: {stats['total_samples']} samples")
    
    if stats['total_samples'] < 10:
        print("\n💡 Tip: You need at least 20 samples to get AI suggestions.")
        if api_key:
            generate = input("Generate some GPT-4o samples to start? (y/n): ").strip().lower()
            if generate == 'y':
                learner.generate_gpt4o_samples(25)
                print("✅ Generated initial GPT-4o samples")
    
    # Start interactive session
    learner.interactive_labeling_session()

if __name__ == "__main__":
    main()