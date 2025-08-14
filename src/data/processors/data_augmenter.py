import re
import random
import numpy as np
from typing import List, Dict, Any
import nltk
from nltk.corpus import wordnet
import openai
from data_collector import DataCollector

class DataAugmenter:
    """
    Data augmentation tools to increase training dataset size
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_client = None
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK WordNet...")
            nltk.download('wordnet')
            nltk.download('punkt')
    
    def synonym_replacement(self, text: str, n: int = 2) -> List[str]:
        """
        Replace random words with synonyms while preserving GPT-4o patterns
        """
        words = text.split()
        if len(words) < 3:
            return [text]
        
        # Words to avoid replacing (preserve GPT-4o patterns)
        preserve_words = {
            'however', 'furthermore', 'moreover', 'nevertheless', 'therefore',
            'firstly', 'secondly', 'thirdly', 'perhaps', 'maybe', 'possibly',
            'important', 'note', 'consider', 'advantages', 'disadvantages',
            'pros', 'cons', 'benefits', 'drawbacks', 'essentially', 'basically'
        }
        
        augmented_texts = []
        
        for _ in range(n):
            new_words = words.copy()
            
            # Randomly select words to replace
            replaceable_indices = [
                i for i, word in enumerate(words) 
                if word.lower() not in preserve_words and len(word) > 3
            ]
            
            if replaceable_indices:
                num_replacements = min(2, len(replaceable_indices))
                replace_indices = random.sample(replaceable_indices, num_replacements)
                
                for idx in replace_indices:
                    word = words[idx].lower()
                    synonyms = []
                    
                    # Get synonyms from WordNet
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if synonym != word and synonym.isalpha():
                                synonyms.append(synonym)
                    
                    if synonyms:
                        # Choose synonym with similar length
                        similar_length_synonyms = [
                            s for s in synonyms 
                            if abs(len(s) - len(word)) <= 2
                        ]
                        chosen_synonyms = similar_length_synonyms if similar_length_synonyms else synonyms
                        new_word = random.choice(chosen_synonyms[:5])  # Top 5 to avoid weird ones
                        
                        # Preserve capitalization
                        if words[idx].isupper():
                            new_word = new_word.upper()
                        elif words[idx][0].isupper():
                            new_word = new_word.capitalize()
                        
                        new_words[idx] = new_word
            
            augmented_text = ' '.join(new_words)
            if augmented_text != text:
                augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def back_translation(self, text: str, intermediate_languages: List[str] = ['es', 'fr', 'de']) -> List[str]:
        """
        Use back-translation through intermediate languages to create variations
        Requires OpenAI API for translation
        """
        if not self.openai_client:
            return []
        
        augmented_texts = []
        
        for lang in intermediate_languages:
            try:
                # Translate to intermediate language
                translate_prompt = f"Translate this text to {lang}: '{text}'"
                response1 = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": translate_prompt}],
                    max_tokens=200
                )
                translated = response1.choices[0].message.content.strip()
                
                # Translate back to English
                back_translate_prompt = f"Translate this {lang} text back to English: '{translated}'"
                response2 = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": back_translate_prompt}],
                    max_tokens=200
                )
                back_translated = response2.choices[0].message.content.strip()
                
                if back_translated != text and len(back_translated) > 20:
                    augmented_texts.append(back_translated)
                    
            except Exception as e:
                print(f"Back-translation error with {lang}: {e}")
                continue
        
        return augmented_texts
    
    def paraphrase_with_gpt(self, text: str, style: str = "same", n: int = 3) -> List[str]:
        """
        Use GPT to paraphrase while maintaining the writing style
        """
        if not self.openai_client:
            return []
        
        style_instructions = {
            "same": "Paraphrase this text while maintaining the exact same writing style and tone",
            "gpt4o": "Paraphrase this text in a balanced, analytical style with formal language patterns",
            "human": "Paraphrase this text in a casual, informal style with natural language"
        }
        
        instruction = style_instructions.get(style, style_instructions["same"])
        
        augmented_texts = []
        
        for i in range(n):
            try:
                prompt = f"{instruction}:\n\n'{text}'\n\nParaphrased version:"
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7 + (i * 0.1)  # Increase variety
                )
                
                paraphrased = response.choices[0].message.content.strip()
                
                # Clean up response
                if paraphrased.startswith('"') and paraphrased.endswith('"'):
                    paraphrased = paraphrased[1:-1]
                
                if paraphrased != text and len(paraphrased) > 20:
                    augmented_texts.append(paraphrased)
                    
            except Exception as e:
                print(f"Paraphrasing error: {e}")
                continue
        
        return augmented_texts
    
    def sentence_reordering(self, text: str) -> List[str]:
        """
        Reorder sentences while preserving meaning and patterns
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return []
        
        augmented_texts = []
        
        # Only reorder if it makes sense (avoid breaking logical flow)
        if len(sentences) <= 3:
            # For short texts, try one reordering
            reordered = sentences.copy()
            
            # Swap first two sentences if neither starts with transition words
            transition_words = ['firstly', 'secondly', 'however', 'furthermore', 'therefore']
            
            if (not any(reordered[0].lower().startswith(tw) for tw in transition_words) and
                not any(reordered[1].lower().startswith(tw) for tw in transition_words)):
                reordered[0], reordered[1] = reordered[1], reordered[0]
                augmented_text = '. '.join(reordered) + '.'
                
                if augmented_text != text:
                    augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def noise_injection(self, text: str, noise_level: float = 0.1) -> List[str]:
        """
        Add subtle noise while preserving readability and patterns
        """
        augmented_texts = []
        
        # Character-level noise (very subtle)
        words = text.split()
        noisy_words = []
        
        for word in words:
            if len(word) > 4 and random.random() < noise_level:
                # Swap two adjacent characters (typo simulation)
                pos = random.randint(1, len(word) - 2)
                word_list = list(word)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                noisy_words.append(''.join(word_list))
            else:
                noisy_words.append(word)
        
        noisy_text = ' '.join(noisy_words)
        if noisy_text != text:
            augmented_texts.append(noisy_text)
        
        return augmented_texts
    
    def pattern_preserving_augmentation(self, text: str, label: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Apply multiple augmentation techniques while preserving label-specific patterns
        """
        augmented_samples = []
        
        # Apply different techniques based on label
        if label == 'gpt4o':
            # For GPT-4o samples, preserve formal patterns
            techniques = [
                ('synonym_replacement', lambda t: self.synonym_replacement(t, 2)),
                ('paraphrase_gpt4o', lambda t: self.paraphrase_with_gpt(t, 'gpt4o', 2)),
                ('sentence_reordering', lambda t: self.sentence_reordering(t)),
            ]
        else:
            # For human samples, preserve casual patterns
            techniques = [
                ('synonym_replacement', lambda t: self.synonym_replacement(t, 1)),
                ('paraphrase_human', lambda t: self.paraphrase_with_gpt(t, 'human', 2)),
                ('noise_injection', lambda t: self.noise_injection(t, 0.05)),
            ]
        
        # Add back-translation if API available
        if self.openai_client:
            techniques.append(('back_translation', lambda t: self.back_translation(t, ['es', 'fr'])))
        
        # Apply techniques
        for technique_name, technique_func in techniques:
            try:
                results = technique_func(text)
                for result in results:
                    if len(augmented_samples) >= n:
                        break
                    
                    augmented_samples.append({
                        'text': result,
                        'label': label,
                        'source': f'augmented_{technique_name}',
                        'original_text': text,
                        'technique': technique_name
                    })
                
                if len(augmented_samples) >= n:
                    break
                    
            except Exception as e:
                print(f"Augmentation error with {technique_name}: {e}")
                continue
        
        return augmented_samples[:n]
    
    def augment_dataset(self, data_collector: DataCollector, 
                       augmentation_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Augment the entire dataset
        
        Args:
            data_collector: DataCollector instance
            augmentation_ratio: Ratio of augmented samples to original samples
        """
        original_samples = data_collector.dataset
        
        if not original_samples:
            return {'augmented_count': 0, 'error': 'No samples to augment'}
        
        target_augmented_count = int(len(original_samples) * augmentation_ratio)
        samples_per_original = max(1, target_augmented_count // len(original_samples))
        
        augmented_count = 0
        
        print(f"Augmenting {len(original_samples)} samples...")
        print(f"Target: {target_augmented_count} augmented samples")
        
        for i, sample in enumerate(original_samples):
            if augmented_count >= target_augmented_count:
                break
            
            augmented = self.pattern_preserving_augmentation(
                sample['text'], 
                sample['label'], 
                samples_per_original
            )
            
            for aug_sample in augmented:
                data_collector.add_sample(
                    text=aug_sample['text'],
                    label=aug_sample['label'],
                    confidence=0.8,  # Lower confidence for augmented data
                    source=aug_sample['source'],
                    metadata={
                        'original_id': sample.get('id'),
                        'technique': aug_sample['technique'],
                        'augmented': True
                    }
                )
                augmented_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(original_samples)} samples...")
        
        print(f"Generated {augmented_count} augmented samples")
        
        return {
            'original_count': len(original_samples),
            'augmented_count': augmented_count,
            'total_count': len(data_collector.dataset),
            'augmentation_ratio_achieved': augmented_count / len(original_samples)
        }

def main():
    """Demo augmentation capabilities"""
    print("Data Augmentation Demo")
    print("="*50)
    
    # Example texts
    gpt4o_sample = "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity. On the other hand, job displacement remains a concern."
    
    human_sample = "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore lol"
    
    # Initialize augmenter (without API key for demo)
    augmenter = DataAugmenter()
    
    print("Original GPT-4o sample:")
    print(f"  {gpt4o_sample}")
    
    print("\nSynonym replacement augmentations:")
    synonyms = augmenter.synonym_replacement(gpt4o_sample, 3)
    for i, aug in enumerate(synonyms, 1):
        print(f"  {i}. {aug}")
    
    print(f"\nOriginal Human sample:")
    print(f"  {human_sample}")
    
    print("\nNoise injection augmentations:")
    noisy = augmenter.noise_injection(human_sample)
    for i, aug in enumerate(noisy, 1):
        print(f"  {i}. {aug}")
    
    # Demo with data collector
    print("\n" + "="*50)
    print("Dataset Augmentation Demo")
    
    collector = DataCollector("../data/demo_dataset.json")
    
    # Add some sample data
    collector.add_sample(gpt4o_sample, 'gpt4o', source='demo')
    collector.add_sample(human_sample, 'human', source='demo')
    
    # Augment dataset
    results = augmenter.augment_dataset(collector, augmentation_ratio=1.0)
    
    print(f"Augmentation Results:")
    print(f"  Original samples: {results['original_count']}")
    print(f"  Augmented samples: {results['augmented_count']}")
    print(f"  Total samples: {results['total_count']}")
    print(f"  Augmentation ratio: {results['augmentation_ratio_achieved']:.2f}")
    
    collector.save_dataset()
    print(f"Augmented dataset saved!")

if __name__ == "__main__":
    main()