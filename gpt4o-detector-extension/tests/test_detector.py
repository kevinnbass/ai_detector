import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mining'))

from detector import GPT4oDetector, FastDetector

class TestGPT4oDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GPT4oDetector()
        self.fast_detector = FastDetector()
        
        self.gpt4o_samples = [
            "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity. On the other hand, job displacement remains a valid concern.",
            "Climate change presents numerous challenges. Firstly, rising sea levels threaten coastal communities. Secondly, extreme weather events are becoming more frequent. However, technological solutions are emerging that could help mitigate these issues.",
            "The blockchain technology debate is fascinating. Not simple hype, but a genuinely transformative innovation. Nevertheless, scalability issues persist. That said, the potential applications are vast and varied.",
        ]
        
        self.human_samples = [
            "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore",
            "climate change is real and we need to act NOW. stop debating and start doing something about it!!", 
            "blockchain = overhyped nonsense. change my mind 🤷‍♂️",
        ]
    
    def test_detector_initialization(self):
        """Test that detector initializes with default patterns"""
        self.assertIsInstance(self.detector, GPT4oDetector)
        self.assertEqual(self.detector.threshold, 0.7)
        self.assertTrue(len(self.detector.rules) > 0)
    
    def test_gpt4o_detection(self):
        """Test detection of GPT-4o samples"""
        results = []
        for sample in self.gpt4o_samples:
            result = self.detector.detect(sample)
            results.append(result.is_gpt4o)
            print(f"GPT-4o sample: {result.is_gpt4o} ({result.confidence:.2%})")
            if result.matched_patterns:
                print(f"  Patterns: {[p['pattern'] for p in result.matched_patterns]}")
        
        accuracy = sum(results) / len(results)
        self.assertGreaterEqual(accuracy, 0.6, f"GPT-4o detection accuracy too low: {accuracy:.2%}")
    
    def test_human_detection(self):
        """Test detection of human samples"""
        results = []
        for sample in self.human_samples:
            result = self.detector.detect(sample)
            results.append(not result.is_gpt4o)
            print(f"Human sample: {not result.is_gpt4o} ({1-result.confidence:.2%})")
        
        accuracy = sum(results) / len(results)
        self.assertGreaterEqual(accuracy, 0.6, f"Human detection accuracy too low: {accuracy:.2%}")
    
    def test_short_text_handling(self):
        """Test handling of short texts"""
        short_text = "Hello world"
        result = self.detector.detect(short_text)
        self.assertFalse(result.is_gpt4o)
        self.assertEqual(result.confidence, 0)
    
    def test_empty_text_handling(self):
        """Test handling of empty texts"""
        result = self.detector.detect("")
        self.assertFalse(result.is_gpt4o)
        self.assertEqual(result.confidence, 0)
    
    def test_threshold_adjustment(self):
        """Test threshold adjustment"""
        original_threshold = self.detector.threshold
        self.detector.update_threshold(0.8)
        self.assertEqual(self.detector.threshold, 0.8)
        
        # Test bounds
        self.detector.update_threshold(1.5)
        self.assertEqual(self.detector.threshold, 1.0)
        
        self.detector.update_threshold(-0.5)
        self.assertEqual(self.detector.threshold, 0.0)
        
        # Restore original
        self.detector.update_threshold(original_threshold)
    
    def test_fast_detector(self):
        """Test fast detector"""
        for sample in self.gpt4o_samples:
            is_gpt, confidence = self.fast_detector.quick_detect(sample)
            print(f"Fast detection: {is_gpt} ({confidence:.2%})")
            self.assertIsInstance(is_gpt, bool)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_batch_detection(self):
        """Test batch detection"""
        all_samples = self.gpt4o_samples + self.human_samples
        results = self.detector.batch_detect(all_samples)
        
        self.assertEqual(len(results), len(all_samples))
        for result in results:
            self.assertIn('is_gpt4o', result.__dict__)
            self.assertIn('confidence', result.__dict__)
    
    def test_pattern_scoring(self):
        """Test individual pattern scoring"""
        # Text with obvious GPT-4o patterns
        gpt_text = "It's important to note that while AI has advantages, it also has disadvantages. Firstly, productivity increases. However, job displacement occurs."
        result = self.detector.detect(gpt_text)
        
        self.assertTrue(result.is_gpt4o)
        self.assertGreater(len(result.matched_patterns), 0)
        self.assertIn('pattern_scores', result.__dict__)
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        sample_text = self.gpt4o_samples[0]
        
        # Clear cache and test
        self.detector.clear_cache()
        stats_before = self.detector.get_stats()
        
        # First detection
        result1 = self.detector.detect(sample_text)
        stats_after = self.detector.get_stats()
        
        # Second detection (should use cache)
        result2 = self.detector.detect(sample_text)
        
        self.assertEqual(result1.confidence, result2.confidence)
        self.assertEqual(result1.is_gpt4o, result2.is_gpt4o)

class TestDetectionPatterns(unittest.TestCase):
    def setUp(self):
        self.detector = GPT4oDetector()
    
    def test_hedge_pattern(self):
        """Test hedging pattern detection"""
        hedge_text = "Perhaps this might possibly seem like it could appear to be likely true, maybe."
        result = self.detector.detect(hedge_text)
        
        hedge_patterns = [p for p in result.matched_patterns if 'hedg' in p['pattern'].lower()]
        self.assertGreater(len(hedge_patterns), 0)
    
    def test_contrast_pattern(self):
        """Test contrast pattern detection"""
        contrast_text = "Not simple solutions, but complex ones. While technology advances, society lags behind."
        result = self.detector.detect(contrast_text)
        
        contrast_patterns = [p for p in result.matched_patterns if 'contrast' in p['pattern'].lower()]
        self.assertGreater(len(contrast_patterns), 0)
    
    def test_formal_pattern(self):
        """Test formal language pattern"""
        formal_text = "Furthermore, we must consider the implications. Moreover, the consequences are significant. Therefore, action is required."
        result = self.detector.detect(formal_text)
        
        formal_patterns = [p for p in result.matched_patterns if 'formal' in p['pattern'].lower()]
        self.assertGreater(len(formal_patterns), 0)
    
    def test_list_pattern(self):
        """Test structured list pattern"""
        list_text = "The solution involves several steps: 1. Analysis 2. Planning 3. Implementation. Firstly, we analyze. Secondly, we plan."
        result = self.detector.detect(list_text)
        
        list_patterns = [p for p in result.matched_patterns if 'list' in p['pattern'].lower()]
        self.assertGreater(len(list_patterns), 0)
    
    def test_qualifier_pattern(self):
        """Test qualifier pattern detection"""
        qualifier_text = "It's important to note that this approach has merit. Keep in mind that results may vary. Remember that context matters."
        result = self.detector.detect(qualifier_text)
        
        qualifier_patterns = [p for p in result.matched_patterns if 'qual' in p['pattern'].lower()]
        self.assertGreater(len(qualifier_patterns), 0)

if __name__ == '__main__':
    print("Running GPT-4o Detector Test Suite")
    print("=" * 50)
    
    unittest.main(verbosity=2)