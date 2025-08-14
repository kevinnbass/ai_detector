"""
Unified Analyzer Module
Consolidates functionality from llm_analyzer.py, gemini_structured_analyzer.py, 
advanced_llm_system.py, and demo_gemini.py
"""

import json
import os
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import hashlib

# LLM imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class AnalysisMode(Enum):
    """Analysis modes"""
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    STRUCTURED = "structured"
    CUSTOM = "custom"


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    provider: Provider = Provider.GEMINI
    mode: AnalysisMode = AnalysisMode.QUICK
    model_name: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_ttl: int = 3600
    rate_limit: int = 60  # requests per minute
    custom_prompt: Optional[str] = None
    response_format: str = "json"
    dimensions: List[str] = field(default_factory=lambda: [
        "linguistic", "cognitive", "emotional", "creativity", "personality"
    ])


@dataclass
class AnalysisResult:
    """Analysis result container"""
    text: str
    ai_probability: float
    prediction: str
    confidence: Dict[str, Any]
    key_indicators: List[str]
    evidence: List[str]
    reasoning: str
    dimensions: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    cached: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class Cache:
    """Simple in-memory cache for analysis results"""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[AnalysisResult]:
        """Get cached result"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: AnalysisResult) -> None:
        """Cache result"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests: int = 60, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = []
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove old requests outside window
        self.requests = [t for t in self.requests if now - t < self.window]
        
        if len(self.requests) >= self.max_requests:
            # Need to wait
            oldest = self.requests[0]
            wait_time = self.window - (now - oldest) + 0.1
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self.requests.append(now)


class ProviderStrategy(ABC):
    """Abstract strategy for different LLM providers"""
    
    @abstractmethod
    async def analyze(self, text: str, config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze text and return raw response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    @abstractmethod
    def get_prompt(self, text: str, mode: AnalysisMode) -> str:
        """Get analysis prompt for the provider"""
        pass


class GeminiProvider(ProviderStrategy):
    """Google Gemini provider"""
    
    def __init__(self):
        self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return GEMINI_AVAILABLE
    
    async def analyze(self, text: str, config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze using Gemini"""
        if not self.is_available():
            raise RuntimeError("Gemini not available")
        
        if not config.api_key:
            raise ValueError("API key required for Gemini")
        
        # Configure Gemini
        genai.configure(api_key=config.api_key)
        
        # Initialize model if needed
        if self.model is None:
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                response_mime_type="application/json" if config.response_format == "json" else "text/plain"
            )
            self.model = genai.GenerativeModel(
                model_name=config.model_name,
                generation_config=generation_config
            )
        
        # Get prompt
        prompt = config.custom_prompt or self.get_prompt(text, config.mode)
        
        # Make API call
        response = await self.model.generate_content_async(prompt)
        response_text = response.candidates[0].content.parts[0].text
        
        # Parse response
        if config.response_format == "json":
            return json.loads(response_text)
        else:
            return {"response": response_text}
    
    def get_prompt(self, text: str, mode: AnalysisMode) -> str:
        """Get Gemini-specific prompt"""
        if mode == AnalysisMode.QUICK:
            return f"""Analyze this text for AI generation markers. Return JSON:

TEXT: "{text}"

Return:
{{
    "ai_probability": 0.0-1.0,
    "prediction": "ai" or "human",
    "confidence": {{"value": 0.0-1.0, "level": "low/medium/high"}},
    "key_indicators": ["list of patterns found"],
    "evidence": ["specific text evidence"],
    "reasoning": "brief explanation"
}}"""
        
        elif mode == AnalysisMode.COMPREHENSIVE:
            return f"""Perform comprehensive AI detection analysis. Return JSON:

TEXT: "{text}"

Analyze across dimensions and return:
{{
    "ai_probability": 0.0-1.0,
    "prediction": "ai" or "human",
    "overall_confidence": {{"value": 0.0-1.0, "level": "low/medium/high"}},
    "dimensions": {{
        "linguistic": {{"score": 0.0-1.0, "patterns": []}},
        "cognitive": {{"score": 0.0-1.0, "consistency": 0.0-1.0}},
        "emotional": {{"score": 0.0-1.0, "authenticity": 0.0-1.0}},
        "creativity": {{"score": 0.0-1.0, "originality": 0.0-1.0}},
        "personality": {{"score": 0.0-1.0, "consistency": 0.0-1.0}}
    }},
    "key_indicators": ["main patterns"],
    "evidence": ["quoted text"],
    "reasoning": "detailed explanation"
}}"""
        
        elif mode == AnalysisMode.STRUCTURED:
            return f"""Analyze for AI markers with structured output. Return JSON:

TEXT: "{text}"

Provide structured analysis:
{{
    "classification": {{
        "label": "ai" or "human",
        "confidence": 0.0-1.0
    }},
    "patterns": {{
        "hedging_language": {{"present": true/false, "examples": [], "score": 0.0-1.0}},
        "balanced_presentation": {{"present": true/false, "examples": [], "score": 0.0-1.0}},
        "formal_transitions": {{"present": true/false, "examples": [], "score": 0.0-1.0}},
        "meta_commentary": {{"present": true/false, "examples": [], "score": 0.0-1.0}}
    }},
    "human_indicators": {{
        "casual_language": {{"present": true/false, "examples": [], "score": 0.0-1.0}},
        "emotional_authenticity": {{"present": true/false, "examples": [], "score": 0.0-1.0}},
        "personal_voice": {{"present": true/false, "examples": [], "score": 0.0-1.0}}
    }},
    "overall_assessment": "detailed reasoning"
}}"""
        
        else:
            return f'Analyze this text and determine if it is AI-generated: "{text}"'


class OpenAIProvider(ProviderStrategy):
    """OpenAI provider"""
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return OPENAI_AVAILABLE
    
    async def analyze(self, text: str, config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze using OpenAI"""
        if not self.is_available():
            raise RuntimeError("OpenAI not available")
        
        if not config.api_key:
            raise ValueError("API key required for OpenAI")
        
        openai.api_key = config.api_key
        
        prompt = config.custom_prompt or self.get_prompt(text, config.mode)
        
        response = await openai.ChatCompletion.acreate(
            model=config.model_name or "gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI detection expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        response_text = response.choices[0].message.content
        
        if config.response_format == "json":
            return json.loads(response_text)
        else:
            return {"response": response_text}
    
    def get_prompt(self, text: str, mode: AnalysisMode) -> str:
        """Get OpenAI-specific prompt"""
        # Similar to Gemini prompts
        return GeminiProvider().get_prompt(text, mode)


class UnifiedAnalyzer:
    """
    Unified analyzer that consolidates all LLM analysis functionality
    Replaces: llm_analyzer.py, gemini_structured_analyzer.py, 
              advanced_llm_system.py, demo_gemini.py
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.provider = self._get_provider()
        self.cache = Cache(ttl=self.config.cache_ttl) if self.config.cache_enabled else None
        self.rate_limiter = RateLimiter(max_requests=self.config.rate_limit)
        self.analysis_history = []
    
    def _get_provider(self) -> ProviderStrategy:
        """Get provider strategy based on configuration"""
        if self.config.provider == Provider.GEMINI:
            return GeminiProvider()
        elif self.config.provider == Provider.OPENAI:
            return OpenAIProvider()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.config.provider.value}_{self.config.mode.value}_{text_hash}"
    
    async def analyze(self, text: str, mode: Optional[AnalysisMode] = None) -> AnalysisResult:
        """
        Analyze text for AI generation markers
        
        Args:
            text: Text to analyze
            mode: Optional override for analysis mode
            
        Returns:
            Analysis result
        """
        start_time = time.time()
        
        # Use provided mode or default
        analysis_mode = mode or self.config.mode
        
        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(text)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Using cached result")
                cached_result.cached = True
                return cached_result
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Retry logic
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                # Make API call
                raw_response = await self.provider.analyze(text, self.config)
                
                # Parse response into result
                result = self._parse_response(text, raw_response, analysis_mode)
                
                # Add metadata
                result.processing_time = time.time() - start_time
                result.provider_used = self.config.provider.value
                result.model_used = self.config.model_name
                
                # Cache result
                if self.cache:
                    self.cache.set(cache_key, result)
                
                # Add to history
                self.analysis_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'text_length': len(text),
                    'mode': analysis_mode.value,
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                })
                
                logger.info(f"Analysis complete in {result.processing_time:.2f}s")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        # All retries failed
        raise RuntimeError(f"Analysis failed after {self.config.retry_attempts} attempts: {last_error}")
    
    def _parse_response(self, text: str, response: Dict[str, Any], mode: AnalysisMode) -> AnalysisResult:
        """Parse raw response into AnalysisResult"""
        
        # Handle different response formats based on mode
        if mode == AnalysisMode.QUICK:
            return AnalysisResult(
                text=text,
                ai_probability=response.get('ai_probability', 0.5),
                prediction=response.get('prediction', 'unknown'),
                confidence=response.get('confidence', {'value': 0.5, 'level': 'low'}),
                key_indicators=response.get('key_indicators', []),
                evidence=response.get('evidence', []),
                reasoning=response.get('reasoning', ''),
                metadata={'mode': mode.value}
            )
        
        elif mode == AnalysisMode.COMPREHENSIVE:
            return AnalysisResult(
                text=text,
                ai_probability=response.get('ai_probability', 0.5),
                prediction=response.get('prediction', 'unknown'),
                confidence=response.get('overall_confidence', {'value': 0.5, 'level': 'low'}),
                key_indicators=response.get('key_indicators', []),
                evidence=response.get('evidence', []),
                reasoning=response.get('reasoning', ''),
                dimensions=response.get('dimensions', {}),
                metadata={'mode': mode.value}
            )
        
        elif mode == AnalysisMode.STRUCTURED:
            classification = response.get('classification', {})
            patterns = response.get('patterns', {})
            
            # Extract indicators from patterns
            key_indicators = []
            evidence = []
            
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.get('present'):
                    key_indicators.append(pattern_name)
                    evidence.extend(pattern_data.get('examples', []))
            
            return AnalysisResult(
                text=text,
                ai_probability=classification.get('confidence', 0.5),
                prediction=classification.get('label', 'unknown'),
                confidence={'value': classification.get('confidence', 0.5), 
                          'level': self._confidence_level(classification.get('confidence', 0.5))},
                key_indicators=key_indicators,
                evidence=evidence,
                reasoning=response.get('overall_assessment', ''),
                dimensions={'patterns': patterns, 'human_indicators': response.get('human_indicators', {})},
                metadata={'mode': mode.value}
            )
        
        else:
            # Custom or fallback
            return AnalysisResult(
                text=text,
                ai_probability=0.5,
                prediction='unknown',
                confidence={'value': 0.5, 'level': 'low'},
                key_indicators=[],
                evidence=[],
                reasoning=response.get('response', ''),
                metadata={'mode': mode.value, 'raw_response': response}
            )
    
    def _confidence_level(self, value: float) -> str:
        """Convert confidence value to level"""
        if value >= 0.8:
            return 'high'
        elif value >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    async def batch_analyze(self, texts: List[str], mode: Optional[AnalysisMode] = None) -> List[AnalysisResult]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts to analyze
            mode: Optional override for analysis mode
            
        Returns:
            List of analysis results
        """
        results = []
        for text in texts:
            try:
                result = await self.analyze(text, mode)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze text: {e}")
                # Add error result
                results.append(AnalysisResult(
                    text=text,
                    ai_probability=0.5,
                    prediction='error',
                    confidence={'value': 0, 'level': 'none'},
                    key_indicators=[],
                    evidence=[],
                    reasoning=f"Analysis failed: {e}",
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.analysis_history:
            return {'total_analyses': 0}
        
        predictions = [h['prediction'] for h in self.analysis_history]
        processing_times = [h['processing_time'] for h in self.analysis_history]
        
        return {
            'total_analyses': len(self.analysis_history),
            'predictions': {
                'ai': predictions.count('ai'),
                'human': predictions.count('human'),
                'unknown': predictions.count('unknown')
            },
            'average_processing_time': sum(processing_times) / len(processing_times),
            'cache_size': self.cache.size() if self.cache else 0,
            'provider': self.config.provider.value,
            'model': self.config.model_name
        }
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Recreate provider if needed
        if 'provider' in kwargs:
            self.provider = self._get_provider()
        
        logger.info(f"Configuration updated: {kwargs}")


# Factory function for convenience
def create_analyzer(provider: str = 'gemini', mode: str = 'quick', **kwargs) -> UnifiedAnalyzer:
    """Create an analyzer instance"""
    provider_enum = Provider(provider)
    mode_enum = AnalysisMode(mode)
    config = AnalysisConfig(provider=provider_enum, mode=mode_enum, **kwargs)
    return UnifiedAnalyzer(config)


# Example usage
async def demo():
    """Demo the unified analyzer"""
    # Create analyzer
    analyzer = create_analyzer(
        provider='gemini',
        mode='comprehensive',
        api_key=os.getenv('GEMINI_API_KEY')
    )
    
    # Sample text
    text = "While AI continues to evolve, it's important to note both advantages and disadvantages."
    
    # Analyze
    result = await analyzer.analyze(text)
    print(f"Result: {result.to_json()}")
    
    # Get statistics
    stats = analyzer.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo())