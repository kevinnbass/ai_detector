"""
GEMINI-POWERED STRUCTURED AI DETECTION SYSTEM
=============================================

Uses Google's Gemini SDK for comprehensive, quantified analysis with structured JSON output.
Every dimension is quantified with numerical scores and confidence intervals.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import time

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")

# ============================================================================
# STRUCTURED DATA CLASSES FOR JSON OUTPUT
# ============================================================================

@dataclass
class ConfidenceScore:
    """Structured confidence scoring"""
    value: float  # 0.0 to 1.0
    certainty: str  # "very_low", "low", "medium", "high", "very_high"
    reliability: float  # 0.0 to 1.0

@dataclass
class DimensionScore:
    """Individual dimension analysis score"""
    score: float  # 0.0 (human) to 1.0 (AI)
    confidence: ConfidenceScore
    indicators: List[str]
    evidence: List[str]
    sub_scores: Dict[str, float]

@dataclass
class CognitiveLoadAnalysis:
    """Cognitive load pattern analysis"""
    overall_load: DimensionScore
    complexity_distribution: Dict[str, float]  # even vs variable
    processing_depth: Dict[str, float]  # surface vs deep
    effort_indicators: Dict[str, float]  # struggle vs effortless
    attention_patterns: Dict[str, float]  # focused vs scattered
    mental_fatigue: Dict[str, float]  # present vs absent

@dataclass
class EmotionalIntelligenceAnalysis:
    """Emotional intelligence assessment"""
    overall_eq: DimensionScore
    emotional_granularity: Dict[str, float]  # specific vs generic
    empathy_authenticity: Dict[str, float]  # genuine vs simulated
    emotional_regulation: Dict[str, float]  # human vs systematic
    vulnerability_markers: Dict[str, float]  # authentic vs calculated
    emotional_progression: Dict[str, float]  # natural vs artificial

@dataclass
class CreativityAnalysis:
    """Creativity and originality assessment"""
    overall_creativity: DimensionScore
    originality_score: Dict[str, float]  # novel vs recombined
    creative_risk_taking: Dict[str, float]  # bold vs safe
    metaphor_authenticity: Dict[str, float]  # fresh vs clichéd
    perspective_uniqueness: Dict[str, float]  # individual vs algorithmic
    artistic_vision: Dict[str, float]  # authentic vs simulated

@dataclass
class LinguisticAnalysis:
    """Linguistic pattern assessment"""
    overall_linguistic: DimensionScore
    hedging_frequency: Dict[str, float]  # excessive vs natural
    contrast_rhetoric: Dict[str, float]  # systematic vs occasional
    formal_register: Dict[str, float]  # inappropriate vs appropriate
    qualifier_usage: Dict[str, float]  # excessive vs natural
    structured_presentation: Dict[str, float]  # systematic vs organic

@dataclass
class DomainExpertiseAnalysis:
    """Domain knowledge authenticity"""
    overall_expertise: DimensionScore
    knowledge_depth: Dict[str, float]  # surface vs deep
    practical_experience: Dict[str, float]  # theoretical vs hands-on
    domain_language: Dict[str, float]  # Wikipedia-like vs authentic
    edge_case_awareness: Dict[str, float]  # novice vs expert
    contextual_wisdom: Dict[str, float]  # missing vs present

@dataclass
class PersonalityAnalysis:
    """Personality consistency and authenticity"""
    overall_personality: DimensionScore
    trait_consistency: Dict[str, float]  # stable vs variable
    voice_authenticity: Dict[str, float]  # genuine vs simulated
    big_five_scores: Dict[str, float]  # O, C, E, A, N
    personality_quirks: Dict[str, float]  # unique vs uniform
    value_coherence: Dict[str, float]  # consistent vs algorithmic

@dataclass
class TemporalAnalysis:
    """Temporal reasoning and time-based patterns"""
    overall_temporal: DimensionScore
    time_perception: Dict[str, float]  # human vs AI understanding
    sequence_reasoning: Dict[str, float]  # natural vs algorithmic
    memory_integration: Dict[str, float]  # episodic vs semantic
    causal_understanding: Dict[str, float]  # experiential vs logical
    temporal_consistency: Dict[str, float]  # natural vs systematic

@dataclass
class CulturalAnalysis:
    """Cultural authenticity and social context"""
    overall_cultural: DimensionScore
    cultural_fluency: Dict[str, float]  # authentic vs learned
    social_embeddedness: Dict[str, float]  # lived vs algorithmic
    generational_markers: Dict[str, float]  # authentic vs stereotypical
    geographic_authenticity: Dict[str, float]  # genuine vs researched
    identity_coherence: Dict[str, float]  # consistent vs simulated

@dataclass
class DeceptionAnalysis:
    """Deception and manipulation detection"""
    overall_deception: DimensionScore
    truth_markers: Dict[str, float]  # genuine vs fabricated
    manipulation_indicators: Dict[str, float]  # present vs absent
    authenticity_assessment: Dict[str, float]  # authentic vs calculated
    emotional_manipulation: Dict[str, float]  # genuine vs systematic
    information_integrity: Dict[str, float]  # honest vs strategic

@dataclass
class MetacognitiveAnalysis:
    """Metacognitive awareness and self-reflection"""
    overall_metacognitive: DimensionScore
    self_awareness: Dict[str, float]  # genuine vs simulated
    uncertainty_acknowledgment: Dict[str, float]  # human vs AI patterns
    bias_recognition: Dict[str, float]  # authentic vs systematic
    learning_indicators: Dict[str, float]  # genuine vs programmed
    cognitive_monitoring: Dict[str, float]  # natural vs artificial

@dataclass
class ComprehensiveAnalysisResult:
    """Complete structured analysis result"""
    # Metadata
    analysis_id: str
    timestamp: str
    text_length: int
    model_used: str
    processing_time: float
    
    # Overall prediction
    ai_probability: float  # 0.0 to 1.0
    prediction: str  # "human" or "ai"
    overall_confidence: ConfidenceScore
    
    # Individual dimension analyses
    cognitive_load: CognitiveLoadAnalysis
    emotional_intelligence: EmotionalIntelligenceAnalysis
    creativity: CreativityAnalysis
    linguistic: LinguisticAnalysis
    domain_expertise: DomainExpertiseAnalysis
    personality: PersonalityAnalysis
    temporal: TemporalAnalysis
    cultural: CulturalAnalysis
    deception: DeceptionAnalysis
    metacognitive: MetacognitiveAnalysis
    
    # Cross-dimensional analysis
    dimension_correlations: Dict[str, float]
    consistency_scores: Dict[str, float]
    contradiction_indicators: List[str]
    
    # Ensemble metrics
    ensemble_agreement: float
    prediction_stability: float
    confidence_calibration: float
    
    # Explanatory information
    key_indicators: List[str]
    human_markers: List[str]
    ai_markers: List[str]
    uncertainty_sources: List[str]
    recommendation: str


# ============================================================================
# GEMINI STRUCTURED ANALYZER
# ============================================================================

class GeminiStructuredAnalyzer:
    """
    Comprehensive AI detection using Gemini with structured JSON output
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        genai.configure(api_key=api_key)
        
        # Configure model with structured output optimization
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low for consistency
            top_p=0.8,
            top_k=20,
            max_output_tokens=4000,
            response_mime_type="application/json"  # Force JSON output
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.model_name = model_name
        self.rate_limit = 1.0  # Seconds between calls
        
    async def analyze_cognitive_load(self, text: str) -> CognitiveLoadAnalysis:
        """Analyze cognitive load patterns with structured output"""
        
        prompt = f"""Analyze the cognitive load and mental processing patterns in this text. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human-like, 1.0 = AI-like):

{{
    "overall_load": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["list", "of", "specific", "indicators"],
        "evidence": ["specific", "text", "evidence"],
        "sub_scores": {{
            "load_consistency": 0.0-1.0,
            "complexity_distribution": 0.0-1.0,
            "processing_efficiency": 0.0-1.0
        }}
    }},
    "complexity_distribution": {{
        "evenness": 0.0-1.0,
        "variability": 0.0-1.0,
        "natural_peaks": 0.0-1.0
    }},
    "processing_depth": {{
        "surface_indicators": 0.0-1.0,
        "deep_thinking_markers": 0.0-1.0,
        "analytical_depth": 0.0-1.0
    }},
    "effort_indicators": {{
        "struggle_markers": 0.0-1.0,
        "effortless_generation": 0.0-1.0,
        "cognitive_strain": 0.0-1.0
    }},
    "attention_patterns": {{
        "focus_consistency": 0.0-1.0,
        "attention_wandering": 0.0-1.0,
        "selective_focus": 0.0-1.0
    }},
    "mental_fatigue": {{
        "fatigue_indicators": 0.0-1.0,
        "energy_consistency": 0.0-1.0,
        "endurance_markers": 0.0-1.0
    }}
}}

Analyze: Does this text show natural human cognitive load variation or systematic AI processing consistency?"""
        
        response = await self._call_gemini(prompt)
        return self._parse_cognitive_load(response)
    
    async def analyze_emotional_intelligence(self, text: str) -> EmotionalIntelligenceAnalysis:
        """Analyze emotional intelligence with quantified scores"""
        
        prompt = f"""Analyze emotional intelligence and authentic emotional patterns. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human-like, 1.0 = AI-like):

{{
    "overall_eq": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["emotional", "intelligence", "markers"],
        "evidence": ["specific", "emotional", "evidence"],
        "sub_scores": {{
            "emotional_authenticity": 0.0-1.0,
            "empathy_indicators": 0.0-1.0,
            "emotional_complexity": 0.0-1.0
        }}
    }},
    "emotional_granularity": {{
        "specificity": 0.0-1.0,
        "emotional_vocabulary": 0.0-1.0,
        "nuance_expression": 0.0-1.0
    }},
    "empathy_authenticity": {{
        "perspective_taking": 0.0-1.0,
        "compassion_markers": 0.0-1.0,
        "emotional_resonance": 0.0-1.0
    }},
    "emotional_regulation": {{
        "natural_regulation": 0.0-1.0,
        "systematic_control": 0.0-1.0,
        "emotional_balance": 0.0-1.0
    }},
    "vulnerability_markers": {{
        "authentic_openness": 0.0-1.0,
        "calculated_vulnerability": 0.0-1.0,
        "emotional_risk_taking": 0.0-1.0
    }},
    "emotional_progression": {{
        "natural_flow": 0.0-1.0,
        "artificial_transitions": 0.0-1.0,
        "emotional_coherence": 0.0-1.0
    }}
}}

Analyze: Does this show genuine human emotional complexity or systematic AI emotional patterns?"""
        
        response = await self._call_gemini(prompt)
        return self._parse_emotional_intelligence(response)
    
    async def analyze_creativity(self, text: str) -> CreativityAnalysis:
        """Analyze creativity and originality patterns"""
        
        prompt = f"""Analyze creativity, originality, and innovative thinking. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human creative, 1.0 = AI systematic):

{{
    "overall_creativity": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["creative", "thinking", "markers"],
        "evidence": ["specific", "creative", "evidence"],
        "sub_scores": {{
            "originality": 0.0-1.0,
            "innovation": 0.0-1.0,
            "creative_risk": 0.0-1.0
        }}
    }},
    "originality_score": {{
        "novel_ideas": 0.0-1.0,
        "unique_perspectives": 0.0-1.0,
        "recombination_patterns": 0.0-1.0
    }},
    "creative_risk_taking": {{
        "bold_choices": 0.0-1.0,
        "safe_patterns": 0.0-1.0,
        "experimental_elements": 0.0-1.0
    }},
    "metaphor_authenticity": {{
        "fresh_metaphors": 0.0-1.0,
        "cliched_comparisons": 0.0-1.0,
        "metaphorical_consistency": 0.0-1.0
    }},
    "perspective_uniqueness": {{
        "individual_viewpoint": 0.0-1.0,
        "algorithmic_perspective": 0.0-1.0,
        "personal_voice": 0.0-1.0
    }},
    "artistic_vision": {{
        "aesthetic_sense": 0.0-1.0,
        "creative_coherence": 0.0-1.0,
        "artistic_authenticity": 0.0-1.0
    }}
}}

Analyze: Does this show genuine human creativity or systematic AI recombination patterns?"""
        
        response = await self._call_gemini(prompt)
        return self._parse_creativity(response)
    
    async def analyze_linguistic_patterns(self, text: str) -> LinguisticAnalysis:
        """Analyze linguistic patterns specific to AI generation"""
        
        prompt = f"""Analyze linguistic patterns for AI generation markers. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human patterns, 1.0 = AI patterns):

{{
    "overall_linguistic": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["linguistic", "pattern", "markers"],
        "evidence": ["specific", "linguistic", "evidence"],
        "sub_scores": {{
            "pattern_systematicity": 0.0-1.0,
            "linguistic_consistency": 0.0-1.0,
            "natural_variation": 0.0-1.0
        }}
    }},
    "hedging_frequency": {{
        "hedge_words": 0.0-1.0,
        "uncertainty_markers": 0.0-1.0,
        "qualification_patterns": 0.0-1.0
    }},
    "contrast_rhetoric": {{
        "not_but_constructions": 0.0-1.0,
        "systematic_contrasts": 0.0-1.0,
        "balanced_presentations": 0.0-1.0
    }},
    "formal_register": {{
        "context_appropriateness": 0.0-1.0,
        "formal_in_casual": 0.0-1.0,
        "register_consistency": 0.0-1.0
    }},
    "qualifier_usage": {{
        "meta_commentary": 0.0-1.0,
        "importance_phrases": 0.0-1.0,
        "consideration_markers": 0.0-1.0
    }},
    "structured_presentation": {{
        "enumeration_patterns": 0.0-1.0,
        "logical_flow": 0.0-1.0,
        "systematic_organization": 0.0-1.0
    }}
}}

Focus on GPT-4o specific patterns: hedging, contrast rhetoric, qualifiers, formal language."""
        
        response = await self._call_gemini(prompt)
        return self._parse_linguistic(response)
    
    async def analyze_domain_expertise(self, text: str, domain: str = None) -> DomainExpertiseAnalysis:
        """Analyze domain knowledge authenticity"""
        
        # Auto-detect domain if not provided
        if not domain:
            domain = await self._detect_domain(text)
        
        prompt = f"""Analyze domain expertise authenticity in the {domain} field. Return ONLY valid JSON.

TEXT: "{text}"
DOMAIN: {domain}

Return this exact JSON structure with numerical scores (0.0 = authentic expertise, 1.0 = AI knowledge):

{{
    "overall_expertise": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["domain", "expertise", "markers"],
        "evidence": ["specific", "knowledge", "evidence"],
        "sub_scores": {{
            "knowledge_authenticity": 0.0-1.0,
            "practical_understanding": 0.0-1.0,
            "expert_insight": 0.0-1.0
        }}
    }},
    "knowledge_depth": {{
        "surface_knowledge": 0.0-1.0,
        "deep_understanding": 0.0-1.0,
        "conceptual_mastery": 0.0-1.0
    }},
    "practical_experience": {{
        "hands_on_indicators": 0.0-1.0,
        "theoretical_knowledge": 0.0-1.0,
        "real_world_context": 0.0-1.0
    }},
    "domain_language": {{
        "technical_accuracy": 0.0-1.0,
        "jargon_appropriateness": 0.0-1.0,
        "insider_perspective": 0.0-1.0
    }},
    "edge_case_awareness": {{
        "nuanced_understanding": 0.0-1.0,
        "common_misconceptions": 0.0-1.0,
        "expert_level_insights": 0.0-1.0
    }},
    "contextual_wisdom": {{
        "situational_awareness": 0.0-1.0,
        "practical_wisdom": 0.0-1.0,
        "experiential_knowledge": 0.0-1.0
    }}
}}

Compare: Real expert experience vs encyclopedic AI knowledge patterns."""
        
        response = await self._call_gemini(prompt)
        return self._parse_domain_expertise(response)
    
    async def analyze_personality(self, text: str) -> PersonalityAnalysis:
        """Analyze personality consistency and authenticity"""
        
        prompt = f"""Analyze personality patterns and authenticity. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human personality, 1.0 = AI personality):

{{
    "overall_personality": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["personality", "authenticity", "markers"],
        "evidence": ["specific", "personality", "evidence"],
        "sub_scores": {{
            "personality_coherence": 0.0-1.0,
            "voice_authenticity": 0.0-1.0,
            "individual_character": 0.0-1.0
        }}
    }},
    "trait_consistency": {{
        "stable_patterns": 0.0-1.0,
        "natural_variation": 0.0-1.0,
        "personality_coherence": 0.0-1.0
    }},
    "voice_authenticity": {{
        "genuine_voice": 0.0-1.0,
        "simulated_personality": 0.0-1.0,
        "personal_style": 0.0-1.0
    }},
    "big_five_scores": {{
        "openness": 0.0-1.0,
        "conscientiousness": 0.0-1.0,
        "extraversion": 0.0-1.0,
        "agreeableness": 0.0-1.0,
        "neuroticism": 0.0-1.0
    }},
    "personality_quirks": {{
        "individual_uniqueness": 0.0-1.0,
        "systematic_uniformity": 0.0-1.0,
        "personal_idiosyncrasies": 0.0-1.0
    }},
    "value_coherence": {{
        "consistent_values": 0.0-1.0,
        "algorithmic_optimization": 0.0-1.0,
        "personal_beliefs": 0.0-1.0
    }}
}}

Analyze: Genuine human personality vs systematic AI personality simulation."""
        
        response = await self._call_gemini(prompt)
        return self._parse_personality(response)
    
    async def analyze_temporal_reasoning(self, text: str) -> TemporalAnalysis:
        """Analyze temporal reasoning and time-based cognition"""
        
        prompt = f"""Analyze temporal reasoning and time-based thinking. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human temporal, 1.0 = AI temporal):

{{
    "overall_temporal": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["temporal", "reasoning", "markers"],
        "evidence": ["specific", "temporal", "evidence"],
        "sub_scores": {{
            "time_understanding": 0.0-1.0,
            "temporal_consistency": 0.0-1.0,
            "experiential_time": 0.0-1.0
        }}
    }},
    "time_perception": {{
        "experiential_time": 0.0-1.0,
        "abstract_time": 0.0-1.0,
        "temporal_authenticity": 0.0-1.0
    }},
    "sequence_reasoning": {{
        "natural_ordering": 0.0-1.0,
        "algorithmic_sequence": 0.0-1.0,
        "causal_flow": 0.0-1.0
    }},
    "memory_integration": {{
        "episodic_markers": 0.0-1.0,
        "semantic_patterns": 0.0-1.0,
        "temporal_context": 0.0-1.0
    }},
    "causal_understanding": {{
        "experiential_causation": 0.0-1.0,
        "logical_causation": 0.0-1.0,
        "temporal_logic": 0.0-1.0
    }},
    "temporal_consistency": {{
        "natural_inconsistency": 0.0-1.0,
        "systematic_consistency": 0.0-1.0,
        "human_temporal_gaps": 0.0-1.0
    }}
}}

Compare: Human experiential time understanding vs AI systematic temporal logic."""
        
        response = await self._call_gemini(prompt)
        return self._parse_temporal(response)
    
    async def analyze_cultural_authenticity(self, text: str) -> CulturalAnalysis:
        """Analyze cultural authenticity and social embeddedness"""
        
        prompt = f"""Analyze cultural authenticity and social context understanding. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = authentic cultural, 1.0 = AI learned):

{{
    "overall_cultural": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["cultural", "authenticity", "markers"],
        "evidence": ["specific", "cultural", "evidence"],
        "sub_scores": {{
            "cultural_embeddedness": 0.0-1.0,
            "social_authenticity": 0.0-1.0,
            "lived_experience": 0.0-1.0
        }}
    }},
    "cultural_fluency": {{
        "authentic_knowledge": 0.0-1.0,
        "learned_patterns": 0.0-1.0,
        "cultural_intuition": 0.0-1.0
    }},
    "social_embeddedness": {{
        "community_markers": 0.0-1.0,
        "social_context": 0.0-1.0,
        "relational_understanding": 0.0-1.0
    }},
    "generational_markers": {{
        "age_appropriate": 0.0-1.0,
        "generational_context": 0.0-1.0,
        "era_authenticity": 0.0-1.0
    }},
    "geographic_authenticity": {{
        "location_knowledge": 0.0-1.0,
        "regional_markers": 0.0-1.0,
        "geographic_context": 0.0-1.0
    }},
    "identity_coherence": {{
        "identity_consistency": 0.0-1.0,
        "authentic_self": 0.0-1.0,
        "cultural_identity": 0.0-1.0
    }}
}}

Compare: Lived cultural experience vs learned cultural patterns."""
        
        response = await self._call_gemini(prompt)
        return self._parse_cultural(response)
    
    async def analyze_deception_patterns(self, text: str) -> DeceptionAnalysis:
        """Analyze deception and manipulation indicators"""
        
        prompt = f"""Analyze potential deception and manipulation patterns. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = truthful/authentic, 1.0 = deceptive/manipulative):

{{
    "overall_deception": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["deception", "authenticity", "markers"],
        "evidence": ["specific", "deception", "evidence"],
        "sub_scores": {{
            "truthfulness": 0.0-1.0,
            "manipulation_presence": 0.0-1.0,
            "authenticity_level": 0.0-1.0
        }}
    }},
    "truth_markers": {{
        "genuine_information": 0.0-1.0,
        "fabricated_content": 0.0-1.0,
        "fact_consistency": 0.0-1.0
    }},
    "manipulation_indicators": {{
        "persuasion_techniques": 0.0-1.0,
        "emotional_manipulation": 0.0-1.0,
        "cognitive_manipulation": 0.0-1.0
    }},
    "authenticity_assessment": {{
        "genuine_communication": 0.0-1.0,
        "calculated_responses": 0.0-1.0,
        "spontaneous_expression": 0.0-1.0
    }},
    "emotional_manipulation": {{
        "authentic_emotion": 0.0-1.0,
        "manufactured_emotion": 0.0-1.0,
        "emotional_authenticity": 0.0-1.0
    }},
    "information_integrity": {{
        "honest_disclosure": 0.0-1.0,
        "strategic_omission": 0.0-1.0,
        "information_accuracy": 0.0-1.0
    }}
}}

Note: AI may show systematic manipulation patterns from training optimization."""
        
        response = await self._call_gemini(prompt)
        return self._parse_deception(response)
    
    async def analyze_metacognitive_patterns(self, text: str) -> MetacognitiveAnalysis:
        """Analyze metacognitive awareness and self-reflection"""
        
        prompt = f"""Analyze metacognitive awareness and thinking-about-thinking patterns. Return ONLY valid JSON.

TEXT: "{text}"

Return this exact JSON structure with numerical scores (0.0 = human metacognition, 1.0 = AI metacognition):

{{
    "overall_metacognitive": {{
        "score": 0.0-1.0,
        "confidence": {{
            "value": 0.0-1.0,
            "certainty": "very_low/low/medium/high/very_high",
            "reliability": 0.0-1.0
        }},
        "indicators": ["metacognitive", "awareness", "markers"],
        "evidence": ["specific", "metacognitive", "evidence"],
        "sub_scores": {{
            "self_awareness": 0.0-1.0,
            "cognitive_monitoring": 0.0-1.0,
            "meta_learning": 0.0-1.0
        }}
    }},
    "self_awareness": {{
        "genuine_introspection": 0.0-1.0,
        "simulated_self_reflection": 0.0-1.0,
        "self_knowledge_accuracy": 0.0-1.0
    }},
    "uncertainty_acknowledgment": {{
        "authentic_uncertainty": 0.0-1.0,
        "systematic_hedging": 0.0-1.0,
        "confidence_calibration": 0.0-1.0
    }},
    "bias_recognition": {{
        "bias_awareness": 0.0-1.0,
        "systematic_bias_patterns": 0.0-1.0,
        "cognitive_humility": 0.0-1.0
    }},
    "learning_indicators": {{
        "genuine_learning": 0.0-1.0,
        "programmed_adaptation": 0.0-1.0,
        "knowledge_growth": 0.0-1.0
    }},
    "cognitive_monitoring": {{
        "natural_monitoring": 0.0-1.0,
        "artificial_self_tracking": 0.0-1.0,
        "metacognitive_strategies": 0.0-1.0
    }}
}}

Compare: Genuine human self-awareness vs systematic AI self-monitoring."""
        
        response = await self._call_gemini(prompt)
        return self._parse_metacognitive(response)
    
    async def comprehensive_analysis(self, text: str, domain: str = None) -> ComprehensiveAnalysisResult:
        """Run complete comprehensive analysis with structured output"""
        
        start_time = time.time()
        analysis_id = f"analysis_{int(time.time())}"
        
        print(f"🧠 Running comprehensive Gemini analysis (ID: {analysis_id})...")
        
        # Run all analyses in parallel
        try:
            cognitive_task = self.analyze_cognitive_load(text)
            emotional_task = self.analyze_emotional_intelligence(text)  
            creativity_task = self.analyze_creativity(text)
            linguistic_task = self.analyze_linguistic_patterns(text)
            domain_task = self.analyze_domain_expertise(text, domain)
            personality_task = self.analyze_personality(text)
            temporal_task = self.analyze_temporal_reasoning(text)
            cultural_task = self.analyze_cultural_authenticity(text)
            deception_task = self.analyze_deception_patterns(text)
            metacognitive_task = self.analyze_metacognitive_patterns(text)
            
            # Wait for all analyses to complete
            results = await asyncio.gather(
                cognitive_task, emotional_task, creativity_task, linguistic_task,
                domain_task, personality_task, temporal_task, cultural_task, 
                deception_task, metacognitive_task,
                return_exceptions=True
            )
            
            # Handle any exceptions
            analyses = {}
            analysis_names = [
                "cognitive", "emotional", "creativity", "linguistic", 
                "domain", "personality", "temporal", "cultural",
                "deception", "metacognitive"
            ]
            
            for i, result in enumerate(results):
                name = analysis_names[i]
                if isinstance(result, Exception):
                    print(f"❌ Error in {name} analysis: {result}")
                    analyses[name] = self._create_default_analysis()
                else:
                    analyses[name] = result
            
            # Create ensemble prediction
            processing_time = time.time() - start_time
            ensemble_result = self._create_ensemble_prediction(analyses, text, analysis_id, processing_time)
            
            return ensemble_result
            
        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
            return self._create_error_result(str(e), analysis_id, text, time.time() - start_time)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with structured output request"""
        try:
            response = await self.model.generate_content_async(prompt)
            
            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                raise Exception("No response generated")
                
        except Exception as e:
            raise Exception(f"Gemini API call failed: {e}")
    
    async def _detect_domain(self, text: str) -> str:
        """Auto-detect domain from text"""
        prompt = f"""Identify the primary domain/field of this text. Return only the domain name.

TEXT: "{text}"

Choose from: technology, science, business, health, education, politics, arts, sports, general

Return only the domain name, nothing else."""
        
        try:
            response = await self._call_gemini(prompt)
            domain = response.strip().lower()
            
            valid_domains = ["technology", "science", "business", "health", "education", 
                           "politics", "arts", "sports", "general"]
            
            return domain if domain in valid_domains else "general"
        except:
            return "general"
    
    def _parse_json_safely(self, response: str) -> Dict[str, Any]:
        """Safely parse JSON response from Gemini"""
        try:
            # Clean up response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            # Parse JSON
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return {"parsing_error": True, "raw_response": response}
    
    def _create_confidence_score(self, value: float, certainty: str = "medium", reliability: float = 0.8) -> ConfidenceScore:
        """Create structured confidence score"""
        return ConfidenceScore(
            value=max(0.0, min(1.0, value)),
            certainty=certainty,
            reliability=max(0.0, min(1.0, reliability))
        )
    
    def _create_dimension_score(self, score: float, indicators: List[str] = None, 
                              evidence: List[str] = None, sub_scores: Dict[str, float] = None) -> DimensionScore:
        """Create structured dimension score"""
        return DimensionScore(
            score=max(0.0, min(1.0, score)),
            confidence=self._create_confidence_score(0.8),
            indicators=indicators or [],
            evidence=evidence or [],
            sub_scores=sub_scores or {}
        )
    
    # Parsing methods for each analysis type
    def _parse_cognitive_load(self, response: str) -> CognitiveLoadAnalysis:
        """Parse cognitive load analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_cognitive_load()
        
        try:
            overall = data.get("overall_load", {})
            return CognitiveLoadAnalysis(
                overall_load=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                complexity_distribution=data.get("complexity_distribution", {}),
                processing_depth=data.get("processing_depth", {}),
                effort_indicators=data.get("effort_indicators", {}),
                attention_patterns=data.get("attention_patterns", {}),
                mental_fatigue=data.get("mental_fatigue", {})
            )
        except Exception as e:
            print(f"❌ Error parsing cognitive load: {e}")
            return self._create_default_cognitive_load()
    
    def _parse_emotional_intelligence(self, response: str) -> EmotionalIntelligenceAnalysis:
        """Parse emotional intelligence analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_emotional_intelligence()
        
        try:
            overall = data.get("overall_eq", {})
            return EmotionalIntelligenceAnalysis(
                overall_eq=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                emotional_granularity=data.get("emotional_granularity", {}),
                empathy_authenticity=data.get("empathy_authenticity", {}),
                emotional_regulation=data.get("emotional_regulation", {}),
                vulnerability_markers=data.get("vulnerability_markers", {}),
                emotional_progression=data.get("emotional_progression", {})
            )
        except Exception as e:
            print(f"❌ Error parsing emotional intelligence: {e}")
            return self._create_default_emotional_intelligence()
    
    def _parse_creativity(self, response: str) -> CreativityAnalysis:
        """Parse creativity analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_creativity()
        
        try:
            overall = data.get("overall_creativity", {})
            return CreativityAnalysis(
                overall_creativity=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                originality_score=data.get("originality_score", {}),
                creative_risk_taking=data.get("creative_risk_taking", {}),
                metaphor_authenticity=data.get("metaphor_authenticity", {}),
                perspective_uniqueness=data.get("perspective_uniqueness", {}),
                artistic_vision=data.get("artistic_vision", {})
            )
        except Exception as e:
            print(f"❌ Error parsing creativity: {e}")
            return self._create_default_creativity()
    
    def _parse_linguistic(self, response: str) -> LinguisticAnalysis:
        """Parse linguistic analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_linguistic()
        
        try:
            overall = data.get("overall_linguistic", {})
            return LinguisticAnalysis(
                overall_linguistic=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                hedging_frequency=data.get("hedging_frequency", {}),
                contrast_rhetoric=data.get("contrast_rhetoric", {}),
                formal_register=data.get("formal_register", {}),
                qualifier_usage=data.get("qualifier_usage", {}),
                structured_presentation=data.get("structured_presentation", {})
            )
        except Exception as e:
            print(f"❌ Error parsing linguistic: {e}")
            return self._create_default_linguistic()
    
    def _parse_domain_expertise(self, response: str) -> DomainExpertiseAnalysis:
        """Parse domain expertise analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_domain_expertise()
        
        try:
            overall = data.get("overall_expertise", {})
            return DomainExpertiseAnalysis(
                overall_expertise=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                knowledge_depth=data.get("knowledge_depth", {}),
                practical_experience=data.get("practical_experience", {}),
                domain_language=data.get("domain_language", {}),
                edge_case_awareness=data.get("edge_case_awareness", {}),
                contextual_wisdom=data.get("contextual_wisdom", {})
            )
        except Exception as e:
            print(f"❌ Error parsing domain expertise: {e}")
            return self._create_default_domain_expertise()
    
    def _parse_personality(self, response: str) -> PersonalityAnalysis:
        """Parse personality analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_personality()
        
        try:
            overall = data.get("overall_personality", {})
            return PersonalityAnalysis(
                overall_personality=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                trait_consistency=data.get("trait_consistency", {}),
                voice_authenticity=data.get("voice_authenticity", {}),
                big_five_scores=data.get("big_five_scores", {}),
                personality_quirks=data.get("personality_quirks", {}),
                value_coherence=data.get("value_coherence", {})
            )
        except Exception as e:
            print(f"❌ Error parsing personality: {e}")
            return self._create_default_personality()
    
    def _parse_temporal(self, response: str) -> TemporalAnalysis:
        """Parse temporal analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_temporal()
        
        try:
            overall = data.get("overall_temporal", {})
            return TemporalAnalysis(
                overall_temporal=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                time_perception=data.get("time_perception", {}),
                sequence_reasoning=data.get("sequence_reasoning", {}),
                memory_integration=data.get("memory_integration", {}),
                causal_understanding=data.get("causal_understanding", {}),
                temporal_consistency=data.get("temporal_consistency", {})
            )
        except Exception as e:
            print(f"❌ Error parsing temporal: {e}")
            return self._create_default_temporal()
    
    def _parse_cultural(self, response: str) -> CulturalAnalysis:
        """Parse cultural analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_cultural()
        
        try:
            overall = data.get("overall_cultural", {})
            return CulturalAnalysis(
                overall_cultural=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                cultural_fluency=data.get("cultural_fluency", {}),
                social_embeddedness=data.get("social_embeddedness", {}),
                generational_markers=data.get("generational_markers", {}),
                geographic_authenticity=data.get("geographic_authenticity", {}),
                identity_coherence=data.get("identity_coherence", {})
            )
        except Exception as e:
            print(f"❌ Error parsing cultural: {e}")
            return self._create_default_cultural()
    
    def _parse_deception(self, response: str) -> DeceptionAnalysis:
        """Parse deception analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_deception()
        
        try:
            overall = data.get("overall_deception", {})
            return DeceptionAnalysis(
                overall_deception=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                truth_markers=data.get("truth_markers", {}),
                manipulation_indicators=data.get("manipulation_indicators", {}),
                authenticity_assessment=data.get("authenticity_assessment", {}),
                emotional_manipulation=data.get("emotional_manipulation", {}),
                information_integrity=data.get("information_integrity", {})
            )
        except Exception as e:
            print(f"❌ Error parsing deception: {e}")
            return self._create_default_deception()
    
    def _parse_metacognitive(self, response: str) -> MetacognitiveAnalysis:
        """Parse metacognitive analysis response"""
        data = self._parse_json_safely(response)
        
        if "parsing_error" in data:
            return self._create_default_metacognitive()
        
        try:
            overall = data.get("overall_metacognitive", {})
            return MetacognitiveAnalysis(
                overall_metacognitive=DimensionScore(
                    score=overall.get("score", 0.5),
                    confidence=ConfidenceScore(**overall.get("confidence", {"value": 0.5, "certainty": "medium", "reliability": 0.5})),
                    indicators=overall.get("indicators", []),
                    evidence=overall.get("evidence", []),
                    sub_scores=overall.get("sub_scores", {})
                ),
                self_awareness=data.get("self_awareness", {}),
                uncertainty_acknowledgment=data.get("uncertainty_acknowledgment", {}),
                bias_recognition=data.get("bias_recognition", {}),
                learning_indicators=data.get("learning_indicators", {}),
                cognitive_monitoring=data.get("cognitive_monitoring", {})
            )
        except Exception as e:
            print(f"❌ Error parsing metacognitive: {e}")
            return self._create_default_metacognitive()
    
    # Default creation methods for error handling
    def _create_default_cognitive_load(self) -> CognitiveLoadAnalysis:
        """Create default cognitive load analysis"""
        return CognitiveLoadAnalysis(
            overall_load=self._create_dimension_score(0.5),
            complexity_distribution={"evenness": 0.5, "variability": 0.5, "natural_peaks": 0.5},
            processing_depth={"surface_indicators": 0.5, "deep_thinking_markers": 0.5, "analytical_depth": 0.5},
            effort_indicators={"struggle_markers": 0.5, "effortless_generation": 0.5, "cognitive_strain": 0.5},
            attention_patterns={"focus_consistency": 0.5, "attention_wandering": 0.5, "selective_focus": 0.5},
            mental_fatigue={"fatigue_indicators": 0.5, "energy_consistency": 0.5, "endurance_markers": 0.5}
        )
    
    def _create_default_emotional_intelligence(self) -> EmotionalIntelligenceAnalysis:
        """Create default emotional intelligence analysis"""
        return EmotionalIntelligenceAnalysis(
            overall_eq=self._create_dimension_score(0.5),
            emotional_granularity={"specificity": 0.5, "emotional_vocabulary": 0.5, "nuance_expression": 0.5},
            empathy_authenticity={"perspective_taking": 0.5, "compassion_markers": 0.5, "emotional_resonance": 0.5},
            emotional_regulation={"natural_regulation": 0.5, "systematic_control": 0.5, "emotional_balance": 0.5},
            vulnerability_markers={"authentic_openness": 0.5, "calculated_vulnerability": 0.5, "emotional_risk_taking": 0.5},
            emotional_progression={"natural_flow": 0.5, "artificial_transitions": 0.5, "emotional_coherence": 0.5}
        )
    
    def _create_default_creativity(self) -> CreativityAnalysis:
        """Create default creativity analysis"""
        return CreativityAnalysis(
            overall_creativity=self._create_dimension_score(0.5),
            originality_score={"novel_ideas": 0.5, "unique_perspectives": 0.5, "recombination_patterns": 0.5},
            creative_risk_taking={"bold_choices": 0.5, "safe_patterns": 0.5, "experimental_elements": 0.5},
            metaphor_authenticity={"fresh_metaphors": 0.5, "cliched_comparisons": 0.5, "metaphorical_consistency": 0.5},
            perspective_uniqueness={"individual_viewpoint": 0.5, "algorithmic_perspective": 0.5, "personal_voice": 0.5},
            artistic_vision={"aesthetic_sense": 0.5, "creative_coherence": 0.5, "artistic_authenticity": 0.5}
        )
    
    def _create_default_linguistic(self) -> LinguisticAnalysis:
        """Create default linguistic analysis"""
        return LinguisticAnalysis(
            overall_linguistic=self._create_dimension_score(0.5),
            hedging_frequency={"hedge_words": 0.5, "uncertainty_markers": 0.5, "qualification_patterns": 0.5},
            contrast_rhetoric={"not_but_constructions": 0.5, "systematic_contrasts": 0.5, "balanced_presentations": 0.5},
            formal_register={"context_appropriateness": 0.5, "formal_in_casual": 0.5, "register_consistency": 0.5},
            qualifier_usage={"meta_commentary": 0.5, "importance_phrases": 0.5, "consideration_markers": 0.5},
            structured_presentation={"enumeration_patterns": 0.5, "logical_flow": 0.5, "systematic_organization": 0.5}
        )
    
    def _create_default_domain_expertise(self) -> DomainExpertiseAnalysis:
        """Create default domain expertise analysis"""
        return DomainExpertiseAnalysis(
            overall_expertise=self._create_dimension_score(0.5),
            knowledge_depth={"surface_knowledge": 0.5, "deep_understanding": 0.5, "conceptual_mastery": 0.5},
            practical_experience={"hands_on_indicators": 0.5, "theoretical_knowledge": 0.5, "real_world_context": 0.5},
            domain_language={"technical_accuracy": 0.5, "jargon_appropriateness": 0.5, "insider_perspective": 0.5},
            edge_case_awareness={"nuanced_understanding": 0.5, "common_misconceptions": 0.5, "expert_level_insights": 0.5},
            contextual_wisdom={"situational_awareness": 0.5, "practical_wisdom": 0.5, "experiential_knowledge": 0.5}
        )
    
    def _create_default_personality(self) -> PersonalityAnalysis:
        """Create default personality analysis"""
        return PersonalityAnalysis(
            overall_personality=self._create_dimension_score(0.5),
            trait_consistency={"stable_patterns": 0.5, "natural_variation": 0.5, "personality_coherence": 0.5},
            voice_authenticity={"genuine_voice": 0.5, "simulated_personality": 0.5, "personal_style": 0.5},
            big_five_scores={"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
            personality_quirks={"individual_uniqueness": 0.5, "systematic_uniformity": 0.5, "personal_idiosyncrasies": 0.5},
            value_coherence={"consistent_values": 0.5, "algorithmic_optimization": 0.5, "personal_beliefs": 0.5}
        )
    
    def _create_default_temporal(self) -> TemporalAnalysis:
        """Create default temporal analysis"""
        return TemporalAnalysis(
            overall_temporal=self._create_dimension_score(0.5),
            time_perception={"experiential_time": 0.5, "abstract_time": 0.5, "temporal_authenticity": 0.5},
            sequence_reasoning={"natural_ordering": 0.5, "algorithmic_sequence": 0.5, "causal_flow": 0.5},
            memory_integration={"episodic_markers": 0.5, "semantic_patterns": 0.5, "temporal_context": 0.5},
            causal_understanding={"experiential_causation": 0.5, "logical_causation": 0.5, "temporal_logic": 0.5},
            temporal_consistency={"natural_inconsistency": 0.5, "systematic_consistency": 0.5, "human_temporal_gaps": 0.5}
        )
    
    def _create_default_cultural(self) -> CulturalAnalysis:
        """Create default cultural analysis"""
        return CulturalAnalysis(
            overall_cultural=self._create_dimension_score(0.5),
            cultural_fluency={"authentic_knowledge": 0.5, "learned_patterns": 0.5, "cultural_intuition": 0.5},
            social_embeddedness={"community_markers": 0.5, "social_context": 0.5, "relational_understanding": 0.5},
            generational_markers={"age_appropriate": 0.5, "generational_context": 0.5, "era_authenticity": 0.5},
            geographic_authenticity={"location_knowledge": 0.5, "regional_markers": 0.5, "geographic_context": 0.5},
            identity_coherence={"identity_consistency": 0.5, "authentic_self": 0.5, "cultural_identity": 0.5}
        )
    
    def _create_default_deception(self) -> DeceptionAnalysis:
        """Create default deception analysis"""
        return DeceptionAnalysis(
            overall_deception=self._create_dimension_score(0.5),
            truth_markers={"genuine_information": 0.5, "fabricated_content": 0.5, "fact_consistency": 0.5},
            manipulation_indicators={"persuasion_techniques": 0.5, "emotional_manipulation": 0.5, "cognitive_manipulation": 0.5},
            authenticity_assessment={"genuine_communication": 0.5, "calculated_responses": 0.5, "spontaneous_expression": 0.5},
            emotional_manipulation={"authentic_emotion": 0.5, "manufactured_emotion": 0.5, "emotional_authenticity": 0.5},
            information_integrity={"honest_disclosure": 0.5, "strategic_omission": 0.5, "information_accuracy": 0.5}
        )
    
    def _create_default_metacognitive(self) -> MetacognitiveAnalysis:
        """Create default metacognitive analysis"""
        return MetacognitiveAnalysis(
            overall_metacognitive=self._create_dimension_score(0.5),
            self_awareness={"genuine_introspection": 0.5, "simulated_self_reflection": 0.5, "self_knowledge_accuracy": 0.5},
            uncertainty_acknowledgment={"authentic_uncertainty": 0.5, "systematic_hedging": 0.5, "confidence_calibration": 0.5},
            bias_recognition={"bias_awareness": 0.5, "systematic_bias_patterns": 0.5, "cognitive_humility": 0.5},
            learning_indicators={"genuine_learning": 0.5, "programmed_adaptation": 0.5, "knowledge_growth": 0.5},
            cognitive_monitoring={"natural_monitoring": 0.5, "artificial_self_tracking": 0.5, "metacognitive_strategies": 0.5}
        )
    
    def _create_default_analysis(self) -> Dict[str, Any]:
        """Create default analysis for errors"""
        return {
            "overall_score": self._create_dimension_score(0.5),
            "error": "Analysis failed, using defaults"
        }
    
    def _create_ensemble_prediction(self, analyses: Dict[str, Any], text: str, 
                                  analysis_id: str, processing_time: float) -> ComprehensiveAnalysisResult:
        """Create final ensemble prediction from all analyses"""
        
        # Extract scores from all analyses
        scores = []
        valid_analyses = 0
        
        dimension_names = [
            "cognitive", "emotional", "creativity", "linguistic", "domain", 
            "personality", "temporal", "cultural", "deception", "metacognitive"
        ]
        
        weights = {
            "cognitive": 0.12,
            "emotional": 0.12, 
            "creativity": 0.10,
            "linguistic": 0.15,  # Higher weight for linguistic patterns
            "domain": 0.08,
            "personality": 0.10,
            "temporal": 0.08,
            "cultural": 0.08,
            "deception": 0.08,
            "metacognitive": 0.09
        }
        
        weighted_score = 0.0
        dimension_scores = {}
        all_indicators = []
        human_markers = []
        ai_markers = []
        
        for dim_name in dimension_names:
            if dim_name in analyses and hasattr(analyses[dim_name], f'overall_{dim_name}'):
                analysis = analyses[dim_name]
                overall_attr = getattr(analysis, f'overall_{dim_name}', None)
                
                if overall_attr:
                    score = overall_attr.score
                    weight = weights.get(dim_name, 0.1)
                    weighted_score += score * weight
                    dimension_scores[dim_name] = score
                    
                    # Collect indicators
                    all_indicators.extend(overall_attr.indicators)
                    
                    # Classify as human or AI markers
                    if score < 0.4:  # More human-like
                        human_markers.extend(overall_attr.indicators)
                    elif score > 0.6:  # More AI-like
                        ai_markers.extend(overall_attr.indicators)
                    
                    valid_analyses += 1
        
        # Final prediction
        prediction = "ai" if weighted_score > 0.5 else "human"
        confidence_value = abs(weighted_score - 0.5) * 2  # Convert to 0-1 scale
        
        # Determine confidence level
        if confidence_value > 0.8:
            certainty = "very_high"
        elif confidence_value > 0.6:
            certainty = "high"
        elif confidence_value > 0.4:
            certainty = "medium"
        elif confidence_value > 0.2:
            certainty = "low"
        else:
            certainty = "very_low"
        
        # Calculate ensemble metrics
        if len(dimension_scores) > 1:
            score_values = list(dimension_scores.values())
            ensemble_agreement = 1.0 - (np.std(score_values) / 0.5)  # Normalized std dev
            prediction_stability = 1.0 - abs(np.mean(score_values) - weighted_score)
        else:
            ensemble_agreement = 0.5
            prediction_stability = 0.5
        
        confidence_calibration = min(confidence_value + (valid_analyses / len(dimension_names)) * 0.2, 1.0)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(weighted_score, confidence_value, valid_analyses)
        
        return ComprehensiveAnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            text_length=len(text),
            model_used=self.model_name,
            processing_time=processing_time,
            
            ai_probability=weighted_score,
            prediction=prediction,
            overall_confidence=ConfidenceScore(
                value=confidence_value,
                certainty=certainty,
                reliability=min(valid_analyses / len(dimension_names), 1.0)
            ),
            
            cognitive_load=analyses.get("cognitive", self._create_default_cognitive_load()),
            emotional_intelligence=analyses.get("emotional", self._create_default_emotional_intelligence()),
            creativity=analyses.get("creativity", self._create_default_creativity()),
            linguistic=analyses.get("linguistic", self._create_default_linguistic()),
            domain_expertise=analyses.get("domain", self._create_default_domain_expertise()),
            personality=analyses.get("personality", self._create_default_personality()),
            temporal=analyses.get("temporal", self._create_default_temporal()),
            cultural=analyses.get("cultural", self._create_default_cultural()),
            deception=analyses.get("deception", self._create_default_deception()),
            metacognitive=analyses.get("metacognitive", self._create_default_metacognitive()),
            
            dimension_correlations=self._calculate_correlations(dimension_scores),
            consistency_scores=dimension_scores,
            contradiction_indicators=self._find_contradictions(dimension_scores),
            
            ensemble_agreement=ensemble_agreement,
            prediction_stability=prediction_stability,
            confidence_calibration=confidence_calibration,
            
            key_indicators=list(set(all_indicators[:10])),  # Top 10 unique indicators
            human_markers=list(set(human_markers[:10])),
            ai_markers=list(set(ai_markers[:10])),
            uncertainty_sources=self._identify_uncertainty_sources(dimension_scores),
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, score: float, confidence: float, valid_analyses: int) -> str:
        """Generate actionable recommendation"""
        
        if valid_analyses < 7:
            return f"Analysis incomplete ({valid_analyses}/10 dimensions). Recommend re-analysis with full system."
        
        if confidence > 0.8:
            pred = "AI-generated" if score > 0.5 else "human-written"
            return f"High confidence prediction: {pred}. Recommendation: Accept prediction."
        elif confidence > 0.5:
            pred = "AI-generated" if score > 0.5 else "human-written"
            return f"Medium confidence prediction: {pred}. Recommendation: Consider additional analysis."
        else:
            return "Low confidence prediction. Recommendation: Manual review required or collect additional context."
    
    def _calculate_correlations(self, dimension_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate correlations between dimensions"""
        if len(dimension_scores) < 2:
            return {}
        
        # Simple correlation calculation
        scores = list(dimension_scores.values())
        mean_score = np.mean(scores)
        
        correlations = {}
        for name, score in dimension_scores.items():
            correlations[f"{name}_vs_mean"] = 1.0 - abs(score - mean_score)
        
        return correlations
    
    def _find_contradictions(self, dimension_scores: Dict[str, float]) -> List[str]:
        """Find contradictory dimension scores"""
        contradictions = []
        
        if len(dimension_scores) < 2:
            return contradictions
        
        scores = list(dimension_scores.items())
        
        for i, (name1, score1) in enumerate(scores):
            for name2, score2 in scores[i+1:]:
                if abs(score1 - score2) > 0.5:  # Significant disagreement
                    if score1 > 0.6 and score2 < 0.4:
                        contradictions.append(f"{name1} indicates AI while {name2} indicates human")
                    elif score1 < 0.4 and score2 > 0.6:
                        contradictions.append(f"{name1} indicates human while {name2} indicates AI")
        
        return contradictions
    
    def _identify_uncertainty_sources(self, dimension_scores: Dict[str, float]) -> List[str]:
        """Identify sources of uncertainty in prediction"""
        uncertainty_sources = []
        
        for name, score in dimension_scores.items():
            if 0.4 < score < 0.6:  # Uncertain range
                uncertainty_sources.append(f"{name} analysis inconclusive (score: {score:.2f})")
        
        if len([s for s in dimension_scores.values() if 0.4 < s < 0.6]) > 5:
            uncertainty_sources.append("Multiple dimensions show inconclusive results")
        
        return uncertainty_sources
    
    def _create_error_result(self, error_msg: str, analysis_id: str, text: str, processing_time: float) -> ComprehensiveAnalysisResult:
        """Create error result for failed analysis"""
        
        return ComprehensiveAnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            text_length=len(text),
            model_used=self.model_name,
            processing_time=processing_time,
            
            ai_probability=0.5,
            prediction="inconclusive",
            overall_confidence=ConfidenceScore(value=0.0, certainty="very_low", reliability=0.0),
            
            cognitive_load=self._create_default_cognitive_load(),
            emotional_intelligence=self._create_default_emotional_intelligence(),
            creativity=self._create_default_creativity(),
            linguistic=self._create_default_linguistic(),
            domain_expertise=self._create_default_domain_expertise(),
            personality=self._create_default_personality(),
            temporal=self._create_default_temporal(),
            cultural=self._create_default_cultural(),
            deception=self._create_default_deception(),
            metacognitive=self._create_default_metacognitive(),
            
            dimension_correlations={},
            consistency_scores={},
            contradiction_indicators=[f"Analysis failed: {error_msg}"],
            
            ensemble_agreement=0.0,
            prediction_stability=0.0,
            confidence_calibration=0.0,
            
            key_indicators=[],
            human_markers=[],
            ai_markers=[],
            uncertainty_sources=[error_msg],
            recommendation=f"Analysis failed: {error_msg}. Recommend retry with different approach."
        )


# ============================================================================
# USAGE EXAMPLE AND TESTING
# ============================================================================

async def main():
    """Demo the Gemini structured analysis system"""
    
    import os
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize analyzer
    analyzer = GeminiStructuredAnalyzer(api_key)
    
    # Demo texts
    gpt4o_text = "While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity across various sectors. On the other hand, concerns about job displacement and ethical implications require careful consideration."
    
    human_text = "AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore. feels like we're heading straight for skynet territory lol 😅"
    
    print("🧠 Gemini Structured AI Detection Analysis")
    print("=" * 70)
    
    # Analyze GPT-4o sample
    print(f"\n📝 Analyzing GPT-4o sample...")
    result1 = await analyzer.comprehensive_analysis(gpt4o_text)
    
    # Convert to JSON and display
    result1_json = asdict(result1)
    
    print(f"🎯 Prediction: {result1.prediction} ({result1.ai_probability:.2%})")
    print(f"📊 Confidence: {result1.overall_confidence.certainty} ({result1.overall_confidence.value:.2%})")
    print(f"⏱️ Processing Time: {result1.processing_time:.2f}s")
    print(f"🔬 Analyses: {len([d for d in result1_json if 'overall' in str(d)])}")
    
    # Key insights
    print(f"\n🔍 Key AI Indicators:")
    for indicator in result1.ai_markers[:5]:
        print(f"  • {indicator}")
    
    print(f"\n💡 Recommendation: {result1.recommendation}")
    
    # Analyze human sample
    print(f"\n📝 Analyzing human sample...")
    result2 = await analyzer.comprehensive_analysis(human_text)
    
    print(f"🎯 Prediction: {result2.prediction} ({result2.ai_probability:.2%})")
    print(f"📊 Confidence: {result2.overall_confidence.certainty} ({result2.overall_confidence.value:.2%})")
    print(f"⏱️ Processing Time: {result2.processing_time:.2f}s")
    
    # Key insights
    print(f"\n👤 Key Human Indicators:")
    for indicator in result2.human_markers[:5]:
        print(f"  • {indicator}")
    
    print(f"\n💡 Recommendation: {result2.recommendation}")
    
    # Save results as JSON
    os.makedirs("../results", exist_ok=True)
    
    with open("../results/gpt4o_analysis.json", 'w') as f:
        json.dump(asdict(result1), f, indent=2)
    
    with open("../results/human_analysis.json", 'w') as f:
        json.dump(asdict(result2), f, indent=2)
    
    print(f"\n💾 Results saved to ../results/")
    print(f"\n✅ Structured Gemini analysis complete!")

if __name__ == "__main__":
    if GEMINI_AVAILABLE:
        asyncio.run(main())
    else:
        print("❌ Install google-generativeai: pip install google-generativeai")