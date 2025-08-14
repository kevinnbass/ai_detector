# 📊 Structured JSON Output Examples

## 🎯 **Complete Analysis Structure**

The Gemini Structured Analyzer provides comprehensive JSON output with quantified scores across every dimension. Here are real examples:

---

## 🤖 **GPT-4o Generated Text Analysis**

### **Input Text:**
```
"While artificial intelligence continues to evolve rapidly, it's important to note that there are both advantages and disadvantages to consider. On one hand, AI can significantly boost productivity across various sectors. On the other hand, concerns about job displacement and ethical implications require careful consideration."
```

### **Structured JSON Output:**
```json
{
  "analysis_id": "analysis_1705123456789",
  "timestamp": "2024-01-13T15:30:45.123456",
  "text_length": 267,
  "model_used": "gemini-1.5-flash",
  "processing_time": 45.67,
  
  "ai_probability": 0.823,
  "prediction": "ai",
  "overall_confidence": {
    "value": 0.784,
    "certainty": "high",
    "reliability": 0.892
  },
  
  "cognitive_load": {
    "overall_load": {
      "score": 0.781,
      "confidence": {
        "value": 0.856,
        "certainty": "high",
        "reliability": 0.923
      },
      "indicators": [
        "systematic_processing",
        "consistent_complexity",
        "uniform_cognitive_load"
      ],
      "evidence": [
        "balanced sentence structure throughout",
        "consistent argumentation pattern",
        "uniform complexity distribution"
      ],
      "sub_scores": {
        "load_consistency": 0.834,
        "complexity_distribution": 0.776,
        "processing_efficiency": 0.889
      }
    },
    "complexity_distribution": {
      "evenness": 0.812,
      "variability": 0.234,
      "natural_peaks": 0.189
    },
    "processing_depth": {
      "surface_indicators": 0.723,
      "deep_thinking_markers": 0.234,
      "analytical_depth": 0.445
    },
    "effort_indicators": {
      "struggle_markers": 0.123,
      "effortless_generation": 0.834,
      "cognitive_strain": 0.156
    },
    "attention_patterns": {
      "focus_consistency": 0.867,
      "attention_wandering": 0.134,
      "selective_focus": 0.723
    },
    "mental_fatigue": {
      "fatigue_indicators": 0.089,
      "energy_consistency": 0.891,
      "endurance_markers": 0.867
    }
  },
  
  "emotional_intelligence": {
    "overall_eq": {
      "score": 0.692,
      "confidence": {
        "value": 0.734,
        "certainty": "medium",
        "reliability": 0.812
      },
      "indicators": [
        "systematic_empathy",
        "calculated_emotional_tone",
        "algorithmic_consideration"
      ],
      "evidence": [
        "balanced emotional appeals",
        "systematic consideration phrasing",
        "absence of personal emotional investment"
      ],
      "sub_scores": {
        "emotional_authenticity": 0.345,
        "empathy_indicators": 0.567,
        "emotional_complexity": 0.423
      }
    },
    "emotional_granularity": {
      "specificity": 0.456,
      "emotional_vocabulary": 0.589,
      "nuance_expression": 0.378
    },
    "empathy_authenticity": {
      "perspective_taking": 0.634,
      "compassion_markers": 0.445,
      "emotional_resonance": 0.367
    },
    "emotional_regulation": {
      "natural_regulation": 0.289,
      "systematic_control": 0.823,
      "emotional_balance": 0.756
    },
    "vulnerability_markers": {
      "authentic_openness": 0.178,
      "calculated_vulnerability": 0.712,
      "emotional_risk_taking": 0.234
    },
    "emotional_progression": {
      "natural_flow": 0.345,
      "artificial_transitions": 0.734,
      "emotional_coherence": 0.687
    }
  },
  
  "creativity": {
    "overall_creativity": {
      "score": 0.645,
      "confidence": {
        "value": 0.678,
        "certainty": "medium",
        "reliability": 0.745
      },
      "indicators": [
        "recombinatorial_patterns",
        "systematic_ideation",
        "safe_creative_choices"
      ],
      "evidence": [
        "standard pros/cons framework",
        "predictable argumentation structure",
        "conventional perspective presentation"
      ],
      "sub_scores": {
        "originality": 0.234,
        "innovation": 0.345,
        "creative_risk": 0.189
      }
    },
    "originality_score": {
      "novel_ideas": 0.189,
      "unique_perspectives": 0.234,
      "recombination_patterns": 0.823
    },
    "creative_risk_taking": {
      "bold_choices": 0.156,
      "safe_patterns": 0.834,
      "experimental_elements": 0.123
    },
    "metaphor_authenticity": {
      "fresh_metaphors": 0.089,
      "cliched_comparisons": 0.756,
      "metaphorical_consistency": 0.623
    },
    "perspective_uniqueness": {
      "individual_viewpoint": 0.178,
      "algorithmic_perspective": 0.812,
      "personal_voice": 0.234
    },
    "artistic_vision": {
      "aesthetic_sense": 0.456,
      "creative_coherence": 0.689,
      "artistic_authenticity": 0.234
    }
  },
  
  "linguistic": {
    "overall_linguistic": {
      "score": 0.856,
      "confidence": {
        "value": 0.923,
        "certainty": "very_high",
        "reliability": 0.945
      },
      "indicators": [
        "excessive_hedging",
        "systematic_contrast_rhetoric",
        "formal_register_inconsistency",
        "meta_commentary_patterns"
      ],
      "evidence": [
        "it's important to note",
        "on one hand... on the other hand",
        "require careful consideration",
        "advantages and disadvantages"
      ],
      "sub_scores": {
        "pattern_systematicity": 0.889,
        "linguistic_consistency": 0.834,
        "natural_variation": 0.167
      }
    },
    "hedging_frequency": {
      "hedge_words": 0.834,
      "uncertainty_markers": 0.756,
      "qualification_patterns": 0.891
    },
    "contrast_rhetoric": {
      "not_but_constructions": 0.689,
      "systematic_contrasts": 0.923,
      "balanced_presentations": 0.867
    },
    "formal_register": {
      "context_appropriateness": 0.445,
      "formal_in_casual": 0.756,
      "register_consistency": 0.623
    },
    "qualifier_usage": {
      "meta_commentary": 0.889,
      "importance_phrases": 0.923,
      "consideration_markers": 0.867
    },
    "structured_presentation": {
      "enumeration_patterns": 0.812,
      "logical_flow": 0.834,
      "systematic_organization": 0.889
    }
  },
  
  "domain_expertise": {
    "overall_expertise": {
      "score": 0.567,
      "confidence": {
        "value": 0.634,
        "certainty": "medium",
        "reliability": 0.723
      },
      "indicators": [
        "surface_level_knowledge",
        "generic_domain_language",
        "encyclopedic_patterns"
      ],
      "evidence": [
        "general AI terminology",
        "broad sector references",
        "non-specific technical details"
      ],
      "sub_scores": {
        "knowledge_authenticity": 0.445,
        "practical_understanding": 0.234,
        "expert_insight": 0.189
      }
    },
    "knowledge_depth": {
      "surface_knowledge": 0.756,
      "deep_understanding": 0.234,
      "conceptual_mastery": 0.345
    },
    "practical_experience": {
      "hands_on_indicators": 0.123,
      "theoretical_knowledge": 0.812,
      "real_world_context": 0.289
    },
    "domain_language": {
      "technical_accuracy": 0.634,
      "jargon_appropriateness": 0.567,
      "insider_perspective": 0.189
    },
    "edge_case_awareness": {
      "nuanced_understanding": 0.234,
      "common_misconceptions": 0.567,
      "expert_level_insights": 0.123
    },
    "contextual_wisdom": {
      "situational_awareness": 0.445,
      "practical_wisdom": 0.178,
      "experiential_knowledge": 0.156
    }
  },
  
  "personality": {
    "overall_personality": {
      "score": 0.734,
      "confidence": {
        "value": 0.689,
        "certainty": "medium",
        "reliability": 0.756
      },
      "indicators": [
        "systematic_personality",
        "optimized_communication_style",
        "algorithmic_value_presentation"
      ],
      "evidence": [
        "balanced perspective presentation",
        "systematic consideration of viewpoints",
        "optimized rhetorical structure"
      ],
      "sub_scores": {
        "personality_coherence": 0.823,
        "voice_authenticity": 0.234,
        "individual_character": 0.189
      }
    },
    "trait_consistency": {
      "stable_patterns": 0.867,
      "natural_variation": 0.156,
      "personality_coherence": 0.823
    },
    "voice_authenticity": {
      "genuine_voice": 0.234,
      "simulated_personality": 0.812,
      "personal_style": 0.189
    },
    "big_five_scores": {
      "openness": 0.634,
      "conscientiousness": 0.823,
      "extraversion": 0.456,
      "agreeableness": 0.756,
      "neuroticism": 0.178
    },
    "personality_quirks": {
      "individual_uniqueness": 0.156,
      "systematic_uniformity": 0.834,
      "personal_idiosyncrasies": 0.123
    },
    "value_coherence": {
      "consistent_values": 0.756,
      "algorithmic_optimization": 0.812,
      "personal_beliefs": 0.234
    }
  },
  
  "temporal": {
    "overall_temporal": {
      "score": 0.623,
      "confidence": {
        "value": 0.567,
        "certainty": "medium",
        "reliability": 0.634
      },
      "indicators": [
        "systematic_temporal_logic",
        "abstract_time_references",
        "logical_causation_patterns"
      ],
      "evidence": [
        "continues to evolve",
        "moving forward implications",
        "systematic temporal progression"
      ],
      "sub_scores": {
        "time_understanding": 0.689,
        "temporal_consistency": 0.756,
        "experiential_time": 0.234
      }
    },
    "time_perception": {
      "experiential_time": 0.189,
      "abstract_time": 0.812,
      "temporal_authenticity": 0.345
    },
    "sequence_reasoning": {
      "natural_ordering": 0.345,
      "algorithmic_sequence": 0.823,
      "causal_flow": 0.756
    },
    "memory_integration": {
      "episodic_markers": 0.178,
      "semantic_patterns": 0.834,
      "temporal_context": 0.567
    },
    "causal_understanding": {
      "experiential_causation": 0.234,
      "logical_causation": 0.812,
      "temporal_logic": 0.789
    },
    "temporal_consistency": {
      "natural_inconsistency": 0.167,
      "systematic_consistency": 0.856,
      "human_temporal_gaps": 0.123
    }
  },
  
  "cultural": {
    "overall_cultural": {
      "score": 0.578,
      "confidence": {
        "value": 0.523,
        "certainty": "medium",
        "reliability": 0.634
      },
      "indicators": [
        "generic_cultural_references",
        "algorithmic_social_awareness",
        "systematic_context_consideration"
      ],
      "evidence": [
        "broad social implications",
        "generic societal concerns",
        "systematic ethical framing"
      ],
      "sub_scores": {
        "cultural_embeddedness": 0.234,
        "social_authenticity": 0.445,
        "lived_experience": 0.189
      }
    },
    "cultural_fluency": {
      "authentic_knowledge": 0.234,
      "learned_patterns": 0.756,
      "cultural_intuition": 0.189
    },
    "social_embeddedness": {
      "community_markers": 0.156,
      "social_context": 0.634,
      "relational_understanding": 0.345
    },
    "generational_markers": {
      "age_appropriate": 0.567,
      "generational_context": 0.445,
      "era_authenticity": 0.378
    },
    "geographic_authenticity": {
      "location_knowledge": 0.445,
      "regional_markers": 0.234,
      "geographic_context": 0.356
    },
    "identity_coherence": {
      "identity_consistency": 0.689,
      "authentic_self": 0.178,
      "cultural_identity": 0.234
    }
  },
  
  "deception": {
    "overall_deception": {
      "score": 0.445,
      "confidence": {
        "value": 0.567,
        "certainty": "medium",
        "reliability": 0.634
      },
      "indicators": [
        "systematic_neutrality",
        "calculated_balance",
        "optimized_presentation"
      ],
      "evidence": [
        "balanced argument presentation",
        "systematic consideration markers",
        "optimized rhetorical structure"
      ],
      "sub_scores": {
        "truthfulness": 0.756,
        "manipulation_presence": 0.345,
        "authenticity_level": 0.445
      }
    },
    "truth_markers": {
      "genuine_information": 0.689,
      "fabricated_content": 0.234,
      "fact_consistency": 0.823
    },
    "manipulation_indicators": {
      "persuasion_techniques": 0.567,
      "emotional_manipulation": 0.345,
      "cognitive_manipulation": 0.445
    },
    "authenticity_assessment": {
      "genuine_communication": 0.345,
      "calculated_responses": 0.712,
      "spontaneous_expression": 0.178
    },
    "emotional_manipulation": {
      "authentic_emotion": 0.234,
      "manufactured_emotion": 0.634,
      "emotional_authenticity": 0.289
    },
    "information_integrity": {
      "honest_disclosure": 0.756,
      "strategic_omission": 0.234,
      "information_accuracy": 0.823
    }
  },
  
  "metacognitive": {
    "overall_metacognitive": {
      "score": 0.689,
      "confidence": {
        "value": 0.634,
        "certainty": "medium",
        "reliability": 0.712
      },
      "indicators": [
        "systematic_self_reflection",
        "algorithmic_uncertainty",
        "programmed_consideration"
      ],
      "evidence": [
        "systematic consideration phrases",
        "algorithmic uncertainty patterns",
        "programmed balance indicators"
      ],
      "sub_scores": {
        "self_awareness": 0.445,
        "cognitive_monitoring": 0.756,
        "meta_learning": 0.234
      }
    },
    "self_awareness": {
      "genuine_introspection": 0.189,
      "simulated_self_reflection": 0.812,
      "self_knowledge_accuracy": 0.445
    },
    "uncertainty_acknowledgment": {
      "authentic_uncertainty": 0.234,
      "systematic_hedging": 0.856,
      "confidence_calibration": 0.623
    },
    "bias_recognition": {
      "bias_awareness": 0.567,
      "systematic_bias_patterns": 0.712,
      "cognitive_humility": 0.445
    },
    "learning_indicators": {
      "genuine_learning": 0.178,
      "programmed_adaptation": 0.834,
      "knowledge_growth": 0.234
    },
    "cognitive_monitoring": {
      "natural_monitoring": 0.234,
      "artificial_self_tracking": 0.789,
      "metacognitive_strategies": 0.634
    }
  },
  
  "dimension_correlations": {
    "cognitive_vs_mean": 0.823,
    "emotional_vs_mean": 0.756,
    "creativity_vs_mean": 0.689,
    "linguistic_vs_mean": 0.934,
    "domain_vs_mean": 0.445,
    "personality_vs_mean": 0.623,
    "temporal_vs_mean": 0.567,
    "cultural_vs_mean": 0.489,
    "deception_vs_mean": 0.234,
    "metacognitive_vs_mean": 0.634
  },
  
  "consistency_scores": {
    "cognitive": 0.781,
    "emotional": 0.692,
    "creativity": 0.645,
    "linguistic": 0.856,
    "domain": 0.567,
    "personality": 0.734,
    "temporal": 0.623,
    "cultural": 0.578,
    "deception": 0.445,
    "metacognitive": 0.689
  },
  
  "contradiction_indicators": [
    "linguistic indicates AI while deception indicates mixed signals",
    "personality systematic patterns contradict creativity recombination"
  ],
  
  "ensemble_agreement": 0.756,
  "prediction_stability": 0.834,
  "confidence_calibration": 0.823,
  
  "key_indicators": [
    "systematic_processing",
    "excessive_hedging", 
    "balanced_presentation",
    "formal_register_inconsistency",
    "systematic_contrast_rhetoric",
    "algorithmic_perspective",
    "meta_commentary_patterns",
    "consistent_complexity",
    "optimized_presentation",
    "systematic_consideration"
  ],
  
  "human_markers": [
    "natural_variation",
    "authentic_uncertainty",
    "genuine_information"
  ],
  
  "ai_markers": [
    "systematic_processing",
    "excessive_hedging",
    "balanced_presentation", 
    "formal_register_inconsistency",
    "systematic_contrast_rhetoric",
    "algorithmic_perspective",
    "meta_commentary_patterns",
    "optimized_presentation",
    "systematic_consideration",
    "programmed_balance"
  ],
  
  "uncertainty_sources": [
    "domain analysis inconclusive (score: 0.57)",
    "cultural analysis inconclusive (score: 0.58)",
    "deception analysis inconclusive (score: 0.45)"
  ],
  
  "recommendation": "High confidence prediction: AI-generated. Recommendation: Accept prediction."
}
```

---

## 👤 **Human Generated Text Analysis**

### **Input Text:**
```
"AI is moving way too fast tbh. like every week there's something new and i can barely keep up anymore. feels like we're heading straight for skynet territory lol 😅"
```

### **Structured JSON Output:**
```json
{
  "analysis_id": "analysis_1705123456790",
  "timestamp": "2024-01-13T15:35:22.987654",
  "text_length": 143,
  "model_used": "gemini-1.5-flash",
  "processing_time": 42.34,
  
  "ai_probability": 0.187,
  "prediction": "human",
  "overall_confidence": {
    "value": 0.823,
    "certainty": "high",
    "reliability": 0.889
  },
  
  "cognitive_load": {
    "overall_load": {
      "score": 0.234,
      "confidence": {
        "value": 0.867,
        "certainty": "high", 
        "reliability": 0.923
      },
      "indicators": [
        "natural_cognitive_variability",
        "authentic_processing_patterns",
        "human_attention_wandering"
      ],
      "evidence": [
        "informal abbreviations (tbh)",
        "stream of consciousness flow",
        "natural cognitive shifts"
      ],
      "sub_scores": {
        "load_consistency": 0.189,
        "complexity_distribution": 0.823,
        "processing_efficiency": 0.345
      }
    },
    "complexity_distribution": {
      "evenness": 0.156,
      "variability": 0.834,
      "natural_peaks": 0.789
    },
    "processing_depth": {
      "surface_indicators": 0.723,
      "deep_thinking_markers": 0.445,
      "analytical_depth": 0.234
    },
    "effort_indicators": {
      "struggle_markers": 0.567,
      "effortless_generation": 0.234,
      "cognitive_strain": 0.445
    },
    "attention_patterns": {
      "focus_consistency": 0.234,
      "attention_wandering": 0.812,
      "selective_focus": 0.456
    },
    "mental_fatigue": {
      "fatigue_indicators": 0.634,
      "energy_consistency": 0.189,
      "endurance_markers": 0.234
    }
  },
  
  "emotional_intelligence": {
    "overall_eq": {
      "score": 0.189,
      "confidence": {
        "value": 0.812,
        "certainty": "high",
        "reliability": 0.856
      },
      "indicators": [
        "authentic_emotion",
        "genuine_concern",
        "natural_expression"
      ],
      "evidence": [
        "genuine anxiety about pace",
        "authentic emotional expression",
        "natural fear indicators (skynet)"
      ],
      "sub_scores": {
        "emotional_authenticity": 0.856,
        "empathy_indicators": 0.634,
        "emotional_complexity": 0.723
      }
    },
    "emotional_granularity": {
      "specificity": 0.789,
      "emotional_vocabulary": 0.456,
      "nuance_expression": 0.634
    },
    "empathy_authenticity": {
      "perspective_taking": 0.567,
      "compassion_markers": 0.445,
      "emotional_resonance": 0.712
    },
    "emotional_regulation": {
      "natural_regulation": 0.823,
      "systematic_control": 0.156,
      "emotional_balance": 0.234
    },
    "vulnerability_markers": {
      "authentic_openness": 0.834,
      "calculated_vulnerability": 0.123,
      "emotional_risk_taking": 0.756
    },
    "emotional_progression": {
      "natural_flow": 0.812,
      "artificial_transitions": 0.123,
      "emotional_coherence": 0.689
    }
  },
  
  "creativity": {
    "overall_creativity": {
      "score": 0.234,
      "confidence": {
        "value": 0.723,
        "certainty": "medium",
        "reliability": 0.789
      },
      "indicators": [
        "creative_metaphor",
        "cultural_reference",
        "humor_integration"
      ],
      "evidence": [
        "skynet reference",
        "creative comparison",
        "emoji usage for tone"
      ],
      "sub_scores": {
        "originality": 0.634,
        "innovation": 0.445,
        "creative_risk": 0.567
      }
    },
    "originality_score": {
      "novel_ideas": 0.445,
      "unique_perspectives": 0.634,
      "recombination_patterns": 0.234
    },
    "creative_risk_taking": {
      "bold_choices": 0.567,
      "safe_patterns": 0.234,
      "experimental_elements": 0.445
    },
    "metaphor_authenticity": {
      "fresh_metaphors": 0.634,
      "cliched_comparisons": 0.189,
      "metaphorical_consistency": 0.567
    },
    "perspective_uniqueness": {
      "individual_viewpoint": 0.789,
      "algorithmic_perspective": 0.123,
      "personal_voice": 0.823
    },
    "artistic_vision": {
      "aesthetic_sense": 0.456,
      "creative_coherence": 0.345,
      "artistic_authenticity": 0.634
    }
  },
  
  "linguistic": {
    "overall_linguistic": {
      "score": 0.123,
      "confidence": {
        "value": 0.912,
        "certainty": "very_high",
        "reliability": 0.945
      },
      "indicators": [
        "informal_language",
        "natural_errors",
        "authentic_voice",
        "casual_abbreviations"
      ],
      "evidence": [
        "tbh (text speak)",
        "lowercase formatting",
        "lol usage",
        "natural sentence fragments"
      ],
      "sub_scores": {
        "pattern_systematicity": 0.089,
        "linguistic_consistency": 0.156,
        "natural_variation": 0.912
      }
    },
    "hedging_frequency": {
      "hedge_words": 0.234,
      "uncertainty_markers": 0.345,
      "qualification_patterns": 0.189
    },
    "contrast_rhetoric": {
      "not_but_constructions": 0.089,
      "systematic_contrasts": 0.067,
      "balanced_presentations": 0.123
    },
    "formal_register": {
      "context_appropriateness": 0.834,
      "formal_in_casual": 0.089,
      "register_consistency": 0.789
    },
    "qualifier_usage": {
      "meta_commentary": 0.123,
      "importance_phrases": 0.089,
      "consideration_markers": 0.067
    },
    "structured_presentation": {
      "enumeration_patterns": 0.089,
      "logical_flow": 0.234,
      "systematic_organization": 0.123
    }
  },
  
  "personality": {
    "overall_personality": {
      "score": 0.178,
      "confidence": {
        "value": 0.834,
        "certainty": "high",
        "reliability": 0.867
      },
      "indicators": [
        "authentic_personality",
        "individual_voice",
        "personal_quirks"
      ],
      "evidence": [
        "personal communication style",
        "individual humor patterns",
        "authentic self-expression"
      ],
      "sub_scores": {
        "personality_coherence": 0.823,
        "voice_authenticity": 0.889,
        "individual_character": 0.834
      }
    },
    "trait_consistency": {
      "stable_patterns": 0.234,
      "natural_variation": 0.823,
      "personality_coherence": 0.756
    },
    "voice_authenticity": {
      "genuine_voice": 0.889,
      "simulated_personality": 0.089,
      "personal_style": 0.834
    },
    "big_five_scores": {
      "openness": 0.667,
      "conscientiousness": 0.234,
      "extraversion": 0.578,
      "agreeableness": 0.445,
      "neuroticism": 0.634
    },
    "personality_quirks": {
      "individual_uniqueness": 0.823,
      "systematic_uniformity": 0.089,
      "personal_idiosyncrasies": 0.789
    },
    "value_coherence": {
      "consistent_values": 0.567,
      "algorithmic_optimization": 0.123,
      "personal_beliefs": 0.789
    }
  }
}
```

---

## 🔍 **Key JSON Structure Elements**

### **Root Level Fields:**
- `analysis_id`: Unique identifier for each analysis
- `timestamp`: ISO format timestamp
- `ai_probability`: 0.0-1.0 overall AI likelihood  
- `prediction`: "human" or "ai"
- `processing_time`: Analysis duration in seconds

### **Confidence Structure:**
```json
"confidence": {
  "value": 0.0-1.0,        // Numerical confidence
  "certainty": "very_low/low/medium/high/very_high",
  "reliability": 0.0-1.0   // How reliable is this confidence
}
```

### **Dimension Score Structure:**
```json
"overall_[dimension]": {
  "score": 0.0-1.0,        // Main dimension score
  "confidence": {...},      // Confidence in this score
  "indicators": [...],      // Specific patterns found
  "evidence": [...],        // Textual evidence
  "sub_scores": {...}       // Component scores
}
```

### **Ensemble Metrics:**
- `ensemble_agreement`: How much dimensions agree (0.0-1.0)
- `prediction_stability`: Consistency of prediction (0.0-1.0)
- `confidence_calibration`: Calibrated confidence score

### **Interpretive Fields:**
- `key_indicators`: Most important patterns found
- `human_markers`: Patterns suggesting human authorship
- `ai_markers`: Patterns suggesting AI generation
- `contradiction_indicators`: Conflicting signals
- `uncertainty_sources`: What causes uncertainty
- `recommendation`: Actionable advice

---

## 🎯 **Score Interpretation Guide**

### **AI Probability Scale:**
- **0.0 - 0.2**: Very likely human
- **0.2 - 0.4**: Likely human  
- **0.4 - 0.6**: Uncertain/Mixed
- **0.6 - 0.8**: Likely AI
- **0.8 - 1.0**: Very likely AI

### **Confidence Levels:**
- **very_low** (0.0-0.2): Highly uncertain
- **low** (0.2-0.4): Low confidence  
- **medium** (0.4-0.7): Moderate confidence
- **high** (0.7-0.9): High confidence
- **very_high** (0.9-1.0): Very high confidence

### **Dimension Interpretation:**
Each dimension score represents how "AI-like" (1.0) vs "human-like" (0.0) that aspect appears:

- **Cognitive Load**: Human variation vs AI consistency
- **Emotional Intelligence**: Authentic emotion vs simulated
- **Creativity**: Original thinking vs recombination  
- **Linguistic**: Natural language vs systematic patterns
- **Domain Expertise**: Real experience vs encyclopedic knowledge
- **Personality**: Individual voice vs optimized communication
- **Temporal**: Experiential time vs logical progression
- **Cultural**: Lived experience vs learned patterns
- **Deception**: Authentic vs calculated communication
- **Metacognitive**: Natural self-awareness vs programmed reflection

---

## 💡 **Using the Structured Output**

### **Programmatic Access:**
```python
import json
from dataclasses import asdict

# Convert result to JSON
result_json = asdict(analysis_result)

# Access specific scores
ai_prob = result_json['ai_probability']
linguistic_score = result_json['linguistic']['overall_linguistic']['score']
confidence = result_json['overall_confidence']['certainty']

# Get top indicators
ai_markers = result_json['ai_markers']
human_markers = result_json['human_markers']

# Check for contradictions
contradictions = result_json['contradiction_indicators']
```

### **Decision Making:**
```python
def make_decision(result):
    prob = result['ai_probability']
    conf = result['overall_confidence']['value']
    
    if conf > 0.8:
        return f"High confidence: {result['prediction']}"
    elif conf > 0.5:
        return f"Medium confidence: {result['prediction']}"
    else:
        return "Manual review recommended"
```

This structured approach provides unprecedented insight into every aspect of AI detection with full quantification and interpretability! 📊✨