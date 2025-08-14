# 📊 Code Duplication Analysis

## Executive Summary
Current code duplication: ~40%
Target: <5%

## Duplication Areas Identified

### 1. Data Collectors (6 files → 1 unified) ✅
**Status**: RESOLVED
- `data_collector.py`
- `simple_collector.py`
- `tweet_data_collector.py`
- `test_collector.py`
- `demo_optimization.py`
- `run_optimization.py`
**Solution**: Created `unified_collector.py`

### 2. Trainer Modules (3 files → 1 unified) 🔄
**Files**:
- `trainer.py` - Basic training logic
- `enhanced_trainer.py` - Extended training with more features
- `active_learner.py` - Active learning implementation

**Duplication**:
- Model initialization code (90% duplicated)
- Training loop logic (85% duplicated)
- Validation methods (95% duplicated)
- Data preprocessing (100% duplicated)

### 3. Analyzer Modules (4 files → 1 unified) 🔄
**Files**:
- `llm_analyzer.py` - Basic LLM analysis
- `gemini_structured_analyzer.py` - Gemini-specific implementation
- `advanced_llm_system.py` - Advanced analysis features
- `demo_gemini.py` - Demo implementation

**Duplication**:
- API call logic (80% duplicated)
- Response parsing (75% duplicated)
- Error handling (100% duplicated)
- Prompt templates (60% duplicated)

### 4. Pattern Definitions 🔄
**Locations**:
- `detector.py` - Hardcoded patterns
- `detector-engine.js` - JavaScript patterns
- `detection-rules.json` - JSON patterns
- `gpt4o_miner.py` - Mining patterns

**Duplication**:
- Pattern regex definitions (100% duplicated)
- Weight values (90% duplicated)
- Threshold values (100% duplicated)

### 5. API Communication Code 🔄
**Locations**:
- Chrome extension background.js
- Chrome extension content.js
- Python API clients
- Integration modules

**Duplication**:
- Request/response handling (70% duplicated)
- Error handling (85% duplicated)
- Retry logic (100% duplicated)

### 6. Test Data 🔄
**Locations**:
- `test_samples.json`
- Hardcoded in test files
- Demo data in collectors
- Sample data in documentation

**Duplication**:
- Sample tweets (100% duplicated)
- Expected results (90% duplicated)

## Resolution Plan

### Priority 1: Trainers Consolidation
Create `unified_trainer.py` with:
- Strategy pattern for different training modes
- Configurable training pipeline
- Shared validation logic
- Common preprocessing

### Priority 2: Analyzers Consolidation
Create `unified_analyzer.py` with:
- Provider abstraction (Gemini, OpenAI, etc.)
- Shared prompt management
- Common response parsing
- Unified error handling

### Priority 3: Pattern Centralization
Create `pattern_registry.py` with:
- Single source of truth for patterns
- Export to different formats (Python, JS, JSON)
- Version management
- Pattern validation

### Priority 4: API Client Unification
Create `api_client.py` with:
- Unified HTTP client
- Retry with exponential backoff
- Circuit breaker pattern
- Request queuing

### Priority 5: Test Data Management
Create `test_data_manager.py` with:
- Centralized test data
- Data generation utilities
- Fixture management
- Mock data providers

## Metrics

### Before Consolidation:
- Total Lines of Code: ~8,000
- Duplicated Lines: ~3,200 (40%)
- Unique Functions: ~150
- Duplicated Functions: ~60

### After Consolidation (Expected):
- Total Lines of Code: ~5,500
- Duplicated Lines: ~275 (5%)
- Unique Functions: ~120
- Duplicated Functions: ~6

## Implementation Priority
1. ✅ Data Collectors (DONE)
2. 🔄 Trainers (IN PROGRESS)
3. ⏳ Analyzers
4. ⏳ Patterns
5. ⏳ API Clients
6. ⏳ Test Data

## Success Criteria
- [ ] Code duplication <5%
- [ ] All tests passing
- [ ] No functionality lost
- [ ] Improved maintainability
- [ ] Clear module boundaries