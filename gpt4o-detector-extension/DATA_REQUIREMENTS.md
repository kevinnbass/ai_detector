# Data Requirements & Sample Size Guide

## 📊 Current Pipeline vs. Improved Pipeline

### Current Pipeline Issues
- ❌ **Hardcoded Patterns**: Based on assumptions, not real data
- ❌ **No Ground Truth**: No verified GPT-4o vs human examples
- ❌ **Tiny Sample Size**: Only ~5 examples of each type
- ❌ **No Validation**: Can't measure true accuracy

### Improved Pipeline (Now Implemented)
- ✅ **Data Collection Module**: `data_collector.py` for labeling
- ✅ **Training Pipeline**: `trainer.py` for ML model training  
- ✅ **Active Learning**: `active_learner.py` for smart sample selection
- ✅ **Validation System**: Cross-validation and proper test splits

## 🎯 Sample Size Requirements

### Minimum Viable Dataset
- **50+ samples total** (25 GPT-4o, 25 human) - Basic functionality
- **100+ samples total** (50 each) - Reasonable accuracy
- **200+ samples total** (100 each) - Good performance
- **500+ samples total** (250 each) - Production ready
- **1000+ samples total** (500 each) - High accuracy

### Sample Size Impact on Performance

| Total Samples | Expected Accuracy | Use Case |
|---------------|------------------|----------|
| 50 | ~65-70% | Proof of concept |
| 100 | ~70-75% | Development testing |
| 200 | ~75-80% | Beta release |
| 500 | ~80-85% | Production v1 |
| 1000+ | ~85-90% | Production v2+ |

## 📝 How to Collect High-Quality Training Data

### 1. Positive Examples (GPT-4o Generated)

#### Option A: Generate with OpenAI API
```python
# Use active_learner.py
from mining.active_learner import ActiveLearner

learner = ActiveLearner(openai_api_key="your-key-here")
learner.generate_gpt4o_samples(num_samples=100)
```

#### Option B: Manual Collection from Known Sources
- Find accounts/posts you KNOW are GPT-4o generated
- Look for telltale patterns you've observed
- Copy-paste into data collector

#### Option C: Create Your Own GPT-4o Content
- Use ChatGPT with GPT-4o model
- Ask it to write tweets on various topics
- Use prompts that encourage GPT-4o style

### 2. Negative Examples (Human Written)

#### Option A: Manual Collection
- Find authentic human tweets/posts
- Look for casual, emotional, or very short content
- Include typos, slang, abbreviated text

#### Option B: Your Own Content
- Write tweets in your natural style
- Include posts from friends/family
- Use informal, spontaneous language

#### Option C: Curated Human Datasets
- Twitter academic datasets
- Reddit comments (known human)
- News article comments

### 3. Balanced Collection Strategy

```python
# Example collection session
from mining.data_collector import DataCollector

collector = DataCollector()

# Add positive examples
collector.add_sample(
    text="While AI continues to evolve rapidly, it's important to note both advantages and disadvantages...", 
    label="gpt4o",
    confidence=0.9,
    source="chatgpt"
)

# Add negative examples  
collector.add_sample(
    text="AI is moving way too fast tbh can barely keep up anymore 😅",
    label="human", 
    confidence=1.0,
    source="twitter"
)

collector.save_dataset()
```

## 🎯 Quality over Quantity

### High-Quality Samples Include:

#### GPT-4o Characteristics to Capture:
- ✅ Balanced "pros and cons" language
- ✅ "It's important to note" type qualifiers
- ✅ "Not X, but Y" contrast structures
- ✅ Formal language in casual contexts
- ✅ Structured lists (firstly, secondly)
- ✅ Hedge words (perhaps, maybe, likely)

#### Human Characteristics to Capture:
- ✅ Informal/casual language
- ✅ Typos and abbreviations
- ✅ Emotional expressions
- ✅ Slang and colloquialisms  
- ✅ Very short posts
- ✅ Stream-of-consciousness style

### Sample Quality Checklist:
- [ ] Text is 30+ characters long
- [ ] Clear human vs GPT-4o distinction
- [ ] Representative of X/Twitter style
- [ ] No mixed authorship (human editing GPT output)
- [ ] High confidence in correct label

## 🔄 Active Learning Strategy

### Phase 1: Bootstrap (0-50 samples)
1. **Generate**: Use OpenAI API to create 25 GPT-4o samples
2. **Collect**: Manually add 25 human samples
3. **Train**: Initial model with basic accuracy

### Phase 2: Uncertainty Sampling (50-200 samples)  
1. **Suggest**: AI finds most uncertain samples
2. **Label**: Focus on borderline cases
3. **Retrain**: Improve model iteratively

### Phase 3: Diverse Sampling (200+ samples)
1. **Diversify**: Find samples with different characteristics
2. **Edge Cases**: Handle short texts, mixed styles
3. **Production**: Deploy with confidence

## 🛠 Using the Tools

### 1. Start Data Collection
```bash
cd mining
python data_collector.py
```

Interactive commands:
- `add gpt4o <text>` - Add GPT-4o sample
- `add human <text>` - Add human sample
- `stats` - Show dataset statistics
- `save` - Save current dataset

### 2. Active Learning Session
```bash
# Set API key (optional)
export OPENAI_API_KEY="your-key-here"

python active_learner.py
```

Commands:
- `generate 25` - Generate GPT-4o samples
- `suggest 5` - Get AI suggestions for labeling
- `stats` - Show progress

### 3. Train Models
```bash
python trainer.py
```

Automatically:
- Evaluates current rule-based detector
- Trains multiple ML models
- Compares performance
- Generates recommendations

## 📈 Expected Learning Curve

### With Active Learning:
- **Day 1**: Collect 50 samples → 70% accuracy
- **Day 2**: Collect 100 samples → 75% accuracy  
- **Week 1**: Collect 200 samples → 80% accuracy
- **Month 1**: Collect 500 samples → 85% accuracy

### Manual Collection Only:
- **Week 1**: Collect 100 samples → 70% accuracy
- **Month 1**: Collect 300 samples → 78% accuracy
- **Month 3**: Collect 500 samples → 82% accuracy

## 🎯 Recommended Collection Strategy

### Week 1: Quick Start (Target: 100 samples)
1. **Generate 50 GPT-4o samples** using API
2. **Collect 50 human samples** from Twitter/personal
3. **Train initial model**
4. **Deploy alpha version**

### Week 2-4: Refinement (Target: 300 samples)
1. **Use active learning** to find uncertain cases
2. **Focus on edge cases** (short text, mixed styles)
3. **Balance dataset** (maintain 50/50 ratio)
4. **Continuous retraining**

### Ongoing: Production (Target: 500+ samples)
1. **User feedback integration**
2. **Error analysis** from false positives/negatives
3. **Seasonal updates** (language patterns evolve)
4. **Cross-validation** with new data

## ⚖️ Class Balance

### Target Distribution:
- **50% GPT-4o samples** (positive class)
- **50% Human samples** (negative class)

### Monitoring Balance:
```python
stats = collector.get_statistics()
ratio = stats['balance_ratio']

if ratio < 0.8 or ratio > 1.25:
    print("⚠️ Dataset imbalanced - collect more of minority class")
```

## 🔍 Quality Assurance

### Regular Audits:
1. **Review random samples** for label accuracy
2. **Check for contamination** (GPT editing human text)
3. **Validate edge cases** work correctly
4. **Cross-annotate** difficult samples

### Performance Monitoring:
1. **Track accuracy** over time
2. **Monitor false positive/negative rates**
3. **A/B test** different model versions
4. **User feedback** integration

---

## 🚀 Quick Start Commands

```bash
# 1. Start collecting data
cd mining
python data_collector.py

# 2. Generate some GPT-4o samples (requires API key)
python active_learner.py
> generate 25

# 3. Add human samples manually
> label "your human text here"

# 4. Train models when you have 50+ samples
python trainer.py

# 5. Check performance
> stats
```

The key is starting small and iterating quickly. Even 50 well-chosen samples will give you a much better detector than the current hardcoded patterns!