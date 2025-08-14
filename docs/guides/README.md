# GPT-4o Text Detector Chrome Extension

A Chrome extension that detects GPT-4o generated posts on X (formerly Twitter) using advanced pattern analysis.

## Overview

This extension analyzes text patterns unique to GPT-4o and overlays visual warnings on detected posts. All detection happens locally in your browser - no data is sent to external servers.

## Features

- **Real-time Detection**: Automatically scans new posts as you browse X
- **Visual Indicators**: Clear overlays showing detection confidence and matched patterns
- **Privacy-First**: All analysis happens locally, no external API calls
- **Customizable**: Adjustable threshold, colors, and display options
- **Fast Performance**: Under 500ms per post analysis
- **Pattern Details**: Shows which specific patterns triggered detection

## Installation

### From Source

1. Clone or download this repository
2. Open Chrome and go to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the `extension` folder
5. The extension icon should appear in your toolbar

### From Chrome Web Store

*Coming soon - pending store review*

## How It Works

The extension detects GPT-4o text by analyzing:

### Language Patterns
- **Contrast Rhetoric**: "Not X, but Y" constructions
- **Hedging Language**: Excessive use of "perhaps", "maybe", "possibly"
- **Formal Transitions**: "Furthermore", "moreover" in casual contexts
- **Balanced Presentation**: Always presenting pros/cons equally
- **Qualifier Phrases**: "It's important to note", "Keep in mind"

### Statistical Features
- Sentence length consistency
- Lexical diversity patterns
- Punctuation usage
- Paragraph structure

### Detection Accuracy
- **Overall**: ~85% accuracy on test data
- **False Positive Rate**: ~12%
- **Processing Time**: <500ms per post

## Usage

1. **Install the extension** following the steps above
2. **Visit X.com** - the extension only works on Twitter/X
3. **Browse normally** - posts are analyzed automatically
4. **Look for indicators**:
   - 🤖 Red warning badge on detected posts
   - Click the badge to see detected patterns
   - Confidence score (if enabled in settings)

### Settings

Click the extension icon to access settings:

- **Enable/Disable**: Toggle detection on/off
- **Threshold**: Adjust detection sensitivity (50-90%)
- **Visual Options**: Show/hide overlays and confidence scores
- **Quick Mode**: Faster detection with slightly lower accuracy
- **Highlight Color**: Customize the warning color

## Privacy

- ✅ **Local Processing**: All analysis happens in your browser
- ✅ **No Data Collection**: No text is stored or transmitted
- ✅ **No External APIs**: No cloud services used for detection
- ✅ **Open Source**: Full code available for review

## Development

### Project Structure

```
gpt4o-detector-extension/
├── mining/                 # Pattern analysis tools
│   ├── gpt4o_miner.py     # Pattern mining module
│   └── detector.py        # Detection algorithms
├── extension/             # Chrome extension files
│   ├── manifest.json      # Extension manifest
│   ├── content/           # Content scripts
│   ├── background/        # Service worker
│   └── popup/             # Settings UI
├── tests/                 # Test suite
└── data/                  # Pattern data and samples
```

### Running Tests

```bash
# Install dependencies
pip install nltk scikit-learn pandas numpy spacy

# Run pattern mining
cd mining
python gpt4o_miner.py

# Run detector tests
cd ../tests
python test_detector.py
```

### Building

The extension is ready to load directly from the `extension` folder. No build process required.

## Pattern Details

The extension detects these specific GPT-4o patterns:

### 1. Excessive Hedging (Weight: 0.15)
**Pattern**: `(perhaps|maybe|possibly|might|could|seems|appears|likely|probably)`
**Example**: "Perhaps this might possibly be true"

### 2. Contrast Rhetoric (Weight: 0.25)
**Pattern**: `not X, but Y` constructions
**Example**: "Not simple solutions, but complex ones"

### 3. Formal in Casual (Weight: 0.20)
**Pattern**: `(furthermore|moreover|consequently|therefore)`
**Example**: "Furthermore, we must consider..."

### 4. Structured Lists (Weight: 0.15)
**Pattern**: `(firstly|secondly|thirdly|\d\.)`
**Example**: "Firstly, we analyze. Secondly, we plan."

### 5. Excessive Qualifiers (Weight: 0.20)
**Pattern**: "It's important to note", "Keep in mind"
**Example**: "It's important to note that..."

### 6. Balanced Presentation (Weight: 0.25)
**Pattern**: `(advantages.*disadvantages|pros.*cons)`
**Example**: "There are both advantages and disadvantages"

## Limitations

- **English Only**: Primarily trained on English text patterns
- **GPT-4o Specific**: May not detect other AI models accurately
- **Context Dependent**: Short posts (<30 characters) are not analyzed
- **Pattern Evolution**: AI writing patterns may change over time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Maintain privacy-first approach
- Optimize for performance (<500ms detection)

## Troubleshooting

### Extension Not Working
- Check that you're on x.com or twitter.com
- Ensure extension is enabled in chrome://extensions/
- Try refreshing the page

### No Detections Showing
- Check settings - ensure detection is enabled
- Try lowering the threshold in settings
- Verify posts have sufficient text (>30 characters)

### Performance Issues
- Enable "Quick Mode" in settings
- Clear cache using the settings button
- Check browser console for errors

## Roadmap

- [ ] Support for more languages
- [ ] Improved pattern detection
- [ ] API for external integration
- [ ] Firefox extension port
- [ ] Advanced statistics dashboard

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for educational and research purposes. Detection accuracy may vary. Always use critical thinking when evaluating content authenticity.

## Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Check the FAQ section
- **Feature Requests**: Submit via GitHub Issues with enhancement label

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**Compatibility**: Chrome 88+, Edge 88+