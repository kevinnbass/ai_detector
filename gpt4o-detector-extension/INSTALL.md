# Installation Guide

## Quick Start

1. **Download/Clone** this repository to your local machine
2. **Open Chrome** and navigate to `chrome://extensions/`
3. **Enable Developer Mode** (toggle in top right corner)
4. **Click "Load unpacked"** and select the `extension` folder
5. **Visit X.com** and start browsing - the extension will automatically detect GPT-4o posts!

## Detailed Installation

### Prerequisites

- **Chrome Browser**: Version 88 or higher
- **Developer Mode**: Must be enabled for unpacked extensions

### Step-by-Step Installation

#### 1. Get the Extension Files
```bash
# Option A: Clone from GitHub
git clone https://github.com/yourusername/gpt4o-detector-extension.git
cd gpt4o-detector-extension

# Option B: Download and extract ZIP
# Download from GitHub releases page and extract
```

#### 2. Install in Chrome
1. Open Chrome browser
2. Navigate to `chrome://extensions/`
3. Toggle "Developer mode" ON (top right corner)
4. Click "Load unpacked" button
5. Navigate to and select the `extension` folder (not the root folder)
6. Click "Select Folder"

#### 3. Verify Installation
- Extension icon should appear in Chrome toolbar
- Extension should be listed in chrome://extensions/ with status "Enabled"
- Try visiting x.com - posts should start being analyzed automatically

### Alternative: Install as Packaged Extension

If you have a `.crx` file:
1. Download the `.crx` file
2. Drag and drop it onto chrome://extensions/
3. Click "Add extension" when prompted

## Configuration

### Initial Setup
1. **Click the extension icon** in the toolbar
2. **Review privacy settings** and accept terms
3. **Adjust detection threshold** if desired (default: 70%)
4. **Choose visual preferences** (colors, overlays, etc.)

### Permissions Explained
- **activeTab**: To analyze content on the current tab
- **storage**: To save your preferences
- **host permissions (x.com, twitter.com)**: To run on X/Twitter pages

## Development Setup

If you want to modify or contribute to the extension:

### Python Environment (for mining tools)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running Tests
```bash
# Test the detection algorithms
cd tests
python test_detector.py

# Mine new patterns (requires OpenAI API key)
cd mining
python gpt4o_miner.py
```

### Development Mode
```bash
# Install web-ext for better development experience
npm install -g web-ext

# Start development server (auto-reload)
web-ext run --source-dir extension/ --target chromium

# Lint extension code
web-ext lint --source-dir extension/
```

## Troubleshooting

### Common Issues

#### Extension Won't Load
- **Check folder structure**: Make sure you're selecting the `extension` folder, not the root
- **Enable Developer Mode**: Must be toggled ON in chrome://extensions/
- **Check permissions**: Ensure Chrome has permission to load local files

#### No Detection Happening  
- **Visit X.com**: Extension only works on twitter.com and x.com
- **Check settings**: Click extension icon and ensure "Enable Detection" is ON
- **Refresh page**: Sometimes requires a page reload after installation

#### Poor Detection Performance
- **Lower threshold**: Try reducing detection threshold in settings
- **Enable Quick Mode**: Faster but slightly less accurate detection
- **Check text length**: Posts must be >30 characters to analyze

#### Privacy Concerns
- **Local processing only**: All analysis happens in your browser
- **No data sent**: Extension doesn't communicate with external servers
- **View source**: All code is open source and auditable

### Getting Help

1. **Check README.md** for detailed usage instructions
2. **Review browser console** for error messages (F12 → Console)
3. **Report issues** on GitHub Issues page
4. **Join discussions** on GitHub Discussions

### Uninstallation

1. Go to `chrome://extensions/`
2. Find "GPT-4o Text Detector" 
3. Click "Remove"
4. Confirm removal

All stored settings will be cleared automatically.

## Advanced Configuration

### Custom Detection Rules
You can modify detection patterns by editing:
- `extension/content/detection-rules.json`
- Adjust weights, thresholds, and regex patterns
- Reload extension after changes

### Performance Tuning
- **Quick Mode**: Faster detection, ~90% accuracy of full mode
- **Threshold Adjustment**: Higher = fewer false positives, more missed detections
- **Cache Settings**: Extension caches results for better performance

### Privacy Settings
- **Data Collection**: Disabled by default, can't be enabled
- **Analytics**: No usage tracking or analytics
- **Storage**: Only saves user preferences locally

## Updates

### Automatic Updates (When published to Chrome Store)
- Extensions auto-update from Chrome Web Store
- No action required from users

### Manual Updates (Developer Mode)
1. Download updated extension files
2. Go to chrome://extensions/
3. Click "Reload" button on the extension
4. Or remove and re-add the extension

---

**Need Help?** Open an issue on GitHub or check the troubleshooting section above.