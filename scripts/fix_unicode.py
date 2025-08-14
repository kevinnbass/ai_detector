#!/usr/bin/env python3
"""Fix Unicode characters in tweet_data_collector.py for Windows compatibility"""

import os

def fix_unicode_in_file(filename):
    """Replace emoji Unicode characters with text alternatives"""
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace emoji characters
    replacements = {
        '❌': 'ERROR:',
        '✅': 'SUCCESS:',
        '⚠️': 'WARNING:',
        '📝': 'INFO:',
        '💾': 'SAVED:',
        '👋': '',
        '📊': '',
        '🧠': 'ANALYZING:',
        '🧪': 'TESTING:',
        '🚀': '',
        '1️⃣': '1.',
        '2️⃣': '2.',
        '3️⃣': '3.',
        '4️⃣': '4.',
        '5️⃣': '5.',
        '🎯': '',
        '📈': '',
        '📤': 'EXPORTED:',
        '🔄': '',
        '🗑️': '',
        '🧠': '',
        '⚙️': '',
        '🔼': '',
        '🔧': '',
        '📋': '',
        '📊': '',
        '💰': '',
        '⚡': '',
        '🎨': '',
        '🎉': '',
        '✨': '',
        '🔥': '',
        '💡': '',
        '🛠️': '',
        '📖': '',
        '🎪': '',
        '🌟': '',
        '💎': '',
        '🚨': '',
        '🔍': '',
        '📱': '',
        '💻': '',
        '🖥️': '',
        '⌨️': '',
        '🖱️': '',
        '🖨️': '',
        '💽': '',
        '💿': '',
        '💾': 'SAVED:',
        '💻': '',
        '🖥️': '',
        '📱': '',
        '☎️': '',
        '📞': '',
        '📟': '',
        '📠': '',
        '📡': '',
        '📺': '',
        '📻': '',
        '🔊': '',
        '🔉': '',
        '🔈': '',
        '🔇': '',
        '🔔': '',
        '🔕': '',
        '📢': '',
        '📣': '',
        '📯': '',
        '🔔': '',
        '🎵': '',
        '🎶': '',
        '🎤': '',
        '🎧': '',
        '📻': '',
        '🎷': '',
        '🎸': '',
        '🎹': '',
        '🥁': '',
        '🎺': '',
    }
    
    # Apply replacements
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Write the file back
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode characters in {filename}")

if __name__ == "__main__":
    fix_unicode_in_file("tweet_data_collector.py")