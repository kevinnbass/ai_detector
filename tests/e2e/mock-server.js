/**
 * Mock Server for E2E Testing
 * Serves mock Twitter pages for testing extension functionality
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8080;

// Serve static files
app.use(express.static(path.join(__dirname, 'mock-pages')));

// Mock Twitter timeline page
app.get('/mock-twitter.html', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Mock Twitter - AI Detector Test</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        .tweet { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .ai-indicator { background: #ff6b6b; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
        .ai-confidence { background: #4ecdc4; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-left: 5px; }
        .ai-details-popup { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border: 2px solid #333; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); z-index: 1000; }
        .detection-loading { opacity: 0.5; }
        [data-testid="offline-indicator"] { background: #ff4444; color: white; padding: 10px; text-align: center; }
        button { margin: 5px; padding: 5px 10px; }
      </style>
    </head>
    <body>
      <h1>Mock Twitter Timeline</h1>
      
      <div class="tweet" data-testid="tweet">
        <p data-testid="tweetText">It is important to note that this analysis requires careful consideration of multiple factors and stakeholders involved in the decision-making process.</p>
        <div class="tweet-actions">
          <button data-testid="mark-human">Mark as Human</button>
          <button data-testid="mark-ai">Mark as AI</button>
          <button data-testid="manual-detect-button">Detect</button>
        </div>
      </div>

      <div class="tweet" data-testid="tweet">
        <p data-testid="tweetText">omg just had the best pizza ever! 🍕 totally recommend this place to everyone, the crust was perfect and the toppings were amazing!</p>
        <div class="tweet-actions">
          <button data-testid="mark-human">Mark as Human</button>
          <button data-testid="mark-ai">Mark as AI</button>
          <button data-testid="manual-detect-button">Detect</button>
        </div>
      </div>

      <div class="tweet" data-testid="tweet">
        <p data-testid="tweetText">The methodology employed in this comprehensive study demonstrates rigorous academic standards and provides valuable insights into the subject matter.</p>
        <div class="tweet-actions">
          <button data-testid="mark-human">Mark as Human</button>
          <button data-testid="mark-ai">Mark as AI</button>
          <button data-testid="manual-detect-button">Detect</button>
        </div>
      </div>

      <div class="tweet" data-testid="tweet">
        <p data-testid="tweetText">lol that's so funny 😂 can't believe this actually happened, made my whole day better!</p>
        <div class="tweet-actions">
          <button data-testid="mark-human">Mark as Human</button>
          <button data-testid="mark-ai">Mark as AI</button>
          <button data-testid="manual-detect-button">Detect</button>
        </div>
      </div>

      <!-- Hidden elements for testing -->
      <div data-testid="sample-saved" style="display: none; background: green; color: white; padding: 10px;">Sample saved successfully!</div>
      <div data-testid="confirm-human" style="display: none;">Confirm Human</div>
      <div data-testid="confirm-ai" style="display: none;">Confirm AI</div>
      <div data-testid="detection-loading" style="display: none;">Analyzing...</div>
      <div data-testid="offline-indicator" style="display: none;">Offline - Using fallback detection</div>

      <script>
        // Simulate extension content script behavior
        setTimeout(() => {
          document.querySelectorAll('[data-testid="tweet"]').forEach((tweet, index) => {
            const text = tweet.querySelector('[data-testid="tweetText"]').textContent;
            
            // Simple pattern matching for demo
            const isAI = text.includes('important to note') || 
                        text.includes('methodology employed') || 
                        text.includes('careful consideration');
            
            const prediction = isAI ? 'ai' : 'human';
            const confidence = isAI ? Math.random() * 0.3 + 0.7 : Math.random() * 0.3 + 0.4;
            
            // Add prediction attributes
            tweet.setAttribute('data-ai-prediction', prediction);
            tweet.setAttribute('data-ai-confidence', confidence.toFixed(2));
            
            if (isAI) {
              const indicator = document.createElement('span');
              indicator.className = 'ai-indicator';
              indicator.textContent = 'AI';
              indicator.onclick = () => {
                const popup = document.createElement('div');
                popup.className = 'ai-details-popup';
                popup.innerHTML = \`
                  <h3>AI-generated content detected</h3>
                  <p>Confidence: \${(confidence * 100).toFixed(0)}%</p>
                  <p>Indicators: formal language, structured writing</p>
                  <button onclick="this.parentElement.remove()">Close</button>
                \`;
                document.body.appendChild(popup);
              };
              
              const confidenceSpan = document.createElement('span');
              confidenceSpan.className = 'ai-confidence';
              confidenceSpan.textContent = \`\${(confidence * 100).toFixed(0)}%\`;
              
              tweet.appendChild(indicator);
              tweet.appendChild(confidenceSpan);
            }
          });
        }, 1000);

        // Handle manual actions
        document.addEventListener('click', (e) => {
          if (e.target.matches('[data-testid="mark-human"]')) {
            const confirmBtn = document.querySelector('[data-testid="confirm-human"]');
            confirmBtn.style.display = 'block';
            confirmBtn.onclick = () => {
              const savedMsg = document.querySelector('[data-testid="sample-saved"]');
              savedMsg.style.display = 'block';
              setTimeout(() => savedMsg.style.display = 'none', 2000);
              confirmBtn.style.display = 'none';
            };
          }
          
          if (e.target.matches('[data-testid="mark-ai"]')) {
            const confirmBtn = document.querySelector('[data-testid="confirm-ai"]');
            confirmBtn.style.display = 'block';
            confirmBtn.onclick = () => {
              const savedMsg = document.querySelector('[data-testid="sample-saved"]');
              savedMsg.style.display = 'block';
              setTimeout(() => savedMsg.style.display = 'none', 2000);
              confirmBtn.style.display = 'none';
            };
          }
        });
      </script>
    </body>
    </html>
  `);
});

// Mock Twitter timeline with more tweets
app.get('/mock-twitter-timeline.html', (req, res) => {
  const tweets = [
    "It is important to note that this analysis requires careful consideration.",
    "lol this is hilarious 😂",
    "The methodology demonstrates academic rigor and scholarly approach.",
    "can't wait for the weekend! gonna be awesome 🎉",
    "This comprehensive study provides valuable insights into the subject matter.",
    "omg just saw the cutest dog ever! 🐕",
    "Furthermore, it should be mentioned that the implications are far-reaching.",
    "pizza night with friends was amazing! 🍕❤️"
  ];

  const tweetElements = tweets.map((text, index) => `
    <div class="tweet" data-testid="tweet">
      <p data-testid="tweetText">${text}</p>
    </div>
  `).join('');

  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Mock Twitter Timeline</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        .tweet { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 8px; }
      </style>
    </head>
    <body>
      <h1>Extended Timeline</h1>
      ${tweetElements}
      
      <script>
        setTimeout(() => {
          document.querySelectorAll('[data-testid="tweet"]').forEach((tweet) => {
            const text = tweet.querySelector('[data-testid="tweetText"]').textContent;
            const isAI = text.includes('important to note') || 
                        text.includes('methodology') || 
                        text.includes('comprehensive study') ||
                        text.includes('Furthermore');
            
            tweet.setAttribute('data-ai-prediction', isAI ? 'ai' : 'human');
          });
        }, 1500);
      </script>
    </body>
    </html>
  `);
});

// Mock infinite scroll page
app.get('/mock-twitter-infinite.html', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Mock Twitter Infinite Scroll</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        .tweet { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 8px; }
      </style>
    </head>
    <body>
      <h1>Infinite Scroll Timeline</h1>
      <div id="tweets-container">
        <!-- Initial tweets will be added by script -->
      </div>
      
      <script>
        const container = document.getElementById('tweets-container');
        let tweetCount = 0;
        
        const sampleTexts = [
          "It is important to note that this requires analysis.",
          "hey everyone! having a great day 😊",
          "The methodology employed demonstrates rigor.",
          "can't believe how good this movie was!",
          "Furthermore, the implications are significant.",
          "lunch was amazing today! 🍽️"
        ];
        
        function addTweets(count = 3) {
          for (let i = 0; i < count; i++) {
            const tweet = document.createElement('div');
            tweet.className = 'tweet';
            tweet.setAttribute('data-testid', 'tweet');
            
            const text = sampleTexts[tweetCount % sampleTexts.length];
            tweet.innerHTML = \`<p data-testid="tweetText">\${text}</p>\`;
            
            container.appendChild(tweet);
            tweetCount++;
            
            // Simulate detection
            setTimeout(() => {
              const isAI = text.includes('important to note') || 
                          text.includes('methodology') || 
                          text.includes('Furthermore');
              tweet.setAttribute('data-ai-prediction', isAI ? 'ai' : 'human');
            }, 500);
          }
        }
        
        // Add initial tweets
        addTweets(5);
        
        // Add more tweets on scroll
        window.addEventListener('scroll', () => {
          if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 1000) {
            addTweets(3);
          }
        });
      </script>
    </body>
    </html>
  `);
});

// Mock page with large text content
app.get('/mock-twitter-large-text.html', (req, res) => {
  const longText = "This is a comprehensive analysis that delves deep into the methodology and implications of artificial intelligence detection systems. ".repeat(20);
  
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Mock Twitter Large Text</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        .tweet { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 8px; }
      </style>
    </head>
    <body>
      <h1>Large Text Content</h1>
      
      <div class="tweet" data-testid="long-tweet">
        <p data-testid="tweetText">${longText}</p>
      </div>
      
      <div class="tweet" data-testid="long-tweet">
        <p data-testid="tweetText">This is another very long tweet that contains extensive text content to test the performance and handling of large text blocks in the AI detection system. ${"The analysis continues with detailed explanations and technical terminology. ".repeat(15)}</p>
      </div>
      
      <script>
        setTimeout(() => {
          document.querySelectorAll('[data-testid="long-tweet"]').forEach((tweet) => {
            tweet.setAttribute('data-ai-prediction', 'ai');
          });
        }, 2000);
      </script>
    </body>
    </html>
  `);
});

// Mock page with sensitive content
app.get('/mock-twitter-sensitive.html', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Mock Twitter Sensitive Content</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        .tweet { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 8px; }
      </style>
    </head>
    <body>
      <h1>Sensitive Content Test</h1>
      
      <div class="tweet" data-testid="tweet">
        <p data-testid="tweetText">Contact me at user@example.com for more information about this opportunity.</p>
      </div>
      
      <div class="tweet" data-testid="private-tweet">
        <p data-testid="tweetText">My password is secret123 and my username is @privateuser</p>
      </div>
      
      <script>
        // Simulate content filtering
        setTimeout(() => {
          document.querySelectorAll('[data-testid="tweet"]').forEach((tweet) => {
            const text = tweet.querySelector('[data-testid="tweetText"]').textContent;
            // Only process non-sensitive content
            if (!text.includes('@') && !text.includes('password')) {
              tweet.setAttribute('data-ai-prediction', 'human');
            }
          });
        }, 1000);
      </script>
    </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`Mock server listening on port ${PORT}`);
});