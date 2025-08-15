# Developer Setup Guide

Complete guide for setting up the AI Detector development environment, including all dependencies, configuration, and development tools.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Development Tools](#development-tools)
- [Running the System](#running-the-system)
- [Testing Setup](#testing-setup)
- [Chrome Extension Development](#chrome-extension-development)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   # Check Python version
   python --version
   # Should output: Python 3.8.x or higher
   ```

2. **Node.js 16+**
   ```bash
   # Check Node.js version
   node --version
   # Should output: v16.x.x or higher
   
   npm --version
   # Should output: 8.x.x or higher
   ```

3. **Git**
   ```bash
   # Check Git version
   git --version
   # Should output: git version 2.x.x or higher
   ```

4. **Docker (Optional but Recommended)**
   ```bash
   # Check Docker version
   docker --version
   # Should output: Docker version 20.x.x or higher
   
   docker-compose --version
   # Should output: docker-compose version 1.29.x or higher
   ```

### Optional Dependencies

1. **CUDA Toolkit (for GPU acceleration)**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Version 11.0 or higher recommended

2. **Redis (for caching)**
   - Can be installed locally or run via Docker
   - Version 6.0 or higher recommended

3. **PostgreSQL (for production database)**
   - Can be installed locally or run via Docker
   - Version 12.0 or higher recommended

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 10 GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Recommended Requirements
- **CPU**: 4+ cores, 3.0 GHz
- **RAM**: 8+ GB
- **Storage**: 20+ GB free space (SSD recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional)

### Development Environment
- **IDE**: VS Code, PyCharm, or similar
- **Browser**: Chrome/Edge for extension development
- **Terminal**: PowerShell (Windows), Terminal (macOS), bash/zsh (Linux)

## Installation

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/kevinnbass/ai_detector.git
cd ai_detector

# Verify the structure
ls -la
# Should show: src/, tests/, docs/, etc.
```

### 2. Python Environment Setup

#### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv ai_detector_env

# Activate virtual environment
# On Windows (PowerShell)
ai_detector_env\Scripts\Activate.ps1

# On Windows (Command Prompt)
ai_detector_env\Scripts\activate.bat

# On macOS/Linux
source ai_detector_env/bin/activate

# Verify activation
which python
# Should point to your virtual environment
```

#### Using Conda (Alternative)

```bash
# Create conda environment
conda create -n ai_detector python=3.9
conda activate ai_detector

# Verify activation
conda info --envs
# Should show ai_detector as active
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install optional GPU dependencies (if CUDA available)
pip install -r requirements-gpu.txt

# Verify installation
pip list | grep -E "(fastapi|numpy|torch|transformers)"
```

### 4. Install Node.js Dependencies

```bash
# Install Node.js dependencies for Chrome extension
cd src/extension
npm install

# Verify installation
npm list --depth=0

# Return to project root
cd ../..
```

### 5. Environment Variables Setup

```bash
# Copy environment template
cp .env.example .env

# Edit environment file
# On Windows
notepad .env

# On macOS
open -e .env

# On Linux
nano .env
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root with the following configuration:

```bash
# API Configuration
API_HOST=localhost
API_PORT=8000
API_DEBUG=true
API_RELOAD=true

# Database Configuration
DATABASE_URL=sqlite:///./ai_detector.db
# For PostgreSQL (production):
# DATABASE_URL=postgresql://user:password@localhost:5432/ai_detector

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
# For local development without Redis:
# REDIS_URL=memory://

# External API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Security
SECRET_KEY=your_super_secret_key_here_change_in_production
API_KEY_SALT=your_api_key_salt_here

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed

# Performance
MAX_WORKERS=4
CACHE_TTL=3600
BATCH_SIZE=100

# Features
ENABLE_GPU=false
ENABLE_MONITORING=true
ENABLE_RATE_LIMITING=false
```

### API Keys Setup

#### OpenRouter API Key
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Navigate to API Keys section
4. Create a new API key
5. Add to `.env` file: `OPENROUTER_API_KEY=your_key_here`

#### Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` file: `GEMINI_API_KEY=your_key_here`

### Optional Configuration Files

#### `config/development.yaml`
```yaml
detection:
  default_method: "ensemble"
  confidence_threshold: 0.7
  timeout_seconds: 30

processing:
  batch_size: 100
  max_workers: 4
  enable_gpu: false
  
cache:
  default_ttl: 3600
  max_size: 1000
  
logging:
  level: "DEBUG"
  format: "detailed"
  file: "logs/ai_detector.log"
```

## Development Tools

### 1. IDE Configuration

#### VS Code Setup
Install recommended extensions:

```bash
# Install VS Code extensions via command line
code --install-extension ms-python.python
code --install-extension ms-python.flake8
code --install-extension ms-python.black-formatter
code --install-extension ms-vscode.vscode-typescript-next
code --install-extension bradlc.vscode-tailwindcss
code --install-extension ms-vscode.vscode-json
```

VS Code settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./ai_detector_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Setup
1. Open project in PyCharm
2. Configure Python interpreter to use virtual environment
3. Enable code style: Black formatter
4. Configure run configurations for FastAPI server

### 2. Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually (optional)
pre-commit run --all-files
```

Pre-commit configuration (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v2.6.2
    hooks:
      - id: prettier
        files: '\.(js|ts|json|css|md)$'
```

### 3. Code Quality Tools

```bash
# Install additional development tools
pip install black flake8 isort mypy pytest-cov

# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Run all quality checks
make lint  # If Makefile is configured
```

## Running the System

### 1. Start the API Server

#### Development Mode
```bash
# Activate virtual environment
source ai_detector_env/bin/activate  # or activate.bat on Windows

# Start development server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Alternative: using the run script
python run_server.py

# Server should start at: http://localhost:8000
```

#### Production Mode
```bash
# Start production server
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or using Docker
docker-compose up --build
```

### 2. Verify API Server

```bash
# Check health endpoint
curl http://localhost:8000/api/health

# Test detection endpoint
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message for AI detection."}'

# Open API documentation
open http://localhost:8000/docs  # Swagger UI
open http://localhost:8000/redoc  # ReDoc
```

### 3. Start Supporting Services

#### Redis (for caching)
```bash
# Using Docker
docker run -d -p 6379:6379 --name ai-detector-redis redis:7-alpine

# Or install locally and start
redis-server

# Verify connection
redis-cli ping
# Should respond: PONG
```

#### PostgreSQL (for production)
```bash
# Using Docker
docker run -d -p 5432:5432 --name ai-detector-postgres \
  -e POSTGRES_DB=ai_detector \
  -e POSTGRES_USER=ai_detector \
  -e POSTGRES_PASSWORD=your_password \
  postgres:14-alpine

# Verify connection
psql -h localhost -U ai_detector -d ai_detector
```

### 4. Database Setup

```bash
# Create database tables
python -m alembic upgrade head

# Or using the setup script
python scripts/setup_database.py

# Verify tables created
python -c "from src.database import engine; print(engine.table_names())"
```

## Testing Setup

### 1. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detection.py

# Run specific test function
pytest tests/test_detection.py::test_pattern_detection

# Run tests with output
pytest -v -s

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### 2. Test Configuration

Create `pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### 3. Test Environment

```bash
# Copy test environment
cp .env.example .env.test

# Edit test-specific settings
# Set TEST_DATABASE_URL, disable external APIs, etc.
```

## Chrome Extension Development

### 1. Extension Setup

```bash
# Navigate to extension directory
cd src/extension

# Install dependencies
npm install

# Build extension for development
npm run build:dev

# Build for production
npm run build:prod

# Watch for changes during development
npm run watch
```

### 2. Load Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `src/extension/dist` directory
5. Extension should appear in your browser

### 3. Extension Development Workflow

```bash
# Start development server (if needed)
python -m uvicorn src.api.main:app --reload --port 8000

# In another terminal, watch extension files
cd src/extension
npm run watch

# Make changes to extension files
# Chrome will reload automatically for manifest changes
# For content script changes, refresh the page
```

### 4. Extension Testing

```bash
# Run extension tests
cd src/extension
npm test

# Run specific test
npm test -- --testNamePattern="detection"

# Test with coverage
npm run test:coverage
```

## Docker Development Environment

### 1. Using Docker Compose

```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose build api
docker-compose up api
```

### 2. Docker Compose Configuration

Create `docker-compose.dev.yml`:
```yaml
version: '3.8'

services:
  api:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./logs:/app/logs
    environment:
      - DEBUG=true
      - RELOAD=true
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: ai_detector
      POSTGRES_USER: ai_detector
      POSTGRES_PASSWORD: development
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## Development Scripts

### 1. Common Development Tasks

Create `scripts/dev.py`:
```python
#!/usr/bin/env python3
"""Development utility scripts."""

import os
import subprocess
import sys
from pathlib import Path

def start_server():
    """Start the development server."""
    os.system("uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")

def run_tests():
    """Run the test suite."""
    os.system("pytest -v")

def lint_code():
    """Run code linting and formatting."""
    os.system("black src/ tests/")
    os.system("isort src/ tests/")
    os.system("flake8 src/ tests/")

def setup_db():
    """Set up the database."""
    os.system("alembic upgrade head")

def build_extension():
    """Build the Chrome extension."""
    os.chdir("src/extension")
    os.system("npm run build:dev")
    os.chdir("../..")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/dev.py [command]")
        print("Commands: server, test, lint, db, extension")
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "server":
        start_server()
    elif command == "test":
        run_tests()
    elif command == "lint":
        lint_code()
    elif command == "db":
        setup_db()
    elif command == "extension":
        build_extension()
    else:
        print(f"Unknown command: {command}")
```

### 2. Makefile for Common Tasks

Create `Makefile`:
```makefile
.PHONY: install dev test lint clean build docker

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	cd src/extension && npm install

# Start development server
dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest -v --cov=src

# Lint and format code
lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ tests/
	mypy src/

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist/
	rm -rf build/

# Build extension
build-extension:
	cd src/extension && npm run build:prod

# Docker development
docker-dev:
	docker-compose -f docker-compose.dev.yml up --build

# Setup database
setup-db:
	alembic upgrade head
```

## Troubleshooting

### Common Issues

#### 1. Python Import Errors
```bash
# Issue: ModuleNotFoundError
# Solution: Ensure virtual environment is activated and PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to .env file
echo "PYTHONPATH=$(pwd)" >> .env
```

#### 2. API Server Won't Start
```bash
# Check if port is in use
netstat -tulpn | grep :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill process using port
kill -9 $(lsof -ti:8000)  # Linux/macOS
taskkill /PID <PID> /F     # Windows
```

#### 3. Extension Not Loading
1. Check Chrome extensions page for errors
2. Verify manifest.json syntax
3. Reload extension after code changes
4. Check browser console for errors

#### 4. Database Connection Issues
```bash
# Check database service status
docker ps  # If using Docker
pg_isready -h localhost -p 5432  # For PostgreSQL

# Reset database (development only)
rm ai_detector.db  # For SQLite
dropdb ai_detector && createdb ai_detector  # For PostgreSQL
```

#### 5. Redis Connection Issues
```bash
# Check Redis service
redis-cli ping

# Start Redis if not running
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### Performance Issues

#### 1. Slow API Responses
- Enable caching: Set `REDIS_URL` in environment
- Optimize database queries: Check logs for slow queries
- Use GPU acceleration: Set `ENABLE_GPU=true` if available

#### 2. High Memory Usage
- Reduce batch size: Lower `BATCH_SIZE` in configuration
- Enable memory optimization: Set appropriate limits
- Monitor with: `htop` or `Activity Monitor`

### Development Tips

1. **Use VS Code Tasks**: Configure tasks for common operations
2. **Enable Hot Reload**: API server automatically reloads on changes
3. **Use Debug Mode**: Set `DEBUG=true` for detailed error messages
4. **Check Logs**: Monitor `logs/ai_detector.log` for issues
5. **Use Chrome DevTools**: Debug extension with F12 in browser

### Getting Help

1. **Check Issues**: [GitHub Issues](https://github.com/kevinnbass/ai_detector/issues)
2. **Read Documentation**: Full documentation in `docs/` directory
3. **Join Discord**: Community support available
4. **Create Issue**: Report bugs or request features

---

## Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install Node.js 16+
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install Python dependencies
- [ ] Install Node.js dependencies
- [ ] Configure environment variables
- [ ] Start API server
- [ ] Load Chrome extension
- [ ] Run tests
- [ ] Verify everything works

You're now ready to develop with the AI Detector system! 🚀

---

*Last updated: January 15, 2025*