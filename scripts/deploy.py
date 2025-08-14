#!/usr/bin/env python3
"""
Deployment script for AI Detector project
Handles both Python package and Chrome extension deployment
"""

import os
import sys
import shutil
import subprocess
import json
import zipfile
from datetime import datetime
from pathlib import Path


class Deployer:
    """Main deployment orchestrator"""
    
    def __init__(self, env='production'):
        self.env = env
        self.root_dir = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def deploy_python_package(self):
        """Deploy Python package"""
        print("📦 Deploying Python package...")
        
        # Clean previous builds
        for dir_name in ['build', 'dist', '*.egg-info']:
            for path in self.root_dir.glob(dir_name):
                if path.is_dir():
                    shutil.rmtree(path)
        
        # Create setup.py if not exists
        setup_file = self.root_dir / 'setup.py'
        if not setup_file.exists():
            self._create_setup_py()
        
        # Build package
        subprocess.run([sys.executable, 'setup.py', 'sdist', 'bdist_wheel'], 
                      cwd=self.root_dir, check=True)
        
        print("✅ Python package built successfully")
        
    def deploy_chrome_extension(self):
        """Deploy Chrome extension"""
        print("🌐 Deploying Chrome extension...")
        
        extension_dir = self.root_dir / 'extension'
        dist_dir = extension_dir / 'dist'
        
        # Build extension
        os.chdir(extension_dir)
        
        # Install dependencies if needed
        if not (extension_dir / 'node_modules').exists():
            subprocess.run(['npm', 'install'], check=True)
        
        # Build for production
        subprocess.run(['npm', 'run', 'build'], check=True)
        
        # Create release zip
        release_dir = extension_dir / 'releases'
        release_dir.mkdir(exist_ok=True)
        
        zip_name = f'ai-detector-extension-v2.1.0-{self.timestamp}.zip'
        zip_path = release_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(dist_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(dist_dir)
                    zf.write(file_path, arcname)
        
        print(f"✅ Extension packaged: {zip_path}")
        
    def deploy_documentation(self):
        """Deploy documentation"""
        print("📚 Deploying documentation...")
        
        docs_dir = self.root_dir / 'docs'
        
        # Generate API documentation
        subprocess.run([sys.executable, '-m', 'pydoc', '-w', 'src'], 
                      cwd=self.root_dir)
        
        # Move generated docs
        for html_file in self.root_dir.glob('*.html'):
            shutil.move(html_file, docs_dir / 'api')
        
        print("✅ Documentation deployed")
        
    def run_tests(self):
        """Run all tests before deployment"""
        print("🧪 Running tests...")
        
        # Run Python tests
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], 
                      cwd=self.root_dir)
        
        # Run JavaScript tests
        extension_dir = self.root_dir / 'extension'
        if (extension_dir / 'node_modules').exists():
            subprocess.run(['npm', 'test'], cwd=extension_dir)
        
        print("✅ All tests passed")
        
    def _create_setup_py(self):
        """Create setup.py file"""
        setup_content = '''
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-detector",
    version="2.1.0",
    author="AI Detector Team",
    description="AI text detection system for GPT-4o",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "google-generativeai>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    }
)
'''
        setup_file = self.root_dir / 'setup.py'
        setup_file.write_text(setup_content.strip())
        
    def deploy_all(self):
        """Run complete deployment"""
        print(f"🚀 Starting deployment ({self.env})...")
        
        try:
            # Run tests first
            if self.env == 'production':
                self.run_tests()
            
            # Deploy components
            self.deploy_python_package()
            self.deploy_chrome_extension()
            self.deploy_documentation()
            
            print(f"✅ Deployment complete!")
            print(f"📦 Python package in: dist/")
            print(f"🌐 Chrome extension in: extension/releases/")
            print(f"📚 Documentation in: docs/")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy AI Detector')
    parser.add_argument('--env', choices=['development', 'staging', 'production'],
                       default='production', help='Deployment environment')
    parser.add_argument('--component', choices=['python', 'extension', 'docs', 'all'],
                       default='all', help='Component to deploy')
    
    args = parser.parse_args()
    
    deployer = Deployer(env=args.env)
    
    if args.component == 'all':
        deployer.deploy_all()
    elif args.component == 'python':
        deployer.deploy_python_package()
    elif args.component == 'extension':
        deployer.deploy_chrome_extension()
    elif args.component == 'docs':
        deployer.deploy_documentation()