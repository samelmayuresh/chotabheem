#!/usr/bin/env python3
"""
Enhanced setup script for Emotion AI application
Handles environment setup, dependency installation, and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔄 Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
        return False
    
    print("✅ Virtual environment created successfully")
    return True

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    pip_command = "venv\\Scripts\\pip" if platform.system() == "Windows" else "venv/bin/pip"
    
    print("🔄 Upgrading pip...")
    if not run_command(f"{pip_command} install --upgrade pip"):
        return False
    
    print("🔄 Installing dependencies...")
    if not run_command(f"{pip_command} install -r requirements.txt"):
        return False
    
    print("✅ Dependencies installed successfully")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ["utils", "data", "logs", "exports"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' ready")

def create_env_file():
    """Create .env file template"""
    env_content = """# Emotion AI Configuration
# Copy this file to .env and fill in your API keys

# OpenRouter API Key (for AI therapy features)
OPENROUTER_KEY=your_openrouter_key_here

# Tenor API Key (for GIF features)
TENOR_API_KEY=your_tenor_key_here

# Weather API Key (for weather-based mood suggestions)
WEATHER_API_KEY=your_weather_key_here

# Supabase Configuration (for data storage)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env.template")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env.template file")
        print("📝 Please copy .env.template to .env and add your API keys")
    else:
        print("✅ .env.template already exists")

def check_system_dependencies():
    """Check for system-level dependencies"""
    print("🔄 Checking system dependencies...")
    
    # Check for ffmpeg (required for audio processing)
    if not run_command("ffmpeg -version", "Checking FFmpeg"):
        print("⚠️  FFmpeg not found. Audio processing may not work properly.")
        print("   Install FFmpeg from: https://ffmpeg.org/download.html")
    
    # Check for git (optional but recommended)
    if run_command("git --version", "Checking Git"):
        print("✅ Git is available")
    else:
        print("⚠️  Git not found (optional)")

def create_run_scripts():
    """Create convenient run scripts"""
    
    # Windows batch script
    windows_script = """@echo off
echo Starting Emotion AI Application...
call venv\\Scripts\\activate
streamlit run app_enhanced.py --server.port=8501 --server.address=0.0.0.0
pause
"""
    
    with open("run_app.bat", "w") as f:
        f.write(windows_script)
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting Emotion AI Application..."
source venv/bin/activate
streamlit run app_enhanced.py --server.port=8501 --server.address=0.0.0.0
"""
    
    with open("run_app.sh", "w") as f:
        f.write(unix_script)
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("run_app.sh", 0o755)
    
    print("✅ Created run scripts (run_app.bat / run_app.sh)")

def main():
    """Main setup function"""
    print("🚀 Emotion AI Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Create environment file template
    create_env_file()
    
    # Check system dependencies
    check_system_dependencies()
    
    # Create run scripts
    create_run_scripts()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Run the application:")
    
    if platform.system() == "Windows":
        print("   - Double-click run_app.bat")
        print("   - Or run: .\\run_app.bat")
    else:
        print("   - Run: ./run_app.sh")
        print("   - Or run: bash run_app.sh")
    
    print("\n3. Open your browser to: http://localhost:8501")
    print("\n📚 For more information, check the README.md file")

if __name__ == "__main__":
    main()