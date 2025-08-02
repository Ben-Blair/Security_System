#!/usr/bin/env python3
"""
Setup script for Security Camera System
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check for camera access
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access available")
            cap.release()
        else:
            print("⚠ Camera access may be limited")
    except ImportError:
        print("⚠ OpenCV not available")
    
    # Check for dlib (required for face_recognition)
    try:
        import dlib
        print("✓ dlib available")
    except ImportError:
        print("⚠ dlib not available - face recognition may not work")
    
    # Check for face_recognition
    try:
        import face_recognition
        print("✓ face_recognition available")
    except ImportError:
        print("⚠ face_recognition not available")

def create_directories():
    """Create necessary directories"""
    directories = ['screenshots', 'recordings', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Security Camera System Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Check system requirements
    check_system_requirements()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    print("\nSetup completed successfully!")
    print("\nTo run the security camera system:")
    print("  python security_camera.py")
    print("\nFor help:")
    print("  python security_camera.py --help")

if __name__ == "__main__":
    main() 