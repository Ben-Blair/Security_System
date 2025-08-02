#!/usr/bin/env python3
"""
Test script for Security Camera System
"""

import cv2
import numpy as np
import sys
import os

def test_opencv():
    """Test OpenCV installation"""
    print("Testing OpenCV...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        return True
    except Exception as e:
        print(f"OpenCV test failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("Testing camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera test successful - Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("Camera test failed - Could not read frame")
                cap.release()
                return False
        else:
            print("Camera test failed - Could not open camera")
            return False
    except Exception as e:
        print(f"Camera test failed: {e}")
        return False

def test_face_recognition():
    """Test face recognition library"""
    print("Testing face recognition...")
    try:
        import face_recognition
        print("✓ face_recognition library available")
        return True
    except ImportError:
        print("✗ face_recognition library not available")
        return False

def test_dlib():
    """Test dlib installation"""
    print("Testing dlib...")
    try:
        import dlib
        print(f"✓ dlib version: {dlib.__version__}")
        return True
    except ImportError:
        print("✗ dlib not available")
        return False

def test_numpy():
    """Test numpy installation"""
    print("Testing numpy...")
    try:
        print(f"✓ numpy version: {np.__version__}")
        return True
    except Exception as e:
        print(f"✗ numpy test failed: {e}")
        return False

def test_models():
    """Test loading of pre-trained models"""
    print("Testing model loading...")
    try:
        # Test Haar cascade loading
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("✗ Could not load face cascade")
            return False
        
        # Test HOG detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        print("✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_detection():
    """Test basic detection functionality"""
    print("Testing detection functionality...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open camera for detection test")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Could not read frame for detection test")
            return False
        
        # Test face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Test person detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        print(f"✓ Detection test successful - Found {len(faces)} faces, {len(boxes)} persons")
        return True
        
    except Exception as e:
        print(f"✗ Detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Security Camera System - System Test")
    print("=" * 40)
    
    tests = [
        ("OpenCV", test_opencv),
        ("Camera", test_camera),
        ("NumPy", test_numpy),
        ("dlib", test_dlib),
        ("Face Recognition", test_face_recognition),
        ("Models", test_models),
        ("Detection", test_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nTo start the security camera system:")
        print("  python security_camera.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the installation.")
        print("\nCommon solutions:")
        print("1. Run: python setup.py")
        print("2. Install missing dependencies manually")
        print("3. Check camera permissions")

if __name__ == "__main__":
    main() 