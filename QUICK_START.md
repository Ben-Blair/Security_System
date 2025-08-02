# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip3 install -r requirements_simple.txt
```

### 2. Test Your System
```bash
python3 test_system.py
```

### 3. Run the Security Camera
```bash
python3 security_camera_simple.py
```

## 🎯 Features Available

### ✅ Working Features
- **Live Video Feed**: Real-time camera feed
- **Person Detection**: Green borders around detected persons
- **Face Detection**: Blue borders around detected faces
- **Suspicious Activity**: Red borders for face hiding detection
- **Lingering Detection**: Yellow borders for persons staying too long
- **Screenshot Capture**: Press 's' to save screenshots

### ⚠️ Limited Features (Simple Mode)
- No face recognition (names not shown)
- Basic face detection only
- No face registration system

## 🎮 Controls

While the system is running:
- **'q'**: Quit the application
- **'s'**: Save a screenshot with timestamp

## 🔧 Customization

### Adjust Detection Sensitivity
```bash
# More sensitive (detects more people)
python3 security_camera_simple.py --confidence 0.3

# Less sensitive (fewer false positives)
python3 security_camera_simple.py --confidence 0.7
```

### Use Different Camera
```bash
# Use external camera
python3 security_camera_simple.py --camera 1

# Use IP camera
python3 security_camera_simple.py --camera "rtsp://192.168.1.100:554/stream"
```

## 📊 What You'll See

The system displays:
- **Green boxes**: Detected persons
- **Blue boxes**: Detected faces
- **Red boxes**: Suspicious activity (face hiding)
- **Yellow boxes**: Lingering persons
- **Status overlay**: Real-time detection counts

## 🆘 Troubleshooting

### Camera Not Working?
1. Check camera permissions in System Preferences
2. Try different camera index: `--camera 1`
3. Make sure no other app is using the camera

### Poor Detection?
1. Ensure good lighting
2. Adjust confidence threshold
3. Position camera for clear view

### High CPU Usage?
1. Reduce video resolution in config.py
2. Increase confidence threshold
3. Close other applications

## 🔄 Next Steps

For advanced features (face recognition, registration):
1. Install dlib: `brew install cmake boost boost-python3`
2. Install face_recognition: `pip3 install face_recognition`
3. Use the full version: `python3 security_camera.py`

## 📁 Files Created

- `screenshots/`: Saved screenshots
- `face_database.pkl`: Face recognition data (if using full version)
- `screenshot_YYYYMMDD_HHMMSS.jpg`: Individual screenshots 