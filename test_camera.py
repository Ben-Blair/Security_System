import cv2
import time

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"✓ Camera {camera_index} opened successfully!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame captured! Shape: {frame.shape}")
                cap.release()
                return camera_index
            else:
                print(f"✗ Camera {camera_index} opened but couldn't read frame")
                cap.release()
        else:
            print(f"✗ Camera {camera_index} failed to open")
    
    print("No working camera found!")
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"\nUse camera index {working_camera} for the security camera system")
        print("Run: python3 security_camera.py --camera", working_camera)
    else:
        print("\nCamera access issues detected. Please check:")
        print("1. Camera permissions in System Preferences")
        print("2. No other app is using the camera")
        print("3. Camera is not disabled") 