import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse

class SimpleSecurityCamera:
    def __init__(self, camera_source=0, confidence_threshold=0.5):
        """
        Initialize the simplified security camera system
        
        Args:
            camera_source: Camera source (0 for webcam, or IP camera URL)
            confidence_threshold: Confidence threshold for detections
        """
        self.camera_source = camera_source
        self.confidence_threshold = confidence_threshold
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera source: {camera_source}")
        
        # Load pre-trained models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Person detection using HOG
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Tracking variables
        self.lingering_persons = {}
        self.suspicious_activities = []
        self.frame_count = 0
        
        # Detection thresholds
        self.lingering_threshold = 5.0  # seconds
        self.face_hiding_threshold = 3.0  # seconds without face detection
        
        # Colors for visualization
        self.colors = {
            'person': (0, 255, 0),      # Green
            'face': (255, 0, 0),        # Blue
            'suspicious': (0, 0, 255),  # Red
            'lingering': (0, 255, 255), # Yellow
            'unknown': (128, 128, 128)  # Gray
        }
        
        print("Simple Security Camera System Initialized")
        print(f"Camera Source: {camera_source}")
        print("Features: Person Detection, Face Detection, Lingering Detection, Suspicious Activity Detection")
    
    def detect_persons(self, frame):
        """Detect persons in the frame using HOG detector"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people using HOG
        boxes, weights = self.hog.detectMultiScale(
            gray, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        persons = []
        for (x, y, w, h), weight in zip(boxes, weights):
            if weight > self.confidence_threshold:
                persons.append({
                    'bbox': (x, y, w, h),
                    'confidence': weight,
                    'center': (x + w//2, y + h//2)
                })
        
        return persons
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2)
            })
        
        return face_data
    
    def detect_suspicious_activity(self, frame, persons, faces):
        """Detect suspicious activities like face hiding"""
        suspicious_activities = []
        
        # Check for persons without detected faces (potential face hiding)
        for person in persons:
            person_center = person['center']
            person_bbox = person['bbox']
            
            # Check if any face is within the person's bounding box
            face_in_person = False
            for face in faces:
                face_center = face['center']
                px, py, pw, ph = person_bbox
                
                if (px <= face_center[0] <= px + pw and 
                    py <= face_center[1] <= py + ph):
                    face_in_person = True
                    break
            
            if not face_in_person:
                # Potential face hiding
                suspicious_activities.append({
                    'type': 'face_hiding',
                    'person_bbox': person_bbox,
                    'confidence': person['confidence']
                })
        
        return suspicious_activities
    
    def detect_lingering(self, persons):
        """Detect persons who are lingering in the same area"""
        current_time = time.time()
        lingering_persons = []
        
        for person in persons:
            person_center = person['center']
            person_id = f"person_{person_center[0]}_{person_center[1]}"
            
            if person_id not in self.lingering_persons:
                self.lingering_persons[person_id] = {
                    'start_time': current_time,
                    'last_seen': current_time,
                    'center': person_center,
                    'bbox': person['bbox']
                }
            else:
                # Update last seen time
                self.lingering_persons[person_id]['last_seen'] = current_time
                self.lingering_persons[person_id]['bbox'] = person['bbox']
            
            # Check if person has been lingering
            duration = current_time - self.lingering_persons[person_id]['start_time']
            if duration > self.lingering_threshold:
                lingering_persons.append({
                    'person_id': person_id,
                    'duration': duration,
                    'bbox': person['bbox']
                })
        
        # Clean up old entries
        to_remove = []
        for person_id, data in self.lingering_persons.items():
            if current_time - data['last_seen'] > 10.0:  # Remove after 10 seconds of not seeing
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.lingering_persons[person_id]
        
        return lingering_persons
    
    def draw_detections(self, frame, persons, faces, suspicious_activities, lingering_persons):
        """Draw all detections on the frame"""
        # Draw persons with green borders
        for person in persons:
            x, y, w, h = person['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['person'], 2)
            cv2.putText(frame, f"Person ({person['confidence']:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['person'], 2)
        
        # Draw faces
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face'], 2)
            cv2.putText(frame, "Face", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['face'], 2)
        
        # Draw suspicious activities
        for activity in suspicious_activities:
            if activity['type'] == 'face_hiding':
                x, y, w, h = activity['person_bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['suspicious'], 3)
                cv2.putText(frame, "SUSPICIOUS: Face Hiding", (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['suspicious'], 2)
        
        # Draw lingering persons
        for lingering in lingering_persons:
            x, y, w, h = lingering['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['lingering'], 3)
            cv2.putText(frame, f"LINGERING: {lingering['duration']:.1f}s", (x, y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['lingering'], 2)
        
        # Draw status information
        self.draw_status_info(frame, persons, faces, suspicious_activities, lingering_persons)
    
    def draw_status_info(self, frame, persons, faces, suspicious_activities, lingering_persons):
        """Draw status information on the frame"""
        info_lines = [
            f"Persons Detected: {len(persons)}",
            f"Faces Detected: {len(faces)}",
            f"Suspicious Activities: {len(suspicious_activities)}",
            f"Lingering Persons: {len(lingering_persons)}",
            f"Simple Mode (No Face Recognition)"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def process_frame(self, frame):
        """Process a single frame and return detections"""
        # Detect persons
        persons = self.detect_persons(frame)
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Detect suspicious activities
        suspicious_activities = self.detect_suspicious_activity(frame, persons, faces)
        
        # Detect lingering persons
        lingering_persons = self.detect_lingering(persons)
        
        return persons, faces, suspicious_activities, lingering_persons
    
    def run(self):
        """Main loop for the security camera system"""
        print("Starting Simple Security Camera System...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            persons, faces, suspicious_activities, lingering_persons = self.process_frame(frame)
            
            # Draw detections
            self.draw_detections(frame, persons, faces, suspicious_activities, lingering_persons)
            
            # Display frame
            cv2.imshow('Simple Security Camera System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(frame)
        
        self.cleanup()
    
    def save_screenshot(self, frame):
        """Save a screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Simple Security Camera System stopped")

def main():
    parser = argparse.ArgumentParser(description='Simple Security Camera System')
    parser.add_argument('--camera', type=str, default='0', 
                       help='Camera source (0 for webcam, or IP camera URL)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Convert camera source to appropriate type
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)
    
    try:
        security_camera = SimpleSecurityCamera(camera_source, args.confidence)
        security_camera.run()
    except KeyboardInterrupt:
        print("\nSecurity Camera System stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 