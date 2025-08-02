import cv2
import numpy as np
import face_recognition
import os
import pickle
import time
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import argparse

class SecurityCamera:
    def __init__(self, camera_source=0, confidence_threshold=0.5):
        """
        Initialize the security camera system
        
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
        
        # Give camera time to initialize
        time.sleep(1)
        
        # Load pre-trained models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Person detection using HOG
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Face recognition storage
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = []  # Store actual face images
        self.face_database_file = "face_database.pkl"
        self.load_face_database()
        
        # Tracking variables
        self.person_trackers = {}
        self.face_trackers = {}
        self.lingering_persons = {}
        self.suspicious_activities = []
        self.frame_count = 0
        
        # Detection thresholds - increased tolerance for less sensitivity
        self.lingering_threshold = 8.0  # seconds (increased from 5.0)
        self.face_hiding_threshold = 5.0  # seconds (increased from 3.0)
        self.suspicious_activity_tolerance = 0.8  # increased tolerance for face hiding detection
        
        # Auto-registration settings
        self.auto_register_faces = True
        self.auto_register_confidence = 0.7  # minimum confidence for auto-registration
        self.auto_register_cooldown = 3.0  # seconds between auto-registrations
        self.last_auto_register_time = 0
        
        # Debug settings
        self.debug_mode = False
        
        # Colors for visualization - Ring-style theme
        self.colors = {
            'person': (0, 150, 255),    # Ring blue
            'face': (255, 255, 255),    # White
            'suspicious': (255, 69, 0), # Orange-red for alerts
            'lingering': (255, 215, 0), # Gold for warnings
            'unknown': (128, 128, 128), # Gray
            'title': (255, 255, 255),   # White
            'background': (15, 15, 15), # Very dark gray (Ring dark)
            'accent': (0, 150, 255),    # Ring blue
            'success': (0, 255, 0),     # Green
            'warning': (255, 215, 0),   # Gold
            'error': (255, 69, 0),      # Orange-red
            'ring_dark': (25, 25, 25),  # Ring dark background
            'ring_blue': (0, 150, 255), # Ring blue
            'ring_white': (255, 255, 255), # Ring white
            'ring_gray': (100, 100, 100)   # Ring gray
        }
        
        print("Security Camera System Initialized")
        print(f"Camera Source: {camera_source}")
        print(f"Face Database: {len(self.known_face_encodings)} known faces")
        print(f"Auto-registration: {'ENABLED' if self.auto_register_faces else 'DISABLED'}")
    
    def load_face_database(self):
        """Load known faces from database"""
        if os.path.exists(self.face_database_file):
            try:
                with open(self.face_database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    # Handle old databases that might not have images
                    if 'images' in data:
                        self.known_face_images = data['images']
                    else:
                        # Create placeholder images for existing faces
                        self.known_face_images = []
                        for _ in range(len(self.known_face_encodings)):
                            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                            placeholder[:] = (100, 100, 100)
                            self.known_face_images.append(placeholder)
                        # Save updated database with images
                        self.save_face_database()
                print(f"Loaded {len(self.known_face_encodings)} known faces")
            except Exception as e:
                print(f"Error loading face database: {e}")
    
    def save_face_database(self):
        """Save known faces to database"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'images': self.known_face_images # Save images
            }
            with open(self.face_database_file, 'wb') as f:
                pickle.dump(data, f)
            print("Face database saved")
        except Exception as e:
            print(f"Error saving face database: {e}")
    
    def register_face(self, name, face_encoding, face_image=None):
        """Register a new face in the database with clean image"""
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        if face_image is not None:
            # Save clean face image without any detection boxes
            self.known_face_images.append(face_image)
        else:
            # Create a placeholder image if none provided
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder[:] = (100, 100, 100)
            self.known_face_images.append(placeholder)
        self.save_face_database()
        print(f"Registered face for: {name}")
    
    def auto_register_new_faces(self, frame, faces):
        """Automatically register new faces that haven't been seen before with better validation"""
        if not self.auto_register_faces:
            return
        
        current_time = time.time()
        if current_time - self.last_auto_register_time < self.auto_register_cooldown:
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Validation checks to prevent false positives
            # 1. Minimum face size (prevent tiny detections)
            if w < 50 or h < 50:
                continue
            
            # 2. Maximum face size (prevent oversized detections)
            if w > 300 or h > 300:
                continue
            
            # 3. Face aspect ratio should be reasonable (not too wide or tall)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # 4. Face should be in reasonable position (not at edges)
            height, width = frame.shape[:2]
            if x < 10 or y < 10 or x + w > width - 10 or y + h > height - 10:
                continue
            
            # Extract face encoding
            face_location = (y, x + w, y + h, x)
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            
            if not face_encodings:
                continue
            
            new_face_encoding = face_encodings[0]
            
            # 5. Check if this face is already registered
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    new_face_encoding,
                    tolerance=0.6
                )
                
                if True in matches:
                    continue  # Face already registered
            
            # 6. Additional validation: face should be stable for a moment
            face_id = f"face_{x}_{y}_{w}_{h}"
            if face_id not in self.face_trackers:
                self.face_trackers[face_id] = {'first_seen': current_time, 'count': 1}
            else:
                self.face_trackers[face_id]['count'] += 1
                # Only register if face has been stable for at least 1 second
                if current_time - self.face_trackers[face_id]['first_seen'] < 1.0:
                    continue
                if self.face_trackers[face_id]['count'] < 3:  # Must be detected multiple times
                    continue
            
            # Auto-register new face
            face_image = frame[y:y+h, x:x+w].copy()  # Clean image without boxes
            name = f"Person_{len(self.known_face_names) + 1}"
            
            print(f"Auto-registering new face: {name} (Size: {w}x{h}, Position: {x},{y})")
            self.register_face(name, new_face_encoding, face_image)
            self.last_auto_register_time = current_time
            
            # Clean up old face trackers
            old_faces = []
            for fid, data in self.face_trackers.items():
                if current_time - data['first_seen'] > 10.0:  # Remove after 10 seconds
                    old_faces.append(fid)
            
            for fid in old_faces:
                del self.face_trackers[fid]
    
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
    
    def recognize_faces(self, frame, face_locations):
        """Recognize faces using face_recognition library"""
        if not self.known_face_encodings:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.6
            )
            
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            recognized_faces.append({
                'location': face_location,
                'name': name,
                'encoding': face_encoding
            })
        
        return recognized_faces
    
    def detect_suspicious_activity(self, frame, persons, faces):
        """Detect suspicious activities like face hiding with reduced sensitivity"""
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
            
            # Only flag as suspicious if no face detected AND confidence is high
            if not face_in_person and person['confidence'] > self.suspicious_activity_tolerance:
                # Additional check: only flag if person has been visible for a while
                person_id = f"person_{person_center[0]}_{person_center[1]}"
                current_time = time.time()
                
                if person_id not in self.person_trackers:
                    self.person_trackers[person_id] = {'first_seen': current_time}
                
                # Only flag as suspicious if person has been visible for more than threshold
                if current_time - self.person_trackers[person_id]['first_seen'] > self.face_hiding_threshold:
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
    
    def draw_detections(self, frame, persons, faces, recognized_faces, suspicious_activities, lingering_persons):
        """Draw all detections on the frame with Ring-style professional styling"""
        # Draw persons with Ring blue borders
        for person in persons:
            x, y, w, h = person['bbox']
            # Draw Ring-style border for person
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['ring_blue'], 2)
            # Add confidence text with Ring-style background
            text = f"Person ({person['confidence']:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 10, y), self.colors['ring_blue'], -1)
            cv2.putText(frame, text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw faces with Ring white borders
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['ring_white'], 2)
            # Add face label with Ring-style background
            text = "Face"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 10, y), self.colors['ring_white'], -1)
            cv2.putText(frame, text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw recognized faces with Ring blue borders and names
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            cv2.rectangle(frame, (left, top), (right, bottom), self.colors['ring_blue'], 3)
            # Add name with Ring-style styling
            name = face['name']
            (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width + 10, top), self.colors['ring_blue'], -1)
            cv2.putText(frame, name, (left + 5, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw suspicious activities with Ring orange-red styling
        for activity in suspicious_activities:
            if activity['type'] == 'face_hiding':
                x, y, w, h = activity['person_bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['error'], 3)
                # Add warning text with Ring styling
                text = "SUSPICIOUS: Face Hiding"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - text_height - 30), (x + text_width + 10, y - 10), self.colors['error'], -1)
                cv2.putText(frame, text, (x + 5, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw lingering persons with Ring gold styling
        for lingering in lingering_persons:
            x, y, w, h = lingering['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['warning'], 3)
            # Add duration text with Ring styling
            text = f"LINGERING: {lingering['duration']:.1f}s"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - text_height - 50), (x + text_width + 10, y - 30), self.colors['warning'], -1)
            cv2.putText(frame, text, (x + 5, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw status information with Ring-style styling
        self.draw_status_info(frame, persons, faces, suspicious_activities, lingering_persons)
    
    def draw_status_info(self, frame, persons, faces, suspicious_activities, lingering_persons):
        """Draw Ring-style status information overlay"""
        height, width = frame.shape[:2]
        
        # Create Ring-style status bar background
        status_bar_height = 100
        status_bar = np.zeros((status_bar_height, width, 3), dtype=np.uint8)
        status_bar[:] = self.colors['ring_dark']
        
        # Add Ring branding
        title = "RING SECURITY"
        subtitle = "Live Monitoring"
        cv2.putText(status_bar, title, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['ring_white'], 2)
        cv2.putText(status_bar, subtitle, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ring_gray'], 1)
        
        # Add auto-registration status with Ring styling
        auto_status = "AUTO-REG: ON" if self.auto_register_faces else "AUTO-REG: OFF"
        cv2.putText(status_bar, auto_status, (width - 180, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ring_blue'], 2)
        
        # Add debug mode indicator
        if self.debug_mode:
            debug_text = "DEBUG: ON"
            cv2.putText(status_bar, debug_text, (width - 180, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['warning'], 2)
        
        # Add status indicators with Ring styling
        y_offset = 75
        indicators = [
            (f"Persons: {len(persons)}", self.colors['ring_blue']),
            (f"Faces: {len(faces)}", self.colors['ring_white']),
            (f"Known: {len(self.known_face_encodings)}", self.colors['ring_gray']),
            (f"Alerts: {len(suspicious_activities) + len(lingering_persons)}", self.colors['error'] if (len(suspicious_activities) + len(lingering_persons)) > 0 else self.colors['ring_gray'])
        ]
        
        x_spacing = width // len(indicators)
        for i, (text, color) in enumerate(indicators):
            x_pos = 20 + i * x_spacing
            cv2.putText(status_bar, text, (x_pos, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add status bar to frame
        frame[0:status_bar_height, 0:width] = status_bar
        
        # Add debug information if debug mode is on
        if self.debug_mode:
            debug_y = status_bar_height + 20
            debug_info = [
                f"Frame: {self.frame_count}",
                f"Face Trackers: {len(self.face_trackers)}",
                f"Person Trackers: {len(self.person_trackers)}",
                f"Last Auto-reg: {time.time() - self.last_auto_register_time:.1f}s ago"
            ]
            
            for i, info in enumerate(debug_info):
                cv2.putText(frame, info, (20, debug_y + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['ring_gray'], 1)
        
        # Add Ring-style control instructions
        instructions = "Controls: Q=Quit | R=Register | S=Screenshot | V=View Faces | D=Debug"
        cv2.putText(frame, instructions, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['ring_gray'], 1)
    
    def process_frame(self, frame):
        """Process a single frame and return detections"""
        # Detect persons
        persons = self.detect_persons(frame)
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Auto-register new faces
        self.auto_register_new_faces(frame, faces)
        
        # Recognize faces - convert face bbox to face_recognition format
        face_locations = []
        for face in faces:
            x, y, w, h = face['bbox']
            face_locations.append((y, x + w, y + h, x))
        
        recognized_faces = self.recognize_faces(frame, face_locations)
        
        # Detect suspicious activities
        suspicious_activities = self.detect_suspicious_activity(frame, persons, faces)
        
        # Detect lingering persons
        lingering_persons = self.detect_lingering(persons)
        
        return persons, faces, recognized_faces, suspicious_activities, lingering_persons
    
    def run(self):
        """Main loop for the security camera system"""
        print("Starting Security Camera System...")
        print("Press 'q' to quit, 'r' to register a face, 's' to save screenshot, 'v' to view registered faces, 'd' to toggle debug mode")
        if self.auto_register_faces:
            print("Auto-registration: ENABLED - New faces will be registered automatically")
        
        # Try to read a few frames to ensure camera is working
        for i in range(5):
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to grab frame {i+1}/5, retrying...")
                time.sleep(0.5)
            else:
                break
        else:
            print("Camera failed to initialize properly. Please check camera permissions.")
            self.cleanup()
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            persons, faces, recognized_faces, suspicious_activities, lingering_persons = self.process_frame(frame)
            
            # Draw detections
            self.draw_detections(frame, persons, faces, recognized_faces, suspicious_activities, lingering_persons)
            
            # Display frame
            cv2.imshow('Security Camera System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.register_face_interactive(frame)
            elif key == ord('s'):
                self.save_screenshot(frame)
            elif key == ord('v'):
                self.view_registered_faces()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        
        self.cleanup()
    
    def register_face_interactive(self, frame):
        """Interactive face registration: register the largest detected face in the current frame."""
        print("Face Registration Mode")
        faces = self.detect_faces(frame)
        if not faces:
            print("No face detected. Please make sure your face is visible to the camera.")
            return
        
        # Apply the same validation as auto-registration
        valid_faces = []
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Validation checks
            if w < 50 or h < 50:
                continue
            if w > 300 or h > 300:
                continue
            
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            height, width = frame.shape[:2]
            if x < 10 or y < 10 or x + w > width - 10 or y + h > height - 10:
                continue
            
            valid_faces.append(face)
        
        if not valid_faces:
            print("No valid faces detected. Please ensure your face is clearly visible and not too close/far from camera.")
            return
        
        # Select the largest valid face
        largest_face = max(valid_faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
        x, y, w, h = largest_face['bbox']
        
        # Extract face encoding using face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # face_recognition expects (top, right, bottom, left)
        face_location = (y, x + w, y + h, x)
        face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        
        if not face_encodings:
            print("Could not extract face encoding. Try again.")
            return
        
        # Check if this face is already registered
        new_face_encoding = face_encodings[0]
        if self.known_face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                new_face_encoding,
                tolerance=0.6
            )
            
            if True in matches:
                first_match_index = matches.index(True)
                existing_name = self.known_face_names[first_match_index]
                print(f"Face already registered as: {existing_name}")
                print("This person is already in the database.")
                return
        
        # Capture the actual face image
        face_image = frame[y:y+h, x:x+w].copy()
        
        # Use a default name instead of blocking input
        name = f"Person_{len(self.known_face_names) + 1}"
        print(f"New face detected! Auto-registering as: {name}")
        print(f"Face details: Size {w}x{h}, Position ({x},{y})")
        self.register_face(name, new_face_encoding, face_image)
        print(f"Face registered for: {name}")
        print("You can rename faces later by editing the database.")
    
    def view_registered_faces(self):
        """Display all registered faces in a Ring-style interface with hover interactions"""
        if not self.known_face_names:
            print("No registered faces found!")
            return
        
        print(f"\n{'='*60}")
        print("           REGISTERED FACES DATABASE")
        print(f"{'='*60}")
        
        for i, (name, encoding) in enumerate(zip(self.known_face_names, self.known_face_encodings), 1):
            print(f"{i:2d}. {name}")
            print(f"     └─ Encoding: {encoding.shape} ({len(encoding)} features)")
            print(f"     └─ Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 Total registered faces: {len(self.known_face_names)}")
        print(f"💾 Database file: {self.face_database_file}")
        
        # Create visual display
        print(f"\n🖼️  Displaying {len(self.known_face_names)} registered faces...")
        print("💡 Hover over faces and click DELETE button to remove")
        
        # Create a window to display faces
        face_window_name = "Ring Security - Face Database"
        cv2.namedWindow(face_window_name, cv2.WINDOW_NORMAL)
        
        # Calculate window size based on number of faces
        num_faces = len(self.known_face_names)
        cols = min(3, num_faces)  # Max 3 columns
        rows = (num_faces + cols - 1) // cols  # Calculate rows needed
        
        # Create a large canvas to display all faces
        face_size = 250
        title_height = 80
        canvas_width = cols * face_size
        canvas_height = rows * face_size + title_height + 100  # Extra space for names and buttons
        
        # Create blank canvas with Ring dark theme
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = self.colors['ring_dark']
        
        # Add Ring-style title bar
        title = "FACE DATABASE"
        subtitle = "Ring Security System"
        cv2.putText(canvas, title, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['ring_white'], 2)
        cv2.putText(canvas, subtitle, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ring_gray'], 1)
        
        # Store face positions for mouse interaction
        self.face_positions = []
        
        # Display actual face images with Ring styling
        for i, (name, encoding) in enumerate(zip(self.known_face_names, self.known_face_encodings)):
            row = i // cols
            col = i % cols
            
            # Get the face image if available
            if i < len(self.known_face_images):
                face_img = self.known_face_images[i].copy()
            else:
                # Create a placeholder if no image available
                face_img = np.zeros((face_size, face_size, 3), dtype=np.uint8)
                face_img[:] = (50, 50, 50)
                # Draw a simple face icon
                center = face_size // 2
                cv2.circle(face_img, (center, center), 40, (255, 255, 255), -1)  # Head
                cv2.circle(face_img, (center - 15, center - 10), 5, (0, 0, 0), -1)  # Left eye
                cv2.circle(face_img, (center + 15, center - 10), 5, (0, 0, 0), -1)  # Right eye
                cv2.circle(face_img, (center, center + 10), 8, (0, 0, 0), -1)  # Mouth
            
            # Resize face image to standard size
            face_img_resized = cv2.resize(face_img, (face_size, face_size))
            
            # Add face to canvas with Ring-style border
            x_offset = col * face_size
            y_offset = title_height + row * face_size
            
            # Add Ring-style border
            border_color = self.colors['ring_blue']
            cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + face_size, y_offset + face_size), 
                         border_color, 3)
            
            # Add face image
            canvas[y_offset:y_offset + face_size, x_offset:x_offset + face_size] = face_img_resized
            
            # Store face position for mouse interaction
            self.face_positions.append({
                'index': i - 1,
                'x': x_offset,
                'y': y_offset,
                'w': face_size,
                'h': face_size,
                'name': name
            })
            
            # Add name below face with Ring styling
            text_y = y_offset + face_size + 25
            text_x = x_offset + face_size // 2
            (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Add name background
            cv2.rectangle(canvas, (text_x - text_width//2 - 10, text_y - text_height - 5), 
                         (text_x + text_width//2 + 10, text_y + 5), self.colors['ring_blue'], -1)
            cv2.putText(canvas, name, (text_x - text_width//2, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add delete button for each face
            delete_y = text_y + 30
            delete_text = "DELETE"
            (delete_width, delete_height), _ = cv2.getTextSize(delete_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Store delete button position
            self.face_positions[-1]['delete_x'] = text_x - delete_width//2
            self.face_positions[-1]['delete_y'] = delete_y
            self.face_positions[-1]['delete_w'] = delete_width + 20
            self.face_positions[-1]['delete_h'] = delete_height + 10
        
        # Set mouse callback for the face viewer
        cv2.setMouseCallback(face_window_name, self.on_face_viewer_click, canvas)
        
        # Display the canvas
        cv2.imshow(face_window_name, canvas)
        
        print("Click DELETE buttons to remove faces. Press any key to close...")
        
        # Wait for key press to close
        cv2.waitKey(0)
        cv2.destroyWindow(face_window_name)
        print("Face viewer closed.")
    
    def on_face_viewer_click(self, event, x, y, flags, param):
        """Handle mouse clicks in the face viewer"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on a delete button
            for face_data in self.face_positions:
                delete_x = face_data['delete_x']
                delete_y = face_data['delete_y']
                delete_w = face_data['delete_w']
                delete_h = face_data['delete_h']
                
                if (delete_x <= x <= delete_x + delete_w and 
                    delete_y - delete_h <= y <= delete_y):
                    
                    # Confirm deletion
                    name = face_data['name']
                    index = face_data['index']
                    
                    print(f"Deleting face: {name}")
                    
                    # Remove from database
                    if 0 <= index < len(self.known_face_names):
                        del self.known_face_names[index]
                        del self.known_face_encodings[index]
                        if index < len(self.known_face_images):
                            del self.known_face_images[index]
                        
                        # Save updated database
                        self.save_face_database()
                        print(f"Face '{name}' deleted successfully!")
                        
                        # Refresh the viewer
                        cv2.destroyAllWindows()
                        self.view_registered_faces()
                        return
    
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
        print("Security Camera System stopped")

def main():
    parser = argparse.ArgumentParser(description='Security Camera System')
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
        security_camera = SecurityCamera(camera_source, args.confidence)
        security_camera.run()
    except KeyboardInterrupt:
        print("\nSecurity Camera System stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 