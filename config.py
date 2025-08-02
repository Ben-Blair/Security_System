# Security Camera System Configuration

# Camera Settings
CAMERA_SETTINGS = {
    'default_source': 0,  # 0 for webcam, or IP camera URL
    'frame_width': 640,
    'frame_height': 480,
    'fps': 30
}

# Detection Settings
DETECTION_SETTINGS = {
    'confidence_threshold': 0.5,
    'face_detection_scale': 1.1,
    'face_detection_min_neighbors': 5,
    'face_detection_min_size': (30, 30),
    'person_detection_scale': 1.05,
    'person_detection_win_stride': (8, 8),
    'person_detection_padding': (4, 4)
}

# Recognition Settings
RECOGNITION_SETTINGS = {
    'face_recognition_tolerance': 0.6,
    'face_database_file': 'face_database.pkl'
}

# Alert Settings
ALERT_SETTINGS = {
    'lingering_threshold': 5.0,  # seconds
    'face_hiding_threshold': 3.0,  # seconds
    'suspicious_activity_cooldown': 10.0,  # seconds
    'tracking_timeout': 10.0  # seconds
}

# Visualization Settings
VISUALIZATION_SETTINGS = {
    'colors': {
        'person': (0, 255, 0),      # Green
        'face': (255, 0, 0),        # Blue
        'suspicious': (0, 0, 255),  # Red
        'lingering': (0, 255, 255), # Yellow
        'unknown': (128, 128, 128)  # Gray
    },
    'line_thickness': 2,
    'font_scale': 0.5,
    'font_thickness': 2
}

# Recording Settings
RECORDING_SETTINGS = {
    'save_screenshots': True,
    'screenshot_format': 'jpg',
    'video_recording': False,
    'video_codec': 'XVID',
    'video_fps': 20
}

# IP Camera Settings (for future use)
IP_CAMERA_SETTINGS = {
    'rtsp_url_template': 'rtsp://{username}:{password}@{ip}:{port}/stream',
    'http_url_template': 'http://{ip}:{port}/video',
    'connection_timeout': 10,
    'reconnect_attempts': 3
} 