#!/usr/bin/env python3
"""
IP Camera Utilities for Security Camera System
"""

import cv2
import requests
import time
from urllib.parse import urlparse
import socket

class IPCameraManager:
    def __init__(self):
        self.camera_configs = {
            'generic_rtsp': {
                'template': 'rtsp://{username}:{password}@{ip}:{port}/stream',
                'default_port': 554,
                'default_username': 'admin',
                'default_password': 'admin'
            },
            'generic_http': {
                'template': 'http://{ip}:{port}/video',
                'default_port': 8080
            },
            'hikvision': {
                'template': 'rtsp://{username}:{password}@{ip}:554/Streaming/Channels/101',
                'default_port': 554,
                'default_username': 'admin',
                'default_password': '12345'
            },
            'dahua': {
                'template': 'rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0',
                'default_port': 554,
                'default_username': 'admin',
                'default_password': 'admin'
            }
        }
    
    def test_camera_connection(self, url, timeout=5):
        """Test if an IP camera is accessible"""
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return True, f"Camera accessible - Frame shape: {frame.shape}"
                else:
                    return False, "Camera accessible but no frame received"
            else:
                return False, "Could not open camera stream"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def scan_network_cameras(self, network_prefix, ports=[554, 8080, 80]):
        """Scan network for potential IP cameras"""
        print(f"Scanning network {network_prefix}.* for cameras...")
        found_cameras = []
        
        for port in ports:
            for i in range(1, 255):
                ip = f"{network_prefix}.{i}"
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((ip, port))
                    sock.close()
                    
                    if result == 0:
                        print(f"Found device at {ip}:{port}")
                        # Test common camera URLs
                        test_urls = [
                            f"rtsp://admin:admin@{ip}:{port}/stream",
                            f"rtsp://admin:admin@{ip}:{port}/live",
                            f"http://{ip}:{port}/video",
                            f"http://{ip}:{port}/live"
                        ]
                        
                        for url in test_urls:
                            success, message = self.test_camera_connection(url, timeout=3)
                            if success:
                                found_cameras.append({
                                    'ip': ip,
                                    'port': port,
                                    'url': url,
                                    'message': message
                                })
                                print(f"✓ Camera found: {url}")
                                break
                
                except Exception as e:
                    continue
        
        return found_cameras
    
    def generate_camera_url(self, camera_type, ip, username=None, password=None, port=None):
        """Generate camera URL based on type and parameters"""
        if camera_type not in self.camera_configs:
            raise ValueError(f"Unknown camera type: {camera_type}")
        
        config = self.camera_configs[camera_type]
        
        # Use defaults if not provided
        if username is None:
            username = config.get('default_username', 'admin')
        if password is None:
            password = config.get('default_password', 'admin')
        if port is None:
            port = config.get('default_port', 554)
        
        return config['template'].format(
            ip=ip,
            port=port,
            username=username,
            password=password
        )
    
    def list_supported_cameras(self):
        """List supported camera types"""
        print("Supported camera types:")
        for camera_type in self.camera_configs:
            config = self.camera_configs[camera_type]
            print(f"  - {camera_type}: {config['template']}")

def main():
    """Test IP camera utilities"""
    manager = IPCameraManager()
    
    print("IP Camera Utilities")
    print("=" * 30)
    
    # List supported cameras
    manager.list_supported_cameras()
    
    # Example usage
    print("\nExample camera URLs:")
    
    # Generic RTSP
    url1 = manager.generate_camera_url('generic_rtsp', '192.168.1.100')
    print(f"Generic RTSP: {url1}")
    
    # Hikvision
    url2 = manager.generate_camera_url('hikvision', '192.168.1.101')
    print(f"Hikvision: {url2}")
    
    # Test connection (commented out to avoid errors)
    # success, message = manager.test_camera_connection(url1)
    # print(f"Connection test: {message}")

if __name__ == "__main__":
    main() 