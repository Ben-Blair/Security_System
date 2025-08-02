#!/usr/bin/env python3
"""
Face Database Viewer - View and manage registered faces
"""

import pickle
import os
import cv2
import numpy as np
from datetime import datetime

def view_face_database():
    """View all registered faces in the database"""
    database_file = "face_database.pkl"
    
    if not os.path.exists(database_file):
        print("No face database found!")
        print("Register some faces first using the security camera system.")
        return
    
    try:
        with open(database_file, 'rb') as f:
            data = pickle.load(f)
            encodings = data['encodings']
            names = data['names']
        
        print("=" * 50)
        print("REGISTERED FACES DATABASE")
        print("=" * 50)
        
        if not names:
            print("No faces registered yet.")
            return
        
        for i, (name, encoding) in enumerate(zip(names, encodings), 1):
            print(f"{i}. {name}")
            print(f"   - Encoding shape: {encoding.shape}")
            print(f"   - Registered: {len(encoding)} features")
        
        print(f"\nTotal registered faces: {len(names)}")
        print(f"Database file: {database_file}")
        print(f"File size: {os.path.getsize(database_file)} bytes")
        
    except Exception as e:
        print(f"Error reading database: {e}")

def rename_faces():
    """Rename faces in the database"""
    database_file = "face_database.pkl"
    
    if not os.path.exists(database_file):
        print("No face database found!")
        return
    
    try:
        with open(database_file, 'rb') as f:
            data = pickle.load(f)
            encodings = data['encodings']
            names = data['names']
        
        if not names:
            print("No faces to rename.")
            return
        
        print("\nCurrent faces:")
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
        
        print("\nTo rename a face, enter the number and new name (e.g., '1 John'):")
        print("Press Enter to skip, 'q' to quit")
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() == 'q':
                    break
                if not user_input:
                    continue
                
                parts = user_input.split()
                if len(parts) != 2:
                    print("Format: <number> <new_name>")
                    continue
                
                index = int(parts[0]) - 1
                new_name = parts[1]
                
                if 0 <= index < len(names):
                    old_name = names[index]
                    names[index] = new_name
                    print(f"Renamed '{old_name}' to '{new_name}'")
                    
                    # Save updated database
                    data = {'encodings': encodings, 'names': names}
                    with open(database_file, 'wb') as f:
                        pickle.dump(data, f)
                    print("Database updated!")
                else:
                    print("Invalid number!")
                    
            except ValueError:
                print("Invalid input! Use format: <number> <new_name>")
            except KeyboardInterrupt:
                break
    
    except Exception as e:
        print(f"Error: {e}")

def delete_faces():
    """Delete faces from the database"""
    database_file = "face_database.pkl"
    
    if not os.path.exists(database_file):
        print("No face database found!")
        return
    
    try:
        with open(database_file, 'rb') as f:
            data = pickle.load(f)
            encodings = data['encodings']
            names = data['names']
        
        if not names:
            print("No faces to delete.")
            return
        
        print("\nCurrent faces:")
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
        
        print("\nTo delete a face, enter the number:")
        print("Press Enter to skip, 'q' to quit")
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() == 'q':
                    break
                if not user_input:
                    continue
                
                index = int(user_input) - 1
                
                if 0 <= index < len(names):
                    deleted_name = names[index]
                    del names[index]
                    del encodings[index]
                    print(f"Deleted '{deleted_name}'")
                    
                    # Save updated database
                    data = {'encodings': encodings, 'names': names}
                    with open(database_file, 'wb') as f:
                        pickle.dump(data, f)
                    print("Database updated!")
                    
                    if not names:
                        print("No faces left in database.")
                        break
                else:
                    print("Invalid number!")
                    
            except ValueError:
                print("Invalid input! Enter a number.")
            except KeyboardInterrupt:
                break
    
    except Exception as e:
        print(f"Error: {e}")

def backup_database():
    """Create a backup of the face database"""
    database_file = "face_database.pkl"
    
    if not os.path.exists(database_file):
        print("No face database to backup!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"face_database_backup_{timestamp}.pkl"
    
    try:
        import shutil
        shutil.copy2(database_file, backup_file)
        print(f"Database backed up to: {backup_file}")
    except Exception as e:
        print(f"Backup failed: {e}")

def main():
    """Main menu for face database management"""
    while True:
        print("\n" + "=" * 50)
        print("FACE DATABASE MANAGER")
        print("=" * 50)
        print("1. View registered faces")
        print("2. Rename faces")
        print("3. Delete faces")
        print("4. Backup database")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            view_face_database()
        elif choice == '2':
            rename_faces()
        elif choice == '3':
            delete_faces()
        elif choice == '4':
            backup_database()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-5.")

if __name__ == "__main__":
    main() 