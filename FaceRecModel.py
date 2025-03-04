import cv2
import face_recognition
import os
import time
import sqlite3
import numpy as np
from datetime import datetime

def capture_image():
    print("Starting the webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not access the camera.")
        exit()
    
    time.sleep(2.0)

    success, frame = cap.read()
    if not success:
        print("Failed to capture the frame. Please try again.")
            
    cap.release()
    
    return frame

def extract_face_data(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    return face_locations, face_encodings

def initialize_database():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    # Create table for storing face encodings
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL,
        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create table for attendance records
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id INTEGER,
        event_type TEXT CHECK(event_type IN ('checkin', 'checkout')),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (face_id) REFERENCES face_data (id)
    )
    ''')
    
    conn.commit()
    return conn

def store_face_data(conn, encodings, name):
    cursor = conn.cursor()
    
    for encoding in encodings:
        # Convert numpy array to bytes for storage
        encoding_bytes = encoding.tobytes()
        
        # Insert face data
        cursor.execute(
            "INSERT INTO face_data (name, encoding) VALUES (?, ?)",
            (name, encoding_bytes)
        )
        
        # Get the ID of the inserted face data
        face_id = cursor.lastrowid
    
    conn.commit()
    return face_id

def get_known_face_encodings(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding FROM face_data")
    
    known_face_ids = []
    known_face_names = []
    known_face_encodings = []
    
    for row in cursor.fetchall():
        face_id, name, encoding_bytes = row
        # Convert bytes back to numpy array
        face_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
        
        known_face_ids.append(face_id)
        known_face_names.append(name)
        known_face_encodings.append(face_encoding)
    
    return known_face_ids, known_face_names, known_face_encodings

def record_attendance(conn, face_id, event_type):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO attendance (face_id, event_type, timestamp) VALUES (?, ?, ?)",
        (face_id, event_type, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()

def attendance_monitor():
    conn = initialize_database()
    
    print("Starting attendance monitoring system...")
    print("Press 'r' to register a new face")
    print("Press 'c' to check-in the currently detected person")
    print("Press 'g' to check-out the currently detected person")
    print("Press 'q' to quit the system")
    
    known_face_ids, known_face_names, known_face_encodings = get_known_face_encodings(conn)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not access the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Lists to store recognized faces in current frame
        face_ids = []
        face_names = []
        
        for face_encoding in face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_id = None
            
            if True in matches:
                match_index = matches.index(True)
                face_id = known_face_ids[match_index]
                name = known_face_names[match_index]
            
            face_ids.append(face_id)
            face_names.append(name)
        
        # Display video feed with names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Display instructions
        cv2.putText(frame, "Press 'c' to check-in", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'g' to check-out", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Attendance System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Register new face if 'r' is pressed
        if key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            register_new_face(conn)
            # Refresh known faces
            known_face_ids, known_face_names, known_face_encodings = get_known_face_encodings(conn)
            cap = cv2.VideoCapture(0)
        
        # Check-in if 'c' is pressed
        elif key == ord('c'):
            if face_ids and face_ids[0] is not None:  # Process the first detected face
                record_attendance(conn, face_ids[0], 'checkin')
                print("Check-in recorded for" , face_names[0], "at" ,datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            else:
                print("No recognized face to check-in.")
        
        # Check-out if 'g' is pressed
        elif key == ord('g'):
            if face_ids and face_ids[0] is not None:  # Process the first detected face
                record_attendance(conn, face_ids[0], 'checkout')
                print("Check-out recorded for", face_names[0], "at" ,datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            else:
                print("No recognized face to check-out.")
        
        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

def register_new_face(conn):
    # Capture image from webcam
    image = capture_image()

    # Get person's name
    name = input("Enter the name of the person: ")

    # Extract face data
    face_locations, encodings = extract_face_data(image)

    if encodings: 
        face_id = store_face_data(conn, encodings, name)
        print("Facial data for" ,name, "stored successfully.")
    else:
        print("No faces found in the image.")

def main():
    print("Face Recognition Attendance System")
    print("1. Register New Face")
    print("2. Start Attendance Monitoring")
    choice = input("Enter your choice (1/2): ")
    
    conn = initialize_database()
    
    if choice == '1':
        register_new_face(conn)
    elif choice == '2':
        # Close the connection as attendance_monitor will open its own
        conn.close()
        attendance_monitor()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()