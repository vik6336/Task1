import cv2
import face_recognition
import pickle
import os
import time

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

def store_face_data(encodings, name, output_path='facial_data.dat'):
  
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {"encodings": [], "names": []}


    data["encodings"].extend(encodings)
    data["names"].append(name)


    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def main():

    image = capture_image()

    name = input("Enter the name of the person: ")

    face_locations, encodings = extract_face_data(image)

    if encodings: 
        store_face_data(encodings, name)
        print("Facial data stored successfully.")
    else:
        print("No faces found in the image.")

if __name__ == "__main__":
    main()