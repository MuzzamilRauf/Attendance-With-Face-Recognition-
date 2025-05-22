import face_recognition
import cv2
import numpy as np
import pandas as pd
import openpyxl
from datetime import datetime
import os
import winsound  # For beep notifications on Windows (alternative for other OSes)

# ----------- Utility Functions -----------

def load_known_faces(resource_folder="Resources"):
    """
    Load images and encodings from the Resources folder.
    Returns known_face_encodings and known_face_names lists.
    """
    known_face_encodings = []
    known_face_names = []

    # You can add more known faces here with their filenames and names
    known_faces = {
        "Bills1.jpg": "Bill-Gates",
        "Ellon1.jpg": "Ellon-Musk",
        "Amjad.jpg": "Amjad",
        "Meer.jpg": "Meer",
        "Rashid.jpg": "Rashid",
        "My.jpg": "Muzzamil"
    }

    for filename, name in known_faces.items():
        path = os.path.join(resource_folder, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"WARNING: No face found in image {filename}!")

    return known_face_encodings, known_face_names


def align_face(frame, face_location):
    """
    Simple face alignment placeholder.
    For better accuracy, you can integrate dlib or mediapipe for face landmarks.
    Here, just returns the cropped face.
    """
    top, right, bottom, left = face_location
    # Scale back up face locations since frame is resized
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    face_image = frame[top:bottom, left:right]
    return face_image


def notify_recognition(name):
    """
    Notification on recognition.
    Here, prints and plays beep (Windows only).
    """
    print(f"[NOTIFICATION] Recognized: {name}")
    try:
        # Frequency, Duration (ms)
        winsound.Beep(1000, 200)
    except:
        # If winsound unavailable (Linux/macOS), just print notification
        pass


def update_attendance(name, attendance_list, attendance_df):
    """
    Add recognized name and current time to attendance if not already present.
    Returns updated list and DataFrame.
    """
    if name not in attendance_list:
        attendance_list.append(name)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        attendance_df = pd.concat([attendance_df, pd.DataFrame([{"Name": name, "Time": current_time}])], ignore_index=True)
        notify_recognition(name)
    return attendance_list, attendance_df


def save_attendance(attendance_df, output_folder=".", prefix="Attendance"):
    """
    Saves attendance DataFrame to CSV and Excel with timestamped filenames.
    """
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_folder, f"{prefix}_{date_time_str}.csv")
    excel_filename = os.path.join(output_folder, f"{prefix}_{date_time_str}.xlsx")

    attendance_df.to_csv(csv_filename, index=False)
    attendance_df.to_excel(excel_filename, index=False)
    print(f"[INFO] Attendance saved to:\n CSV: {csv_filename}\n Excel: {excel_filename}")


def process_frame(frame, known_face_encodings, known_face_names, attendance_list, attendance_df, process_this_frame):
    """
    Detect faces, recognize, update attendance.
    Returns face locations, names, updated attendance list, DataFrame.
    """
    face_locations = []
    face_encodings = []
    face_names = []

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            # Update attendance if recognized
            if name != "Unknown":
                attendance_list, attendance_df = update_attendance(name, attendance_list, attendance_df)

    return face_locations, face_names, attendance_list, attendance_df


def display_results(frame, face_locations, face_names):
    """
    Draw rectangles and names on the frame.
    """
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale locations back to original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


def main():
    # Initialization
    attendance_list = []
    attendance_df = pd.DataFrame(columns=["Name", "Time"])
    output_folder = "."  # Change if you want to save elsewhere

    known_face_encodings, known_face_names = load_known_faces()

    video_capture = cv2.VideoCapture(0)
    process_this_frame = True

    print("[INFO] Starting attendance system. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Process frame for recognition
        face_locations, face_names, attendance_list, attendance_df = process_frame(
            frame,
            known_face_encodings,
            known_face_names,
            attendance_list,
            attendance_df,
            process_this_frame,
        )
        process_this_frame = not process_this_frame

        # Display rectangles and names
        display_results(frame, face_locations, face_names)

        # Real-time attendance display in console
        print(f"\r[Attendance] {attendance_list}", end="")

        # Show video frame
        cv2.imshow('Video', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Quitting...")
            break

    # Save attendance after exiting
    save_attendance(attendance_df, output_folder)

    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
