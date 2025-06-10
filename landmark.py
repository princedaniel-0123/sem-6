import cv2
import mediapipe as mp
import time
import csv
import logging
import os

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe modules
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video input (update path to match your local machine)
input_video_path = 'D:\Sign-Language-To-Text-and-Speech-Conversion-master\S11.mp4\S11.mp4'  # Adjust to your local path, e.g., 'C:/path/to/S11.mp4'
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file. Please ensure '{input_video_path}' exists.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video output
output_path = 'output_video_with_landmarks.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# CSV output
csv_file = 'landmarks.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Type', 'Landmark_ID', 'X', 'Y', 'Z'])

    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or error reading frame.")
                break

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}")

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Process landmarks
            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)

            # Convert back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks (first 25 for upper body)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Fallback style
                )
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark[:25]):
                    writer.writerow([frame_count, 'Pose', idx, landmark.x, landmark.y, landmark.z])

            # Draw hand landmarks
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        writer.writerow([frame_count, f'Hand_{hand_idx}', idx, landmark.x, landmark.y, landmark.z])

            # Draw face landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        writer.writerow([frame_count, 'Face', idx, landmark.x, landmark.y, landmark.z])

            # Add FPS text
            cv2.putText(frame_bgr, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write frame to output video
            out.write(frame_bgr)

            # Display frame in VS Code
            cv2.imshow('MediaPipe Landmarks', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

            # Flush CSV periodically
            if frame_count % 100 == 0:
                f.flush()

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        pose.close()
        hands.close()
        face_mesh.close()
        print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
        if os.path.exists(output_path):
            print(f"Output video saved to {output_path}")
        else:
            print("Output video file was NOT created!")
        if os.path.exists(csv_file):
            print(f"Landmarks saved to {csv_file}")
        else:
            print("CSV file was NOT created!")