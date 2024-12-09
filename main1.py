from ultralytics import YOLO
import cv2
import pyttsx3
import math

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' or any other YOLO model

# Initialize variables
PAUSE = False  # To pause/resume detection
DETECTION_THRESHOLD = 0.5  # Confidence threshold
detected_objects_list = set()  # To store detected objects and avoid repetition

# Camera parameters (for distance estimation)
FOCAL_LENGTH = 500  # Example focal length of the camera in pixels
KNOWN_OBJECT_WIDTH = 0.5  # Real-world width of the object (in meters, for example)

# Initialize object trackers
trackers = cv2.MultiTracker_create()

def speak(text):
    """
    Converts text to speech using pyttsx3 and directly speaks it.
    """
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error with text-to-speech: {e}")

def calculate_distance(object_width_in_pixels):
    """
    Estimate the distance from the camera to the object based on object size.
    """
    if object_width_in_pixels == 0:
        return None
    distance = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / object_width_in_pixels
    return round(distance, 2)

def detect_and_speak(frame):
    """
    Detect objects in a frame, provide audio feedback for detected objects.
    """
    global model, detected_objects_list, trackers

    results = model(frame)
    current_objects = []
    distances = []

    # Get the current detected objects and create trackers
    new_trackers = []
    for r in results[0].boxes.data.tolist():
        confidence = r[4]
        if confidence >= DETECTION_THRESHOLD:
            class_id = int(r[5])
            class_name = model.names[class_id]

            # Get bounding box coordinates (x, y, width, height)
            x1, y1, x2, y2 = map(int, r[:4])
            object_width_in_pixels = x2 - x1

            # Calculate the distance to the object
            distance = calculate_distance(object_width_in_pixels)
            if distance:
                distances.append(f"{class_name} at {distance} meters")
            else:
                distances.append(f"{class_name} at unknown distance")

            current_objects.append(class_name)

            # Initialize a tracker for each detected object
            tracker = cv2.TrackerCSRT_create()
            new_trackers.append((tracker, (x1, y1, x2 - x1, y2 - y1)))

    # Initialize MultiTracker with new trackers
    if new_trackers:
        trackers = cv2.MultiTracker_create()
        for tracker, bbox in new_trackers:
            trackers.add(tracker, frame, bbox)

    # Filter out new objects that haven't been announced yet
    new_objects = [obj for obj in current_objects if obj not in detected_objects_list]

    # Update the detected objects list
    detected_objects_list.update(new_objects)

    # Provide audio feedback for new detections
    if new_objects:
        speak_text = "I see " + ", ".join([f"a {obj}" for obj in new_objects])
        print(speak_text)
        speak(speak_text)

    # Provide distance information
    if distances:
        speak_distance = " ".join(distances)
        print(speak_distance)
        speak(speak_distance)

    # Update trackers
    success, boxes = trackers.update(frame)
    for i, box in enumerate(boxes):
        # Draw the bounding box for each tracked object
        x1, y1, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    return frame

def main():
    """
    Main function to capture video, detect objects, and provide audio feedback.
    """
    global PAUSE

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    print("Starting object detection. Press 'p' to pause/resume, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam.")
            break

        if not PAUSE:
            # Detect and annotate objects
            annotated_frame = detect_and_speak(frame)
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Paused", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Object Detection for Visually Impaired", annotated_frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Resume
            PAUSE = not PAUSE

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
