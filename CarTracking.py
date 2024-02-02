import math
import time
import keyinput
import cv2
import mediapipe as mp
import socket

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a Hands object
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,  # Adjusted min_detection_confidence
    min_tracking_confidence=0.5,   # Adjusted min_tracking_confidence
    max_num_hands=1
)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# Initialize variables to track the last recognized gesture
last_gesture = None
last_gesture_time = time.time()

# Gesture threshold and variation thresholds
gesture_threshold = 0.1  # Adjusted gesture_threshold
partial_gesture_threshold = 0.05  # Adjusted partial_gesture_threshold

def recognize_gesture(thumb_y, index_y, middle_y, ring_y, pinky_y, thumb_x, index_x, middle_x, ring_x, pinky_x):
    is_thumb_up = thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y and abs(thumb_x - index_x) > gesture_threshold
    is_index_up = index_y < middle_y and index_y < ring_y and index_y < pinky_y and abs(index_x - middle_x) > gesture_threshold
    is_middle_up = middle_y < index_y and middle_y < ring_y and middle_y < pinky_y and abs(middle_x - index_x) > gesture_threshold
    is_pinky_up = pinky_y < index_y and pinky_y < middle_y and pinky_y < ring_y and abs(pinky_x - index_x) > gesture_threshold

    is_thumb_half_up = thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y and abs(thumb_x - index_x) < partial_gesture_threshold
    is_index_half_up = index_y < middle_y and index_y < ring_y and index_y < pinky_y and abs(index_x - middle_x) < partial_gesture_threshold
    is_middle_half_up = middle_y < index_y and middle_y < ring_y and middle_y < pinky_y and abs(middle_x - index_x) < partial_gesture_threshold
    is_pinky_half_up = pinky_y < index_y and pinky_y < middle_y and pinky_y < ring_y and abs(pinky_x - index_x) < partial_gesture_threshold

    if is_thumb_up:
        return "Thumb Up (Move Straight)"
    elif is_index_up:
        return "Index Finger Up (Turn Right)"
    elif is_middle_up:
        return "Middle Finger Up (Turn Left)"
    elif is_pinky_up:
        return "Pinky Finger Up (Move Back)"
    elif is_thumb_half_up:
        return "Thumb Half Up (Partial Acceleration)"
    elif is_index_half_up:
        return "Index Finger Half Up (Partial Right Turn)"
    elif is_middle_half_up:
        return "Middle Finger Half Up (Partial Left Turn)"
    elif is_pinky_half_up:
        return "Pinky Finger Half Up (Partial Move Back)"
    else:
        return "No Gesture"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract coordinates of specific landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Get landmark coordinates
            thumb_y, index_y, middle_y, ring_y, pinky_y = [tip.y * image.shape[0] for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]]
            thumb_x, index_x, middle_x, ring_x, pinky_x = [tip.x * image.shape[1] for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]]

            # Recognize gesture
            gesture = recognize_gesture(thumb_y, index_y, middle_y, ring_y, pinky_y, thumb_x, index_x, middle_x, ring_x, pinky_x)

            # Check if a new gesture is recognized and update the last recognized gesture
            if gesture != last_gesture:
                last_gesture = gesture
                print("Recognized Gesture: " + gesture)
                sock.sendto(gesture.encode(), serverAddressPort)

                # Perform actions based on recognized gestures
                if "Thumb" in gesture:
                    keyinput.press_key('w')
                else:
                    keyinput.release_key('w')

                if "Index" in gesture:
                    keyinput.press_key('d')
                else:
                    keyinput.release_key('d')

                if "Middle" in gesture:
                    keyinput.press_key('a')
                else:
                    keyinput.release_key('a')

                if "Pinky" in gesture:
                    keyinput.press_key('s')
                else:
                    keyinput.release_key('s')

                last_gesture_time = time.time()  # Update last_gesture_time

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            # No hand detected, release all keys
            keyinput.release_key('w')
            keyinput.release_key('a')
            keyinput.release_key('s')
            keyinput.release_key('d')
            last_gesture = None

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()