import math
import keyinput
import cv2
import mediapipe as mp
import socket

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Import the drawing utilities

# Create a Hands object
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# Initialize variables to track the last recognized gesture
last_gesture = None

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

            # Initialize variables to track which gesture is recognized
            is_thumb_up = False
            is_index_up = False
            is_middle_up = False
            is_ring_up = False
            is_pinky_up = False

            # Gesture recognition logic for each finger
            if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
                is_thumb_up = True
                gesture = "Thumb Up (Move Straight)"
            elif index_tip.y < middle_tip.y and index_tip.y < ring_tip.y and index_tip.y < pinky_tip.y:
                is_index_up = True
                gesture = "Index Finger Up (Turn Left)"
            elif middle_tip.y < index_tip.y and middle_tip.y < ring_tip.y and middle_tip.y < pinky_tip.y:
                is_middle_up = True
                gesture = "Middle Finger Up (Turn Right)"
            elif ring_tip.y < index_tip.y and ring_tip.y < middle_tip.y and ring_tip.y < pinky_tip.y:
                is_ring_up = True
                gesture = "Ring Finger Up (Move Back)"
            else:
                is_pinky_up = True
                gesture = "Pinky Finger Up (Unknown Gesture)"

            # Check if a new gesture is recognized and update the last recognized gesture
            if gesture != last_gesture:
                last_gesture = gesture
                print("Recognized Gesture: " + gesture)
                sock.sendto(gesture.encode(), serverAddressPort)

            # Perform actions based on recognized gestures
            if is_thumb_up:
                keyinput.press_key('w')
            else:
                keyinput.release_key('w')

            if is_index_up:
                keyinput.press_key('a')
            else:
                keyinput.release_key('a')

            if is_middle_up:

                keyinput.press_key('d')
            else:
                keyinput.release_key('d')

            if is_ring_up:
                keyinput.press_key('s')
            else:
                keyinput.release_key('s')

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
