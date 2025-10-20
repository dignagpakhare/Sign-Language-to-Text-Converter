## Abstract

Sign language is one of the oldest and most natural form of language for communication, but since most people do not know sign language and interpreters are very difficult to come by we have come up with a real time method using neural networks for fingerspelling based american sign language.
 
In this method, the hand is first passed through a filter and after the filter is applied the hand is passed through a classifier which predicts the class of the hand gestures. This method provides 98.00 % accuracy for the 26 letters of the alphabet.

## Project Description

American sign language is a predominant sign language Since the only disability D&M people have is communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. 

Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. 

Deaf and Mute(Dumb)(D&M) people make use of their hands to express different gestures to express their ideas with other people. 

Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language. 

Sign language is a visual language and consists of 3 major components 

<img width="712" height="208" alt="image" src="https://github.com/user-attachments/assets/7f66dab7-84f3-46d9-b256-1b869c75a205" />


In this project I basically focus on producing a model which can recognize Fingerspelling based hand gestures in order to form a complete word by combining each gesture. 

The gestures I  trained are as given in the image below.

<img width="1272" height="706" alt="image" src="https://github.com/user-attachments/assets/d04debd5-9b85-48af-835b-15c906c1ba93" />


<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/775978f9-2bd2-49c0-a913-8bbd90fb7ad1" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/a7ac8559-7776-4164-b4ad-7c4741328a89" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/527caa4f-a5e2-467c-8079-7cb5a44a10e8" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/fe959592-85ea-4e69-8519-bd0ee993154c" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/02a0c45a-51ca-49f5-a73f-e31c56c488a4" />


# Steps of building this project

### 1. The first Step of building this project was of creating the folders and make main.py file install all necessary libraries and boom!! run the code

``` python
# Importing the Libraries Required

import cv2
import mediapipe as mp

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture = ""

def y(landmarks, id): return landmarks[id].y
def x(landmarks, id): return landmarks[id].x

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = ""

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),  # Titik hitam
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)               # Garis kuning
            )

            lm = hand_landmarks.landmark

            # === Halo ===
            all_fingers_up = all(
                y(lm, tip) < y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if all_fingers_up:
                gesture = "Hello"

            # === OK ===
            thumb_index_close = (
                abs(x(lm, mp_hands.HandLandmark.THUMB_TIP) - x(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05 and
                abs(y(lm, mp_hands.HandLandmark.THUMB_TIP) - y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05
            )
            middle_up = y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            if thumb_index_close and middle_up:
                gesture = "OK"

            # === I Love You ===
            love_you = (
                y(lm, mp_hands.HandLandmark.THUMB_TIP) < y(lm, mp_hands.HandLandmark.THUMB_IP) and
                y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                y(lm, mp_hands.HandLandmark.PINKY_TIP) < y(lm, mp_hands.HandLandmark.PINKY_PIP) and
                y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) > y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                y(lm, mp_hands.HandLandmark.RING_FINGER_TIP) > y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            )
            if love_you:
                gesture = "I Love You"

            # === Peace (2 jari ke atas, lainnya turun) - support 2 arah
            index_up = y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
            middle_up = y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            ring_down = y(lm, mp_hands.HandLandmark.RING_FINGER_TIP) > y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            pinky_down = y(lm, mp_hands.HandLandmark.PINKY_TIP) > y(lm, mp_hands.HandLandmark.PINKY_PIP)
            thumb_down = y(lm, mp_hands.HandLandmark.THUMB_TIP) > y(lm, mp_hands.HandLandmark.THUMB_IP)

            if index_up and middle_up and ring_down and pinky_down and thumb_down:
                gesture = "Peace"

            # === Fist ===
            all_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if all_folded:
                gesture = "Fist"

            # === Sip (Jempol) - support 2 sisi ===
            thumb_up = y(lm, mp_hands.HandLandmark.THUMB_TIP) < y(lm, mp_hands.HandLandmark.THUMB_IP)
            others_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if thumb_up and others_folded and not all_folded:
                gesture = "Good"

    # Tampilkan teks di layar
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 3)  # Warna teks hitam

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()










