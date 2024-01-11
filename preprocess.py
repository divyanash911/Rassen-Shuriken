import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def preprocess_data(result):

    if result is None:
        return
    
    print(result.multi_hand_landmarks)

    


