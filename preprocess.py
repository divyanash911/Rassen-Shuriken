import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense , Flatten , Conv2D
from tensorflow.keras import models 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

reference = {"bird":0,"boar":1,"dog":2,"dragon":3,"hare":4,"horse":5,"monkey":6,"ox":7,"ram":8,"rat":9,"snake":10,"tiger":11,"zero":12}

def preprocess_data(result,model):

    if result.multi_hand_landmarks is None:
        return
    else:
        # print(len(result.multi_hand_world_landmarks))
        row_data = []
        if len(result.multi_hand_world_landmarks) != 2: 
            return
        for hand in result.multi_hand_world_landmarks:
            for point in hand.landmark:
                row_data.append(point.x)
                row_data.append(point.y)
                row_data.append(point.z)
        df = pd.DataFrame(row_data)
        input_data = df.to_numpy()

# Reshape the input for prediction
        input_data = input_data.reshape(1, -1)
        prediction = model.predict(input_data)
        for key, value in reference.items():
            if value == prediction.argmax():
                print(key)
                return key  
                






