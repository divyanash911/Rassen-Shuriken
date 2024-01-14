import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense , Flatten , Conv2D
from tensorflow.keras import models 
import cv2
import mediapipe as mp
from preprocess import preprocess_data
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

labels = []
for count in range(1,43):
    labels.append("x"+str(count))
    labels.append("y"+str(count))
    labels.append("z"+str(count))

labels.append("sign")
data = pd.read_csv("data.csv",names=labels)

reference = {"bird":0,"boar":1,"dog":2,"dragon":3,"hare":4,"horse":5,"monkey":6,"ox":7,"ram":8,"rat":9,"snake":10,"tiger":11,"zero":12}
data.sign = data.sign.map(reference)
# print(data.sign)
## Setting up data values
y = data.sign
y =tf.keras.utils.to_categorical(y, num_classes=13)
x = []
for count in range(1,43):
    x.append("x"+str(count))
    x.append("y"+str(count))
    x.append("z"+str(count))
# print(y)
X = data[x]

print(X.shape)
# Define the model
normalize = layers.Normalization()
normalize.adapt(X)
model = models.Sequential([normalize])
model.add(layers.InputLayer(input_shape=(126,)))
# Add a dense hidden layer with 128 neurons and 'relu' activation function
model.add(layers.Dense(64, activation='relu'))

# Add another dense hidden layer with 64 neurons and 'relu' activation function
model.add(layers.Dense(32, activation='relu'))

# Output layer with appropriate activation function for your task (e.g., sigmoid for binary classification, softmax for multiclass)
# Adjust the number of units based on your task requirements
model.add(layers.Dense(13, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
# model.summary()
## Compiling model
checkpoint_path = "./cp.ckpt"


history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
model.save_weights("weights.h5")
# test_data = pd.read_csv("test.csv",names=labels)

# y_test = test_data.sign.map(reference)
# y_test =tf.keras.utils.to_categorical(y_test, num_classes=13)
# X_test = test_data[x]

# # model.evaluate(X_test, y_test, verbose=2)
# predictions = model.predict(X_test)

# print("accuracy: ",np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)))


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    
    preprocess_data(results,model)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()