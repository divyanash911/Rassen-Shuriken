import cv2
import csv
import mediapipe as mp
import os
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image



def get_image(directory_path):
  file_paths = []
  for root, directories, files in os.walk(directory_path):
      for filename in files:
        file_path = os.path.join(root, filename)
        file_paths.append(file_path)
  return file_paths

IMAGE_FILES = get_image("./dataset/train")
# print(IMAGE_FILES)
csv_path = "data.csv"
row_anomaly = 0
with open(csv_path,mode="w") as filef:
  count = 0
  for idx, file in enumerate(IMAGE_FILES):
    label = file.removeprefix('./dataset/train/')
    label = label[:label.find('/'):1]
    # print(label)
    imgname = file.removeprefix('./dataset/train/')    
   
    print(file)
    imgname = imgname[imgname.find('/')+1::1]
    
    base_options =python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                          num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(file)
    # print(image)
    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    # print(image.numpy_view().shape)
    # STEP 5: Process the classification result. In this case, visualize it.
    if detection_result is not None and image.numpy_view().shape[2] == 3 and len(detection_result.hand_landmarks) == 2:
      annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
      cv2.imwrite(f"./annotated/{imgname}",annotated_image)
    
    results = detection_result
      
    # cv2.imshow('image',annotated_image)
    # Draw hand world landmarks.
    if not results.hand_landmarks:
      continue
    
    writer = csv.writer(filef)

    row_data = []
    count = 0
    # for hand_world_landmarks in results.hand_landmarks:
    #   print(hand_world_landmarks[0])
    if len(results.handedness) == 2:
      for hand_point in results.hand_landmarks:
        # print(len(hand_point))
        for point in hand_point:
          row_data.append(point.x)
          row_data.append(point.y)
          row_data.append(point.z)
    
      row_data.append(label)
      if len(row_data) != 127:
        row_anomaly += 1
        continue
      writer.writerow(row_data)
print(row_anomaly)

      


        
        



        