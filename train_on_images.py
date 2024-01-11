import cv2
import csv
import mediapipe as mp
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.8) as hands:
  count = 0
  for idx, file in enumerate(IMAGE_FILES):
    label = file.removeprefix('./dataset/train/')
    label = label[:label.find('/'):1]
    # print(label)
    image = cv2.flip(cv2.imread(file), 1)

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()



    for hand_landmarks in results.multi_hand_landmarks:
      
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite('annotated_image/' + str(idx) + '.png', cv2.flip(annotated_image, 1))
      
    # cv2.imshow('image',annotated_image)
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    labels = []
    for count in range(1,43):
      labels.append("x"+str(count))
      labels.append("y"+str(count))
      labels.append("z"+str(count))
    
    with open(csv_path,mode="a") as file:
        writer = csv.writer(file)

        row_data = []
        count = 0
        for hand_world_landmarks in results.multi_hand_landmarks:
          if count == 1:
            if len(row_data) < 21 * 3 : 
              while len(row_data) < 21 * 3:
                row_data.append(0)
          
          for coordinates in hand_world_landmarks.landmark:
            count = count +1
            row_data.append(coordinates.x)
            row_data.append(coordinates.y)
            row_data.append(coordinates.z)
          
        if len(row_data) < 42*3:
          while len(row_data) < 42 * 3:
                row_data.append(0)
        row_data.append(label)
        # print(len(row_data))
        writer.writerow(row_data)
        

      


        
        



        