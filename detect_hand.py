import mediapipe as mp 
import cv2 as cv
import numpy as np
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

res_obj = None

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    
def draw_landmarks_on_image(rgb_image, detection_result):
  if detection_result is None:
    return rgb_image
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
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print(result)
    global res_obj
    res_obj = result
    print('hand landmarker result: {}'.format(result))



if 1:
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret,frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            # Frame captured is stored in the frame variable which can be use for further image processing
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        options = HandLandmarkerOptions(
                  base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
                  running_mode=VisionRunningMode.LIVE_STREAM,
                  result_callback=print_result)
        
        with HandLandmarker.create_from_options(options) as landmarker:
          result_img =landmarker.detect_async(mp_image,mp.Timestamp.from_seconds(time.time()).value)

        
        

        if res_obj is not None:
            print(res_obj)
            annotated_image = draw_landmarks_on_image(frame, res_obj)
            cv.imshow('frame',annotated_image)
      
                
        if cv.waitKey(1) == ord('q'):
            break
        
        
    cap.release()
    cv.destroyAllWindows()

