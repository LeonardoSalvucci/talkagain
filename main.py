import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from math import hypot
from speach import speach


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
  return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_blinking_ratio(eye_points, landmarks):
  left_point = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
  right_point = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
  center_top = midpoint(landmarks.part(eye_points[1]), landmarks.part(eye_points[2]))
  center_bottom = midpoint(landmarks.part(eye_points[5]), landmarks.part(eye_points[4]))

  def get_np_array_from_point(point):
    return np.array([point.x, point.y])

  A = dist.euclidean(get_np_array_from_point(landmarks.part(eye_points[1])), get_np_array_from_point(landmarks.part(eye_points[5])))
  B = dist.euclidean(get_np_array_from_point(landmarks.part(eye_points[2])), get_np_array_from_point(landmarks.part(eye_points[4])))
  C = dist.euclidean(get_np_array_from_point(landmarks.part(eye_points[0])), get_np_array_from_point(landmarks.part(eye_points[3])))

  hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
  ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

  hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
  ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
  ear = (A + B) / (2.0 * C)
  return ear

def get_gaze_ratio(eye_points, landmarks):
  eye_region = np.array([
    (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
    (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
    (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
    (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
    (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
    (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)
  ], np.int32)

  height, width, _ = frame.shape
  mask = np.zeros((height, width), np.uint8)
  cv2.polylines(mask, [eye_region], True, 255, 2)
  cv2.fillPoly(mask, [eye_region], 255)
  mask_eye = cv2.bitwise_and(gray, gray, mask=mask)

  # Eye rectangle area
  min_x = np.min(eye_region[:, 0])
  max_x = np.max(eye_region[:, 0])
  min_y = np.min(eye_region[:, 1])
  max_y = np.max(eye_region[:, 1])

  eye = mask_eye[min_y: max_y, min_x: max_x]
  _, threshold_eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)
  #threshold_eye = cv2.adaptiveThreshold(eye,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  #          cv2.THRESH_BINARY,11,2)
  height, width = threshold_eye.shape

  cv2.imshow("Eye Threshold", threshold_eye)

  left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
  left_side_white = cv2.countNonZero(left_side_threshold)

  right_side_threshold = threshold_eye[0: height, int(width/2): width]
  right_side_white = cv2.countNonZero(right_side_threshold)

  if (right_side_white > 0):
    return left_side_white / right_side_white
  return 0

def drawBoard(index=0):
  height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = 1600
  width = 3500
  board = np.zeros((height, width, 3), np.uint8)
  color_si = (255, 255, 255)
  color_no = (255, 255, 255)
  if(index == 1):
    color_si = (255, 0 ,0)
  elif(index == 2):
    color_no = (255, 0, 0)
  cv2.putText(board, "SI", (50, int(height/2)), font, 20, color_si, 10)
  cv2.putText(board, "NO", (2650, int(height/2)), font, 20, color_no, 10)
  cv2.imshow("Board", board)

def pose_estimate(image, landmarks):
    """
    Given an image and a set of facial landmarks generates the direction of pose
    """
    size = image.shape
    image_points = np.array([
        (landmarks.part(33).x, landmarks.part(33).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
        ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
        ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return p1, p2  

blinking_frames = 0
der_frames = 0
izq_frames = 0

si_no_selected_index = 0

speach("Inicializando")

drawBoard()

while True:
  _, frame = cap.read()
  frame += 1
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)
  for face in faces:
    # This is for printing a rectangle around face
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

    landmarks = predictor(gray, face)

    _, pose_prediction = pose_estimate(frame, landmarks)
    cv2.putText(frame, str(pose_prediction[0]), (50, 150), font, 2, (0, 0, 255), 3)
    
    # Detect Blinking
    left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
    right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

    cv2.putText(frame, str(blinking_ratio), (50, 100), font, 2, (0, 0, 255), 3)

    if blinking_ratio < 0.25:
      cv2.putText(frame, "BLINKING", (50, 150), font, 3, (0, 255, 0))
      blinking_frames += 1
      
      if blinking_frames > 2:
        if(si_no_selected_index == 1):
          speach("La respuesta es si")
        elif(si_no_selected_index == 2):
          speach("La respuesta es no")
    else:
      blinking_frames = 0

    # Pose prediction
    print(pose_prediction[0], si_no_selected_index)
    if(pose_prediction[0] < 600):
      si_no_selected_index = 2
    if(pose_prediction[0]>=600):
      si_no_selected_index = 1
               

    # Gaze detection
    gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
    gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

    gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

    if(gaze_ratio < 0.6):
      #cv2.putText(frame, "DERECHA", (50, 100), font, 2, (0, 0, 255), 3)
      der_frames += 1
      if(der_frames > 3 ):
        si_no_selected_index = 2
      if(der_frames == 8):
        #speach("Derecha")
        der_frames = 0
    elif 0.7 < gaze_ratio < 2:
      #cv2.putText(frame, "MEDIO", (50, 100), font, 2, (0, 0, 255), 3)
      der_frames = 0
      izq_frames = 0
    else:
      #cv2.putText(frame, "IZQUIERDA", (50, 100), font, 2, (0, 0, 255), 3)
      izq_frames += 1
      if(izq_frames > 3):
        si_no_selected_index = 1
      if(izq_frames == 8):
        #speach("Izquierda")
        izq_frames = 0
    
  drawBoard(si_no_selected_index)

  cv2.imshow("Frame", frame)

  key = cv2.waitKey(1)
  if key == 27:
    break

cap.relsease()
cv2.destroyAllWindows()
