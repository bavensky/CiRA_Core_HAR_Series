#preparation here
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt') 

class GetKeypoint(BaseModel):
	NOSE:           int = 0
	LEFT_EYE:       int = 1
	RIGHT_EYE:      int = 2
	LEFT_EAR:       int = 3
	RIGHT_EAR:      int = 4
	LEFT_SHOULDER:  int = 5
	RIGHT_SHOULDER: int = 6
	LEFT_ELBOW:     int = 7
	RIGHT_ELBOW:    int = 8
	LEFT_WRIST:     int = 9
	RIGHT_WRIST:    int = 10
	LEFT_HIP:       int = 11
	RIGHT_HIP:      int = 12
	LEFT_KNEE:      int = 13
	RIGHT_KNEE:     int = 14
	LEFT_ANKLE:     int = 15
	RIGHT_ANKLE:    int = 16

get_keypoint = GetKeypoint()

def extract_keypoint(keypoint):
	# nose
	nose_x, nose_y = keypoint[get_keypoint.NOSE]
	# eye
	left_eye_x, left_eye_y = keypoint[get_keypoint.LEFT_EYE]
	right_eye_x, right_eye_y = keypoint[get_keypoint.RIGHT_EYE]
	# ear
	left_ear_x, left_ear_y = keypoint[get_keypoint.LEFT_EAR]
	right_ear_x, right_ear_y = keypoint[get_keypoint.RIGHT_EAR]
	# shoulder
	left_shoulder_x, left_shoulder_y = keypoint[get_keypoint.LEFT_SHOULDER]
	right_shoulder_x, right_shoulder_y = keypoint[get_keypoint.RIGHT_SHOULDER]
	# elbow
	left_elbow_x, left_elbow_y = keypoint[get_keypoint.LEFT_ELBOW]
	right_elbow_x, right_elbow_y = keypoint[get_keypoint.RIGHT_ELBOW]
	# wrist
	left_wrist_x, left_wrist_y = keypoint[get_keypoint.LEFT_WRIST]
	right_wrist_x, right_wrist_y = keypoint[get_keypoint.RIGHT_WRIST]
	# hip
	left_hip_x, left_hip_y = keypoint[get_keypoint.LEFT_HIP]
	right_hip_x, right_hip_y = keypoint[get_keypoint.RIGHT_HIP]
	# knee
	left_knee_x, left_knee_y = keypoint[get_keypoint.LEFT_KNEE]
	right_knee_x, right_knee_y = keypoint[get_keypoint.RIGHT_KNEE]
	# ankle
	left_ankle_x, left_ankle_y = keypoint[get_keypoint.LEFT_ANKLE]
	right_ankle_x, right_ankle_y = keypoint[get_keypoint.RIGHT_ANKLE]
	
	return [
		nose_x, nose_y,
		left_eye_x, left_eye_y,
		right_eye_x, right_eye_y,
		left_ear_x, left_ear_y,
		right_ear_x, right_ear_y,
		left_shoulder_x, left_shoulder_y,
		right_shoulder_x, right_shoulder_y,
		left_elbow_x, left_elbow_y,
		right_elbow_x, right_elbow_y,
		left_wrist_x, left_wrist_y,
		right_wrist_x, right_wrist_y,
		left_hip_x, left_hip_y,
		right_hip_x, right_hip_y,
		left_knee_x, left_knee_y,
		right_knee_x, right_knee_y,        
		left_ankle_x, left_ankle_y,
		right_ankle_x, right_ankle_y
	]
	

def handle_request(req):
	payload, img = getReqData(req)

	results = model.predict(img, save=False, conf=0.5)
	
	for result in results:
		keypoints = result.keypoints.xyn.numpy() 
		for keypoint in keypoints:
			nose_x, nose_y = keypoint[0]
			print("nose_x= ", nose_x , " and nose_y = ", nose_y)

	img = result.plot()
	
	resp = makeRespData(payload, img)
	
	return resp

