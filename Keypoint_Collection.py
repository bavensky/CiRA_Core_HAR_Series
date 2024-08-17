#preparation here
from ultralytics import YOLO
import numpy as np
from pydantic import BaseModel

import xlsxwriter
import pandas as pd

model = YOLO('yolov8n-pose.pt') 

keypoint_path = 'C:/Users/.../Activity Dataset/sitting.xlsx'	

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
	readDataframe = pd.read_excel(keypoint_path, engine='openpyxl')

	results = model.predict(img, save=False, conf=0.5)
	
	for result in results:
		keypoints = result.keypoints.xyn.numpy() 
		for keypoint in keypoints:
			if len(keypoint) <= 17:
				keypoint_list = extract_keypoint(keypoint)
				print(keypoint_list)
				
				nose = keypoint_list[0], keypoint_list[1]
				lEye = keypoint_list[2], keypoint_list[3]
				rEye = keypoint_list[4], keypoint_list[5]
				lEar = keypoint_list[6], keypoint_list[7]
				rEar = keypoint_list[8], keypoint_list[9]
				lShoulder = keypoint_list[10], keypoint_list[11]
				rShoulder = keypoint_list[12], keypoint_list[13]
				lElbow = keypoint_list[14], keypoint_list[15]
				rElbow = keypoint_list[16], keypoint_list[17]
				lWrist = keypoint_list[18], keypoint_list[19]
				rWrist = keypoint_list[20], keypoint_list[21]
				lHip = keypoint_list[22], keypoint_list[23]
				rHip = keypoint_list[24], keypoint_list[25]
				lKnee = keypoint_list[26], keypoint_list[27]
				rKnee = keypoint_list[28], keypoint_list[29]
				lAnkle = keypoint_list[30], keypoint_list[31]
				rAnkle = keypoint_list[32], keypoint_list[33]
				
				newDataframe = pd.DataFrame({ "pose": ["walking"], 
				"nose_x": [str(nose[0])], "nose_y": [str(nose[1])], 
				"lEye_x": [str(lEye[0])], "lEye_y": [str(lEye[1])],
				"rEye_x": [str(rEye[0])], "rEye_y": [str(rEye[1])], 
				"lEar_x": [str(lEar[0])], "lEar_y": [str(lEar[1])],
				"rEar_x": [str(rEar[0])], "rEar_y": [str(rEar[1])],
				"lShoulder_x": [str(lShoulder[0])], "lShoulder_y": [str(lShoulder[1])],
				"rShoulder_x": [str(rShoulder[0])], "rShoulder_y": [str(lShoulder[1])],
				"lElbow_x": [str(lElbow[0])], "lElbow_y": [str(lElbow[1])],
				"rElbow_x": [str(rElbow[0])], "rElbow_y": [str(rElbow[1])],
				"lWrist_x": [str(lWrist[0])], "lWrist_y": [str(lWrist[1])],
				"rWrist_x": [str(rWrist[0])], "rWrist_y": [str(rWrist[1])],
				"lHip_x": [str(lHip[0])], "lHip_y": [str(lHip[1])],
				"rHip_x": [str(rHip[0])], "rHip_y": [str(rHip[1])],
				"lKnee_x": [str(lKnee[0])], "lKnee_y": [str(lKnee[1])],
				"rKnee_x": [str(rKnee[0])], "rKnee_y": [str(rKnee[1])],
				"lAnkle_x": [str(lAnkle[0])], "lAnkle_y": [str(lAnkle[1])],
				"rAnkle_x": [str(rAnkle[0])], "rAnkle_y": [str(rAnkle[1])]})
				
				conC = pd.concat([readDataframe, newDataframe],  axis=0)
				conC.to_excel(keypoint_path, index=False, engine='openpyxl')

	img = result.plot()
	
	resp = makeRespData(payload, img)
	
	return resp