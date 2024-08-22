#preparation here
from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt') 

def handle_request(req):
	payload, img = getReqData(req)

	results = model.predict(img, save=False, conf=0.5)[0] # only one person will be detected
	
	for result in results:
		keypoints = result.keypoints  
		print(keypoints)
		
	
	img = result.plot()
	
	resp = makeRespData(payload, img)
	
	return resp

