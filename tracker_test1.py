from norfair import Detection, Tracker
from pathlib import Path
import torch
import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Twist

YOLOV5_PATH = Path("./")

#load model
def load_model(weights="./yolov5n.pt"):
	model = torch.hub.load(str(YOLOV5_PATH), 'custom', path=weights, source='local')
	return model

model = load_model()

#run detections on one frame
def get_dects(img, model):
	results = model(img)
	detections = results.xyxy[0].cpu().numpy()

	formatted_detections = []

	for *xyxy, conf, cls in detections:
		x_center = (xyxy[2] + xyxy[0]) / 2
		y_center = (xyxy[3] + xyxy[1]) / 2
		width = xyxy[2] - xyxy[0]
		height = xyxy[3] - xyxy[1]
		formatted_detections.append((x_center, y_center, width, height))

	return formatted_detections

# Function to convert yolov5 detections to norfair detections
def yolov5_to_norfair_detections(raw_detections):
	detections = []
	for raw_detection in raw_detections:
		x_center, y_center, width, height = raw_detection
		detections.append(Detection(points=np.array([[x_center, y_center]]), scores=np.array([1.0]), data={'size': np.array([width, height])}))
	return detections

# Set up a tracker
# Here, we still need to define the distance_function
tracker = Tracker(distance_function="euclidean", distance_threshold=100)

cap = cv2.VideoCapture(1)

rospy.init_node('jackal_velocity_controller')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

while cap.isOpened():
	# Read a video frame
	ret, frame = cap.read()
	if not ret:
		break

	# Get detections
	# raw_detections = get_yolov5_detections(frame)
	raw_detections = get_dects(frame, model)
 
	# Convert raw detections to norfair detections
	detections = yolov5_to_norfair_detections(raw_detections)

	# Update the tracker based on the new detections
	objects_tracked = tracker.update(detections=detections)

	# Now you can use objects_tracked which contains the tracked objects in the frame
	# The track.points gives the centroid of the tracked object
	for track in objects_tracked:
		# x, y = track.points[-1] 	# Take the latest point in the tracked object
		x, y = track.estimate[0]
		# Use cv2 to draw on your image here, for example:
		cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

		twist = Twist()

		twist.angular.z = (x - frame.shape[1]/2) * 0.01

		pub.publish(twist)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

