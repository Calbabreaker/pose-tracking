import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import numpy as np
import mediapipe as mp
import argparse
import queue
import pythonosc
from pythonosc import udp_client

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=9000,
                    help="The port the OSC server is listening on")
parser.add_argument("--frametime", type=int, default=100,
                    help="Amount of time to wait between every frame")
parser.add_argument("--visualize", type=bool, default=True,
                    help="Whether or not to visualize the pose tracking output")
parser.add_argument("--cam", type=int, default=0,
                    help="The index of the video (/dev/videoX)")
args = parser.parse_args()

image_queue = queue.Queue()
osc_client = udp_client.SimpleUDPClient(args.ip, args.port)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
)
detector = PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for pose_landmarks in detection_result.pose_landmarks:
        print(len(pose_landmarks))

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def result_callback(detection_result, image, timestamp):
    if args.visualize:
        annotated_image = draw_landmarks_on_image(
            image.numpy_view(), detection_result)
        image_queue.put(annotated_image)

def send_osc_messages():
    tracker_addresses = {
        "head": "/tracking/trackers/head",
        "left_hand": "/tracking/trackers/2",
        "right_hand": "/tracking/trackers/3",
        "hips": "/tracking/trackers/4",
        "left_foot": "/tracking/trackers/5",
        "right_foot": "/tracking/trackers/6",
        "spine": "/tracking/trackers/7",
        "chest": "/tracking/trackers/8"
    }


def track_video():
    prev_timestamp = 0
    video = cv2.VideoCapture(args.cam)

    while True:
        # Get frame from video feed
        ok, frame = video.read()
        if not ok:
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp <= prev_timestamp:
            continue
        prev_timestamp = timestamp

        detection_result = detector.detect_async(image, timestamp)

        if args.visualize:
            try:
                image = image_queue.get_nowait()
                cv2.imshow("Image", image)
                cv2.waitKey(args.frametime)
                continue
            except queue.Empty:
                pass

        time.sleep(1 / args.frametime)


if __name__ == "__main__":
    track_video()
    cv2.destroyAllWindows()
