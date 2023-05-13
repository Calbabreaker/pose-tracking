import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import numpy as np
import mediapipe as mp
import queue

FRAME_TIME = 10
VISUALIZE = True

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

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

image_queue = queue.Queue()

def result_callback(detection_result, image, timestamp):
    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    if VISUALIZE:
        image_queue.put(annotated_image)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = result_callback,
)
detector = PoseLandmarker.create_from_options(options)

def track_video():
    prev_timestamp = 0
    video = cv2.VideoCapture(7)

    while True:
        # Get frame from video feed
        ok,frame=video.read()
        if not ok: 
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp <= prev_timestamp:
            continue
        prev_timestamp = timestamp

        detection_result = detector.detect_async(image, timestamp)
 
        if VISUALIZE:
            try:
                image = image_queue.get_nowait()
                cv2.imshow("Image", image)
                cv2.waitKey(FRAME_TIME)
                continue
            except queue.Empty:
                pass

        time.sleep(1 / FRAME_TIME)

if __name__ == "__main__":
    track_video()
    cv2.destroyAllWindows()
