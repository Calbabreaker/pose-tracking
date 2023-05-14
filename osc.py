from pythonosc import udp_client
import math

# See https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
pose_labels = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

class OSCSender:
    def __init__(self, ip, port, height):
        self.ip = ip
        self.port = port
        self.height = height
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_messages(self, landmarks):
        head_pos = as_tuple(landmarks, "nose")
        hips_pos = midpoint(landmarks, ["left_hip", "right_hip"])
        chest_pos = midpoint(landmarks, ["left_hip", "right_hip", "left_shoulder", "right_shoulder"])
        left_foot_pos = as_tuple(landmarks, "left_ankle")
        right_foot_pos = as_tuple(landmarks, "right_ankle")
        left_knee_pos = as_tuple(landmarks, "left_knee")
        right_knee_pos = as_tuple(landmarks, "right_knee")
        left_elbow_pos = as_tuple(landmarks, "left_elbow")
        right_elbow_pos = as_tuple(landmarks, "right_elbow")

        self.send_pos("head", head_pos)
        self.send_pos("1", hips_pos)
        self.send_pos("2", chest_pos)
        self.send_pos("3", left_foot_pos)
        self.send_pos("4", right_foot_pos)
        # self.send_pos("5", left_knee_pos)
        # self.send_pos("6", right_knee_pos)
        # self.send_pos("7", left_elbow_pos)
        # self.send_pos("8", right_elbow_pos)

#         hips_rot = angle(landmarks, "left_hip", "right_hip")
#         self.client.send_message(tracker_addresses["hips"] + "/rotation", (0, 0, hips_rot))

    def send_pos(self, label, raw_pos):
        pos = (raw_pos[0] * self.height, raw_pos[1] * -1 * self.height, raw_pos[2] * self.height)
        self.client.send_message("/tracking/trackers/" + label + "/position", pos)


def as_tuple(landmarks, label):
    mark = landmarks[pose_labels[label]]
    return (mark.x, mark.y, mark.z)

def angle(landmarks, label1, label2):
    mark1 = landmarks[pose_labels[label1]]
    mark2 = landmarks[pose_labels[label2]]
    return -math.atan2(mark1.x - mark2.z, mark1.y - mark2.z) * 180 / math.pi

def midpoint(landmarks, labels):
    return (sum(landmarks[pose_labels[label]].x for label in labels) / len(labels), 
            sum(landmarks[pose_labels[label]].y for label in labels) / len(labels), 
            sum(landmarks[pose_labels[label]].z for label in labels) / len(labels))
