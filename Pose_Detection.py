import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def calculate_angle(point1, point2, point3):
    # Calculate the angle between three points (in radians)
    angle = math.atan2(point3.y - point2.y, point3.x - point2.x) - math.atan2(point1.y - point2.y, point1.x - point2.x)
    angle = abs(angle) * 180.0 / math.pi
    if angle > 180:
        angle = 360 - angle
    return angle

def detect_action(pose_landmarks):
    if not pose_landmarks:
        return 0

    # Extract key landmarks
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Calculate arm angles
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    ### 💥 **Attack Poses** ###
    # 1️⃣ Punching
    if left_arm_angle < 50 and left_wrist.x > left_elbow.x:
        return "Left Punch - Attacking"
    if right_arm_angle < 50 and right_wrist.x < right_elbow.x:
        return "Right Punch - Attacking"

    # 2️⃣ Pushing (both hands extended forward)
    if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y) and \
       (left_wrist.x > left_shoulder.x and right_wrist.x > right_shoulder.x):
        return "Pushing Forward - Attacking"

    # 3️⃣ Grabbing (both wrists close together in front)
    if abs(left_wrist.x - right_wrist.x) < 0.05 and abs(left_wrist.y - right_wrist.y) < 0.05:
        return "Grabbing - Possible Threat"

    # 4️⃣ Kicking (one knee raised above hips)
    if left_knee.y < left_hip.y or right_knee.y < right_hip.y:
        return "Kicking - Attacking"

    # 5️⃣ Aggressive Forward Lean (shoulders ahead of hips)
    if left_shoulder.x > left_hip.x and right_shoulder.x > right_hip.x:
        return "Aggressive Lean - Threatening"

    ### ⚠ **Threat Poses** ###
    # 6️⃣ Hands Raised (Possible surrender or threat)
    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        return "Raised Hands - Possible Threat"

    # 7️⃣ Chest Puffing (shoulders pulled back, chest forward)
    if (left_shoulder.x < left_hip.x and right_shoulder.x > right_hip.x) and \
       (abs(left_shoulder.y - right_shoulder.y) < 0.05):
        return "Chest Puffing - Threatening"

    # 8️⃣ Wide Stance (feet spread apart)
    if abs(left_ankle.x - right_ankle.x) > 0.25:
        return "Wide Stance - Intimidation"

    ### 🛡 **Defensive Poses** ###
    # 9️⃣ Crossed Arms
    if left_wrist.x < left_elbow.x and right_wrist.x > right_elbow.x:
        return "Crossed Arms - Defensive Posture"

    # 🔟 Blocking (one hand raised above head)
    if left_wrist.y < left_elbow.y and left_wrist.x < left_shoulder.x:
        return "Left Block - Defensive"
    if right_wrist.y < right_elbow.y and right_wrist.x > right_shoulder.x:
        return "Right Block - Defensive"

    # 1️⃣1️⃣ Dodging (upper body leaning sideways)
    if abs(left_shoulder.y - right_shoulder.y) > 0.1:
        return "Dodging - Defensive"

    ### ✅ Default Case: Standing ###
    return 0
