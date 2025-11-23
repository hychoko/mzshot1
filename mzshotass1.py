import av
import cv2
import time
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def is_victory(lms, w, h):
    """ V 손가락 제스처 판단 """
    def c(i):
        lm = lms.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    i_tip, m_tip = c(8), c(12)
    r_tip, p_tip = c(16), c(20)

    i_kn, m_kn = c(5), c(9)
    r_kn, p_kn = c(13), c(17)

    return (
        i_tip[1] < i_kn[1] and
        m_tip[1] < m_kn[1] and
        r_tip[1] > r_kn[1] and
        p_tip[1] > p_kn[1]
    )


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
        self.captured = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 얼굴 인식
        face_res = self.face_detector.process(rgb)
        face_detected = face_res.detections is not None

        # 손 인식
        hand_res = self.hand_detector.process(rgb)
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                if is_victory(handLms, w, h):
                    victory_detected = True
                    cv2.putText(img, "VICTORY!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    break

        # 얼굴 박스
        if face_detected:
            for d in face_res.detections:
                mp_draw.draw_detection(img, d)

        # 캡처 조건
        if face_detected and victory_detected and not self.captured:
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            print("📸 이미지 저장:", filename)
            self.captured = True

        if not victory_detected:
            self.captured = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===========================
# Streamlit UI
# ===========================

st.title("🖐 스마트 셀카 웹앱 (얼굴 + Victory 제스처 감지)")

st.write("얼굴이 보이고 Victory(V) 손가락 제스처를 하면 자동 촬영됩니다.")

webrtc_streamer(
    key="selfie-app",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
