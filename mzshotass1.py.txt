import cv2
import mediapipe as mp
import time
from pathlib import Path
import winsound

# ---------------- 카메라 열기 ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라 열기 실패!")
    exit()

# ---------------- Mediapipe 초기화 ----------------
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# ---------------- Desktop 저장 경로 ----------------
desktop_path = Path.home() / "Desktop"
desktop_path.mkdir(exist_ok=True)

# ---------------- Victory 제스처 판단 ----------------
def is_victory(lms, w, h):
    """검지+중지 펴짐, 약지+새끼 접힘이면 V 사인 True"""
    def c(i):
        lm = lms.landmark[i]
        return int(lm.x*w), int(lm.y*h)

    i_tip = c(8)    # 검지 끝
    m_tip = c(12)   # 중지 끝
    r_tip = c(16)   # 약지 끝
    p_tip = c(20)   # 새끼 끝

    i_kn = c(5)     # 검지 knuckle
    m_kn = c(9)     # 중지 knuckle
    r_kn = c(13)    # 약지 knuckle
    p_kn = c(17)    # 새끼 knuckle

    # 검지/중지는 펴짐, 약지/새끼는 접힘
    return i_tip[1] < i_kn[1] and m_tip[1] < m_kn[1] and r_tip[1] > r_kn[1] and p_tip[1] > p_kn[1]

# ---------------- 메인 루프 ----------------
captured = False

while True:
    ret, img = cap.read()
    if not ret:
        continue

    img_h, img_w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 얼굴 인식
    face_res = face_detector.process(rgb)
    face_detected = face_res.detections is not None

    # 손 인식
    hand_res = hand_detector.process(rgb)
    victory_detected = False

    if hand_res.multi_hand_landmarks:
        for handLms in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            if is_victory(handLms, img_w, img_h):
                victory_detected = True
                cv2.putText(img, "VICTORY!", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                break  # 한 손이라도 V 사인면 True, 루프 종료

    # 얼굴 박스 그리기
    if face_detected:
        for d in face_res.detections:
            mp_draw.draw_detection(img, d)

    # 상태 표시
    cv2.putText(img, f"Face: {face_detected}, Victory: {victory_detected}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ---------------- 캡처 조건 ----------------
    if face_detected and victory_detected and not captured:
        filename = desktop_path / f"capture_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), img)
        print("✔ 캡처 저장 (Desktop):", filename)

        # 1초 소리 재생
        winsound.Beep(1000, 1000)

        captured = True
        print("촬영 완료! 손을 내리면 다시 촬영 가능")

    # 캡처 리셋: V 사인이 화면에서 사라지면 다시 촬영 가능
    if not victory_detected:
        captured = False

    # 영상창 출력
    cv2.imshow("Face + Victory Detector", img)

    # q 키 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
