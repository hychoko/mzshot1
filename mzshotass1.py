import cv2
import mediapipe as mp
import time
from pathlib import Path
import platform

# 윈도우 환경에서만 winsound 임포트 (Mac/Linux 호환성 유지)
try:
    import winsound
except ImportError:
    winsound = None

class GestureCamera:
    def __init__(self, output_dir="Desktop"):
        """
        초기화 함수: 미디어파이프 모델 로드 및 저장 경로 설정
        """
        # Mediapipe 초기화
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

        # 저장 경로 설정 (기본값: 바탕화면)
        self.save_path = Path.home() / output_dir
        self.save_path.mkdir(exist_ok=True)
        
        # 상태 변수
        self.is_captured = False

    def play_sound(self):
        """촬영음 재생 (OS에 따라 다르게 처리)"""
        if winsound:
            # 주파수 1000Hz, 지속시간 500ms (1초는 너무 길 수 있어 줄임)
            winsound.Beep(1000, 500)
        else:
            # Mac/Linux에서는 시스템 벨소리 출력 (터미널 설정에 따라 안 들릴 수 있음)
            print('\a')

    def is_victory(self, lms, w, h):
        """
        제스처 판단 로직: 검지+중지 펴짐, 약지+새끼 접힘 여부 확인
        """
        def c(i):
            lm = lms.landmark[i]
            return int(lm.x * w), int(lm.y * h)

        # 손가락 끝 좌표
        i_tip, m_tip = c(8), c(12)  # 검지, 중지
        r_tip, p_tip = c(16), c(20) # 약지, 새끼

        # 손가락 마디(Knuckle) 좌표
        i_kn, m_kn = c(5), c(9)
        r_kn, p_kn = c(13), c(17)

        # Y축 비교: 화면 상단이 0이므로 숫자가 작을수록 위쪽
        # 검지/중지는 펴짐(Tip < Knuckle), 약지/새끼는 접힘(Tip > Knuckle)
        return (i_tip[1] < i_kn[1] and 
                m_tip[1] < m_kn[1] and 
                r_tip[1] > r_kn[1] and 
                p_tip[1] > p_kn[1])

    def run(self):
        """메인 실행 루프"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: 카메라를 열 수 없습니다.")
            return

        print(f"✔ 저장 경로: {self.save_path}")
        print("✔ 'q'를 누르면 종료됩니다.")

        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    continue

                img_h, img_w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 1. 얼굴 인식
                face_res = self.face_detector.process(rgb)
                face_detected = face_res.detections is not None

                # 2. 손 인식 및 제스처 확인
                hand_res = self.hand_detector.process(rgb)
                victory_detected = False

                if hand_res.multi_hand_landmarks:
                    for handLms in hand_res.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
                        
                        if self.is_victory(handLms, img_w, img_h):
                            victory_detected = True
                            cv2.putText(img, "VICTORY!", (50, 300),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            # 한 손만 인식돼도 촬영 조건 충족
                            break 

                # 얼굴 박스 그리기
                if face_detected:
                    for detection in face_res.detections:
                        self.mp_draw.draw_detection(img, detection)

                # 상태 텍스트 표시
                status_color = (0, 255, 255) if not self.is_captured else (0, 0, 255)
                status_text = f"Face: {face_detected}, Victory: {victory_detected}"
                cv2.putText(img, status_text, (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # 3. 캡처 로직
                # 얼굴과 V사인이 모두 있고, 아직 캡처하지 않은 상태일 때
                if face_detected and victory_detected and not self.is_captured:
                    filename = self.save_path / f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(str(filename), img)
                    print(f"📸 캡처 완료: {filename}")
                    
                    self.play_sound() # 소리 재생
                    self.is_captured = True

                # 리셋 로직: V 사인을 풀면 다시 촬영 가능 상태로 변경
                if not victory_detected:
                    self.is_captured = False

                cv2.imshow("Smart Selfie Camera", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureCamera()
    app.run()
