# =====================================================================================
# 🚀 [초간단] VS Code에서 이 프로그램 실행하는 방법 (학생/선생님 필독!)
# =====================================================================================
# 1단계: VS Code에서 이 파일이 있는 폴더를 엽니다.
# 2단계: 터미널을 켭니다. (단축키: Ctrl + `)
# 3단계: 라이브러리 설치 (터미널에 입력 후 Enter): pip install opencv-python mediapipe
# 4단계: 오른쪽 맨 위 [ ▶ (Run Python File) ] 버튼을 눌러 실행합니다.
#
# 💡 단축키 안내:
# - [ Q ] 키: 프로그램 안전하게 종료
# - [ C ] 키: 그린 그림 초기화 (Clear)
# - [ S ] 키: 그림을 이미지 파일로 저장 (Save)
# =====================================================================================
# ⚠️ [필독] 자주 발생하는 오류 및 해결 방법 (Troubleshooting)
# =====================================================================================
# Q1. ModuleNotFoundError: No module named 'cv2' 또는 'mediapipe'
# 👉 원인: 라이브러리가 현재 파이썬 버전에 설치되지 않았습니다.
# 👉 해결: VS Code 하단 파란색 상태바에서 Python 버전을 3.12 등 설치한 버전으로 
#          맞춰주거나, 터미널에 pip install 명령어를 다시 입력하세요.
#
# Q2. AttributeError: module 'mediapipe' has no attribute 'solutions'
# 👉 원인 1: 너무 최신 버전의 파이썬(예: 3.13, 3.14)을 사용 중일 수 있습니다.
# 👉 해결 1: 파이썬 버전을 3.12 이하로 변경하여 실행하세요.
# 👉 원인 2: 현재 폴더에 파일 이름을 'mediapipe.py'로 저장한 파일이 있습니다.
# 👉 해결 2: 파일 이름을 'my_pixel_art.py' 등으로 변경하세요.
#
# Q3. cv2.error: (-215:Assertion failed) ...
# 👉 원인: 카메라가 연결되지 않았거나 다른 프로그램(줌, 카메라 앱)이 사용 중입니다.
# 👉 해결: 다른 카메라 관련 프로그램을 모두 끄고 다시 실행해 보세요.
# =====================================================================================

# 1. 필수 도구 상자(라이브러리) 가져오기
import cv2            # 카메라 영상 처리 및 화면 출력을 담당하는 OpenCV 라이브러리
import mediapipe as mp # 구글에서 개발한 실시간 손가락 랜드마크 인식 인공지능 라이브러리

# MediaPipe의 핵심 기능 중 '손 인식(hands)' 모듈과 화면에 '뼈대 그리기(drawing_utils)' 도구를 변수에 저장합니다.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 2. 웹캠 카메라 연결하기 (0번은 내장 웹캠 또는 기본 연결된 카메라를 뜻합니다)
cap = cv2.VideoCapture(0)

# 💡 [다중 색상 지원] set(집합) 대신 dict(딕셔너리)를 사용하여 각 픽셀 위치마다 고유한 색상 정보를 쌍으로 기록합니다.
# 데이터 매핑 형식 -> {(격자_X_좌표, 격자_Y_좌표): (B, G, R 색상값)}
painted_cells = {} 
CELL_SIZE = 50 # 화면에 그려질 사각형 픽셀 격자 한 칸의 크기 (50픽셀 단위)

# 🎨 [초기 기본 상태 설정] 처음 시작할 때는 빨간색 붓으로 그리기 모드가 활성화됩니다.
current_color = (0, 0, 255)  # 빨간색 (OpenCV는 색상을 Blue, Green, Red 순서로 인식합니다)
current_mode = "draw"        # 현재 상태: "draw"(그리기) 또는 "erase"(지우개로 지우기)

# 3. 인공지능 손 인식 엔진 초기화 (정확도 매개변수 조절)
with mp_hands.Hands(
    max_num_hands=1,              # 화면 속에서 가장 먼저 발견되는 손 딱 1개만 정밀 추적합니다.
    min_detection_confidence=0.7, # 손을 처음 찾아낼 때의 확률 기준 (70% 이상 확신할 때만 인식 시작)
    min_tracking_confidence=0.7   # 움직이는 손을 계속 쫓아갈 때의 신뢰도 기준
) as hands:

    # 웹캠 카메라가 켜져서 올바르게 작동하는 동안 무한 루프를 돕니다.
    while cap.isOpened():
        success, image = cap.read() # 카메라 센서로부터 실시간 정지 영상 한 장(프레임)을 읽어옵니다.
        if not success:
            print("카메라 프레임을 정상적으로 읽어오지 못했습니다.")
            break

        # 사용자가 화면을 보며 손을 직관적으로 움직일 수 있도록 거울 모드(좌우 대칭 반전)를 적용합니다.
        image = cv2.flip(image, 1)
        
        # 4. 이미지 색상 필터 변환하기
        # OpenCV는 기본적으로 BGR(청-녹-적) 형태로 이미지를 가공하지만, MediaPipe 인공지능은 RGB 순서만 이해하므로 색 공간을 변환해 줍니다.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 변환된 이미지를 인공지능 신경망에 흘려보내 손의 마디 위치(results)를 알아냅니다.
        results = hands.process(image_rgb)
        
        # 현재 연결된 웹캠의 해상도 너비(w)와 높이(h) 정보를 픽셀 단위로 추출합니다.
        h, w, c = image.shape

        # 5. 가이드라인 격자 선 그리기 (스케치북 모눈종이 효과)
        for x in range(0, w, CELL_SIZE):
            cv2.line(image, (x, 0), (x, h), (220, 220, 220), 1) # 세로 격자 선
        for y in range(0, h, CELL_SIZE):
            cv2.line(image, (0, y), (w, y), (220, 220, 220), 1) # 가로 격자 선

        # 6. 저장되어 있는 픽셀 그림들을 화면에 채색하여 불러오기
        # 딕셔너리에 보관된 좌표 정보를 매 프레임 루프마다 읽어와 격자 규격에 맞춰 해당 색으로 사각형을 칠합니다.
        for (cell_x, cell_y), color in painted_cells.items():
            start_point = (cell_x * CELL_SIZE, cell_y * CELL_SIZE) # 사각형의 좌측 상단 꼭지점
            end_point = (start_point[0] + CELL_SIZE, start_point[1] + CELL_SIZE) # 우측 하단 꼭지점
            cv2.rectangle(image, start_point, end_point, color, -1) # 내부를 가득 채운(-1) 사각형 그리기

        # 7. 상단 메뉴판 인터페이스 그리기 (전체 해상도 가로 너비를 4등분하여 배치)
        button_w = w // 4 
        
        # 각 색상 및 지우개 영역을 구분하는 UI 사각형 박스 그리기
        cv2.rectangle(image, (0, 0), (button_w, 60), (0, 0, 255), -1)        # 빨간색 선택 영역
        cv2.rectangle(image, (button_w, 0), (button_w*2, 60), (0, 255, 0), -1)   # 초록색 선택 영역
        cv2.rectangle(image, (button_w*2, 0), (button_w*3, 60), (255, 0, 0), -1)  # 파란색 선택 영역
        cv2.rectangle(image, (button_w*3, 0), (w, 60), (200, 200, 200), -1)    # 지우개 선택 영역 (회색)

        # 메뉴 버튼 중앙에 가독성 높은 영어 라벨명 달기
        cv2.putText(image, "RED", (button_w // 2 - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, "GREEN", (button_w + button_w // 2 - 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, "BLUE", (button_w*2 + button_w // 2 - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, "ERASE", (button_w*3 + button_w // 2 - 25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 8. 인공지능이 영상 속에서 사람의 손을 정상적으로 감지했다면 실행
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 검지손가락과 중지손가락의 주요 관절 정보(끝점 TIP, 중간마디 PIP)를 가져옵니다.
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

                # 9. [제스처 감지 수학 원리] 각 손가락이 위로 꼿꼿이 서 있는지 여부를 판별합니다.
                # 화면 좌표계는 아래로 내려갈수록 Y값이 증가하므로, 끝점의 Y좌표가 마디의 Y좌표보다 작아야 펴진 상태입니다.
                index_up = index_tip.y < index_pip.y
                middle_up = middle_tip.y < middle_pip.y

                # 검지손가락 끝의 소수점 형태 상대 좌표(0.0 ~ 1.0)를 모니터 화면의 실제 정수형 픽셀 좌표(X, Y)로 환산합니다.
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                grid_x, grid_y = cx // CELL_SIZE, cy // CELL_SIZE # 현재 마우스 끝이 머무르는 격자의 칸 번호를 정수형 나눗셈으로 도출합니다.

                # 10. 세 가지 상태 제스처 시나리오 (상태 머신 제어)
                
                # 10-A. [선택 모드] 검지와 중지를 모두 활짝 편 상태 (V자 형태 제스처)
                if index_up and middle_up:
                    # 선택 모드일 때는 캔버스에 색칠을 중지하고 노란색 원(가이드 조준점)만 화면에 표기합니다.
                    cv2.circle(image, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

                    # 조준점의 중심 좌표가 상단 메뉴 영역(높이 60픽셀 이하)에 들어오면 해당 위치에 매칭되는 도구를 선택합니다.
                    if cy < 60:
                        if 0 <= cx < button_w:
                            current_color = (0, 0, 255) # 빨간색 선택
                            current_mode = "draw"
                        elif button_w <= cx < button_w*2:
                            current_color = (0, 255, 0) # 초록색 선택
                            current_mode = "draw"
                        elif button_w*2 <= cx < button_w*3:
                            current_color = (255, 0, 0) # 파란색 선택
                            current_mode = "draw"
                        elif button_w*3 <= cx < w:
                            current_mode = "erase"      # 지우개 선택

                # 10-B. [그리기/지우기 작동 모드] 오직 검지만 펴고 중지는 접은 상태 (검지 가리키기 제스처)
                elif index_up and not middle_up:
                    if current_mode == "draw":
                        # 검지 끝이 위치한 격자 번호에 현재 설정된 붓 색상 데이터를 딕셔너리에 저장합니다.
                        painted_cells[(grid_x, grid_y)] = current_color
                        # 검지 손가락 끝에 현재 내가 무슨 색을 칠하고 있는지 알 수 있도록 붓 브러시 색을 원으로 보여줍니다.
                        cv2.circle(image, (cx, cy), 15, current_color, cv2.FILLED)
                    elif current_mode == "erase":
                        # 지우개 상태인 경우, 해당 격자 번호에 기록되어 있던 색상 값을 사물함(딕셔너리)에서 안전하게 버립니다.
                        painted_cells.pop((grid_x, grid_y), None)
                        # 지우개 상태임을 직관적으로 보여주기 위해 검지 끝에 흰색 원 브러시를 표시합니다.
                        cv2.circle(image, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

                # 10-C. [대기 모드] 주먹을 꽉 쥐거나 손가락이 모두 잡힌 경우 (그리기 동작 중단)
                else:
                    # 화면 중앙 등 원하지 않는 위치에 강제로 점이 찍히거나 그림이 망가지는 일을 차단하기 위해 아무 로직도 실행하지 않고 그냥 넘어갑니다.
                    pass

                # 인공지능이 계산해낸 손가락 관절 구조와 라인(뼈대)을 화려하게 화면 상에 연동하여 그려줍니다.
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 모든 픽셀 아트와 손가락 그래픽 요소가 반영된 실시간 비디오를 새 팝업 창에 표시합니다.
        cv2.imshow('AI Hand Pixel Art', image)

        # 11. 키보드 단축키 실시간 입력 대기 및 이벤트 핸들링
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break # 'q' 키가 입력되면 전체 메인 루프를 탈출하여 프로그램을 안전하게 종료합니다.
        elif key == ord('c'): 
            painted_cells.clear() # 'c' 키를 누르면 딕셔너리 내부를 싹 비워 화면에 칠해진 그림들을 초기화합니다.
        elif key == ord('s'):
            cv2.imwrite('pixel_art.png', image) # 's' 키를 누르면 현재 캡처된 화면 프레임을 사진 파일로 저장합니다.

# 12. 프로그램 종료 시 메모리 누수를 방지하기 위해 카메라 장치를 연결 해제하고 생성된 임시 윈도우 창들을 전부 파괴합니다.
cap.release()
cv2.destroyAllWindows()