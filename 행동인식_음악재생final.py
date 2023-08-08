# 모듈 import
import cv2
import pygame
from pathlib import Path
import time

#  MPII 데이터셋에서 각 신체 부위의 번호와 이름을 매핑한 딕셔너리
BODY_PARTS = {
    "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
    "Background": 15
}

# 연결될 신체 부위 쌍을 정의한 리스트
POSE_PAIRS = [
    ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]
]

# 신경망 모델 파일 경로
# BASE_DIR: 현재 스크립트 파일 경로 | protoFile: 신경망 모델 구조 경로 | weightsFile: 학습된 신경망 모델의 가중치 파일 경로
BASE_DIR = Path(__file__).resolve().parent
protoFile = str(BASE_DIR) + "/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = str(BASE_DIR) + "/pose_iter_160000.caffemodel"

# 위의 path에 있는 network 모델 불러오기
# "cv2.dnn.readNetFromCaffe()": Caffe 프레임워크의 모델 구조 파일과 해당 모델의 학습된 가중치 파일을 입력으로 받아 신경망 모델을 불러옴
# "readNetFromCaffe()": 불러온 모델을 나타내는 net 객체를 반환, net를 통해 영상에서 입력 데이터를 전달해 신경망을 실행하고 결과를 도출함
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 웹캠과 연결
capture = cv2.VideoCapture(0)

# 입력 이미지의 크기와 스케일 설정
inputWidth = 320
inputHeight = 240
inputScale = 1.0 / 255

# 배경음악 파일 경로
blanksound_path = str(BASE_DIR) + "/blanksound.mp3"
happy_music_path = str(BASE_DIR) + "/happy_bgm.mp3"
sad_music_path = str(BASE_DIR) + "/sad_bgm.mp3"

# 배경음악 초기화
pygame.mixer.init()
pygame.mixer.music.stop()
# 팔과 다리 동작 변수 초기화
happy_pose = False
crying_pose = False
# 시간 측정을 위한 변수 초기화
start_time = time.time()
elapsed_time = 0

###################################################################

# 반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
while cv2.waitKey(1) < 0:  # 아무 키나 누르면 끝남
    # 웹캠으로부터 영상 가져옴
    hasFrame, frame = capture.read()

    # 웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
    if not hasFrame:
        cv2.waitKey()
        break

    # 프레임의 너비, 높이 설정
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # 이미지를 전처리해 네트워크에 입력
    inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    # 네트워크를 통해 예측 수행
    net.setInput(inpBlob)
    # 네트워크의 출력 반환
    output = net.forward()

    # 각 키포인트를 검출해 이미지에 그림
    points = []
    for i in range(0, 15): # 신체부위는 0~15번까지
        # 해당 신체부위의 신뢰도를 얻음.
        probMap = output[0, i, :, :]

        # 신뢰도 맵에서 가장 큰 값과 그 위치를 찾아 가장 확률이 높은 위치를 키포인트로 간주함
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 키포인트 위치 변경
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]

        # 키포인트 검출한 결과가 0.1보다 크면(검출한 위치가 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로 설정
        if prob > 0.1:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                       lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # points 리스트에서 사용할 신체 부위의 좌표를 추출해 변수에 할당함
    neck = points[BODY_PARTS["Neck"]]
    chest = points[BODY_PARTS["Chest"]]
    right_shoulder = points[BODY_PARTS["RShoulder"]]
    left_shoulder = points[BODY_PARTS["LShoulder"]]
    right_elbow = points[BODY_PARTS["RElbow"]]
    left_elbow = points[BODY_PARTS["LElbow"]]
    right_wrist = points[BODY_PARTS["RWrist"]]
    left_wrist = points[BODY_PARTS["LWrist"]]
    right_hip = points[BODY_PARTS["RHip"]]
    left_hip = points[BODY_PARTS["LHip"]]
    right_knee = points[BODY_PARTS["RKnee"]]
    left_knee = points[BODY_PARTS["LKnee"]]

    # 쪼그려 앉아 울기 동작 인식
    if right_shoulder and right_wrist and left_shoulder and left_wrist and chest and right_knee:
        shoulder_to_wrist = abs(right_shoulder[0] - right_wrist[0]) + abs(left_shoulder[0] - left_wrist[0])
        chest_to_knee = abs(chest[1] - right_knee[1])
        if shoulder_to_wrist < 100 and chest_to_knee < 100:
            crying_pose = True
        else:
            crying_pose = False

    # 팔 벌리기 동작 인식
    if right_elbow and left_elbow and right_hip and left_hip and left_knee and neck:
        elbow_to_hip = abs(right_elbow[1] - right_hip[1]) + abs(left_elbow[1] - left_hip[1])
        neck_to_knee = abs(neck[1] - left_knee[1])
        if elbow_to_hip > 200 and neck_to_knee > 50:
            happy_pose = True
        else:
            happy_pose = False

    # 10초마다 elbow_to_ankle 프롬프트에 출력
    # happy_bgm의 재생이 끊겨, 동작이 위의 조건에 부합하는지 확인하기 위해 추가함
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10:
        print("elbow_to_hip:", elbow_to_hip)
        start_time = time.time()



    # 쪼그려 앉아 울기 동작일 때 슬픈 배경음악 재생
    if crying_pose:
        if pygame.mixer.music.get_busy() == 0 or pygame.mixer.music.get_pos() == -1:
            pygame.mixer.music.load(sad_music_path)
            pygame.mixer.music.play(-1)  # 반복 재생
    else:
        pygame.mixer.music.stop()

    # 팔 벌리기 동작일 때 행복한 배경음악 재생
    if happy_pose:
        if pygame.mixer.music.get_busy() == 0 or pygame.mixer.music.get_pos() == -1:
            pygame.mixer.music.load(happy_music_path)
            pygame.mixer.music.play(-1)  # 반복 재생
    else:
        pygame.mixer.music.stop()

    # POSE_PAIRS 리스트에 정의된 신체 부위 쌍에 선을 그림
    for pair in POSE_PAIRS:
        partA = pair[0]  # 머리
        partA = BODY_PARTS[partA]
        partB = pair[1]  # 목
        partB = BODY_PARTS[partB]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Output-Keypoints", frame)

capture.release()  # 카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows()  # 모든 윈도우 창 닫음
# pygame.mixer.music.stop()  # 배경음악 재생 중지










