#!/usr/bin/env python
# -*- coding: utf-8 -*- 1
import numpy as np
import cv2, rospy, time, math, os
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ar_track_alvar_msgs.msg import AlvarMarkers
import importlib.util
import json
from datetime import datetime
import signal, sys

import camera_processing
import slide_window

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
drive_mode = 0
motor = None  # 모터 노드 변수
Fix_Speed = 5 # 모터 속도 고정 상수값 
new_angle = 0  # 모터 조향각 초기값
new_speed = Fix_Speed  # 모터 속도 초기값
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 
lidar_points = None  # 라이다 데이터를 담을 변수
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
motor_msg = xycar_motor()  # 모터 토픽 메시지
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
Blue =  (255,0,0) # 파란색
Green = (0,255,0) # 녹색
Red =   (0,0,255) # 빨간색
Yellow = (0,255,255) # 노란색
stopline_num = 1 # 정지선 발견때마다 1씩 증가
View_Center = WIDTH//2  # 화면의 중앙값 = 카메라 위치
ar_msg = {"ID":[],"DX":[],"DZ":[]}  # AR태그 토픽을 담을 변수

#===================라바콘 변수 ==================
safe_distance = 0
angle_factor = 15
error = 0
k = 30.0
kp = 7
#=============================================
# 학습결과 파일의 위치 지정
#=============================================
PATH_TO_CKPT = '/home/pi/xycar_ws/src/study/track_drive/src/detect.tflite'
PATH_TO_LABELS = '/home/pi/xycar_ws/src/study/track_drive/src/labelmap.txt'

#=============================================
# 차선인식 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30  # 카메라 FPS 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 300  # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 380  # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW  # ROI 영역의 세로 크기  
L_ROW = 40  # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 

#=============================================
# 프로그램에서 사용할 이동평균필터 클래스
#=============================================
class MovingAverage:

    # 클래스 생성과 초기화 함수 (데이터의 개수를 지정)
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n + 1))

    # 새로운 샘플 데이터를 추가하는 함수
    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data.pop(0)  # 가장 오래된 샘플 제거
            self.data.append(new_sample)

    # 저장된 샘플 데이터의 갯수를 구하는 함수
    def get_sample_count(self):
        return len(self.data)

    # 이동평균값을 구하는 함수
    def get_mavg(self):
        if not self.data:
            return 0.0
        return float(sum(self.data)) / len(self.data)

    # 중앙값을 사용해서 이동평균값을 구하는 함수
    def get_mmed(self):
        if not self.data:
            return 0.0
        return float(np.median(self.data))

    # 가중치를 적용하여 이동평균값을 구하는 함수        
    def get_wmavg(self):
        if not self.data:
            return 0.0
        s = sum(x * w for x, w in zip(self.data, self.weights[:len(self.data)]))
        return float(s) / sum(self.weights[:len(self.data)])

    # 샘플 데이터 중에서 제일 작은 값을 반환하는 함수
    def get_min(self):
        if not self.data:
            return 0.0
        return float(min(self.data))
    
    # 샘플 데이터 중에서 제일 큰 값을 반환하는 함수
    def get_max(self):
        if not self.data:
            return 0.0
        return float(max(self.data))
        
#=============================================
# 프로그램에서 사용할 PID 클래스
#=============================================  
class PID:

    def __init__(self, kp, ki, kd):
        # PID 게인 초기화
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        # 이전 오차 초기화
        self.cte_prev = 0.0
        # 각 오차 초기화
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0
        # 적분오차 제한값 설정
        self.i_min = -10
        self.i_max = 10

    def pid_control(self, cte):
        # 미분오차 계산
        self.d_error = cte - self.cte_prev
        # 비례오차 계산
        self.p_error = cte
        # 적분오차 계산 및 제한 적용
        self.i_error += cte
        self.i_error = max(min(self.i_error, self.i_max), self.i_min)
        # 이전 오차 업데이트
        self.cte_prev = cte

        # PID 제어 출력 계산
        return self.Kp * self.p_error + self.Ki * self.i_error + self.Kd * self.d_error

#=============================================
# 초음파 거리정보에 대해서 이동평균필터를 적용하기 위한 선언
#=============================================
avg_count = 10  # 이동평균값을 계산할 데이터 갯수 지정
ultra_data = [MovingAverage(avg_count) for i in range(8)]

#=============================================
# 조향각에 대해서 이동평균필터를 적용하기 위한 선언
#=============================================
angle_avg_count = 10  # 이동평균값을 계산할 데이터 갯수 지정
angle_avg = MovingAverage(angle_avg_count)

#=============================================
# 특정 게인으로 PID 제어기 인스턴스를 생성
#=============================================
#초기 1.0 / 0.1 / 0.01
#속도 4000 20 k0.5 i 0.3 d 1.5
ar_pid = PID(kp=0.5, ki=0.1, kd=0.1)
#0.5 0.07 1.0
#0.35 0.07 0.7
pid = PID(kp=0.3, ki=0.0001, kd=0.5)


#=============================================
# ctrl+c 누르면 로그파일 저장
#=============================================
def signal_handler(sig, frame):
    global logger
    print('Ctrl+C was pressed. Exiting gracefully...')
    logger.save_logs_to_file()
    # 여기에 종료 전에 실행할 코드를 추가하세요.
    sys.exit(0)

# SIGINT 시그널이 발생했을 때 signal_handler 함수를 호출하도록 설정
signal.signal(signal.SIGINT, signal_handler)


#=============================================
# 콜백함수 - USB 카메라 토픽을 받아서 처리하는 콜백함수
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")


def compressed_image_callback(data):
    global compressed_image
    np_arr = np.frombuffer(data.data, np.uint8)
    compressed_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#=============================================
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
#=============================================

def lidar_callback(data):
    global lidar_points
    lidar_points = data.ranges


#=============================================
# 콜백함수 - 초음파 토픽을 받아서 처리하는 콜백함수
#=============================================
def ultra_callback(data):
    global ultra_msg
    ultra_msg = data.data

    # 초음파센서로부터 받은 데이터를 필터링 처리함.
    #ultra_filtering()

#=============================================
# 초음파 거리정보에 이동평균값을 적용하는 필터링 함수
#=============================================
def ultra_filtering():
    global ultra_msg

    # 이동평균필터를 적용해서 튀는 값을 제거하는 필터링 작업 수행
    for i in range(8):
        ultra_data[i].add_sample(float(ultra_msg[i]))
        
    # 여기서는 중앙값(Median)을 이용 - 평균값 또는 가중평균값을 이용하는 것도 가능 
    ultra_list = [int(ultra_data[i].get_mmed()) for i in range(8)]
    
    # 평균값(Average)을 이용 
    #ultra_list = [int(ultra_data[i].get_mavg()) for i in range(8)]
    
    # 가중평균값(Weighted Average)을 이용 
    #ultra_list = [int(ultra_data[i].get_wmavg()) for i in range(8)]
        
    # 최소값(Min Value)을 이용 
    #ultra_list = [int(ultra_data[i].get_min()) for i in range(8)]
    
    # 최대값(Max Value)을 이용 
    #ultra_list = [int(ultra_data[i].get_max()) for i in range(8)]
    
    ultra_msg = tuple(ultra_list)

#=============================================
# 콜백함수 - AR태그 토픽을 받아서 처리하는 콜백함수
#=============================================
def ar_callback(data):
    global ar_msg

    # AR태그의 ID값, X 위치값, Z 위치값을 담을 빈 리스트 준비
    ar_msg["ID"] = []
    ar_msg["DX"] = []
    ar_msg["DZ"] = []

    # 발견된 모두 AR태그에 대해서 정보 수집하여 ar_msg 리스트에 담음
    for i in data.markers:
        ar_msg["ID"].append(i.id) # AR태그의 ID값을 리스트에 추가
        ar_msg["DX"].append(int(i.pose.pose.position.x*100)) # X값을 cm로 바꿔서 리스트에 추가
        ar_msg["DZ"].append(int(i.pose.pose.position.z*100)) # Z값을 cm로 바꿔서 리스트에 추가
    
#=============================================
# 모터 토픽을 발행하는 함수 
#=============================================
def drive(angle, speed):
    global CURRENT_ANGLE, logger
    num = 100000
    #500000
    #10000
    CURRENT_ANGLE = (CURRENT_ANGLE + math.radians(angle/num)) % (2 * math.pi)
    logger.write_log(speed, CURRENT_ANGLE)

    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)
    
#=============================================
# 차량을 정차시키는 함수  
# 입력으로 지속시간을 받아 그 시간동안 속도=0 토픽을 모터로 보냄.
# 지속시간은 0.1초 단위임. 만약 15이면 1.5초가 됨.
#=============================================
def stop_car(duration):
    for i in range(int(duration)): 
        drive(angle=0, speed=0)
        time.sleep(0.1)
    
#=============================================
# 차량을 이동시키는 함수 
# 입력으로 조향각과 속도, 지속시간을 받아 차량을 이동시킴.
# 지속시간은 0.1초 단위임. 만약 15이면 1.5초가 됨. 
#=============================================
def move_car(move_angle, move_speed, duration):
    for i in range(int(duration)): 
        drive(move_angle, move_speed)
        time.sleep(0.1)
#=============================================
# 카메라의 Exposure 값을 변경하는 함수 
# 입력으로 0~255 값을 받는다.
#=============================================
def cam_exposure(value):
    command = 'v4l2-ctl -d /dev/videoCAM -c exposure_absolute=' + str(value)
    os.system(command)
    
#=============================================
# 특정 ROS 노드를 중단시키고 삭제하는 함수 
# 더 이상 사용할 필요가 없는 ROS 노드를 삭제할 때 사용한다.
#=============================================
def kill_node(node_name):
    try:
        # rosnode kill 명령어를 사용하여 노드를 종료
        result = os.system(f"rosnode kill {node_name}")
        if result == 0:
            rospy.loginfo(f"Node {node_name} has been killed successfully.")
        else:
            rospy.logwarn(f"Failed to kill node {node_name}. It may not exist.")
    except Exception as e:
        rospy.logerr(f"Failed to kill node {node_name}: {e}")

#=============================================
# AR 패지키지가 발행하는 토픽을 받아서 
# 제일 가까이 있는 AR Tag에 적힌 ID 값을 반환하는 함수
# 거리값과 좌우치우침값을 함께 반환
#=============================================
def check_AR():

    ar_data = ar_msg
    id_value = 99

    if (len(ar_msg["ID"]) == 0):
        # 발견된 AR태그가 없으면 
        # ID값 99, Z위치값 500cm, X위치값 500cm로 리턴
        return 99, 500, 500  

    # 새로 도착한 AR태그에 대해서 아래 작업 수행
    z_pos = 500  # Z위치값을 500cm로 초기화
    x_pos = 500  # X위치값을 500cm로 초기화
    
    for i in range(len(ar_msg["ID"])):
        # 발견된 AR태그 모두에 대해서 조사

        if(ar_msg["DZ"][i] < z_pos):
            # 더 가까운 거리에 AR태그가 있으면 그걸 사용
            id_value = ar_msg["ID"][i]
            z_pos = ar_msg["DZ"][i]
            x_pos = ar_msg["DX"][i]

    # ID번호, 거리값(미터), 좌우치우침값(미터) 리턴
    return id_value, round(z_pos,2), round(x_pos,2)
    
#=============================================
# 신호등의 파란불을 체크해서 True/False 값을 반환하는 함수
#=============================================
def check_traffic_sign():
    MIN_RADIUS, MAX_RADIUS = 15, 25
    
    # 원본이미지를 복제한 후에 특정영역(ROI Area)을 잘라내기
    cimg = image.copy()
    Center_X, Center_Y = 340, 100  # ROI 영역의 중심위치 좌표 
    XX, YY = 300, 80  # 위 중심 좌표에서 좌우로 XX, 상하로 YY만큼씩 벌려서 ROI 영역을 잘라냄   

    # ROI 영역에 녹색 사각형으로 테두리를 쳐서 표시함 
    cv2.rectangle(cimg, (Center_X-XX, Center_Y-YY), (Center_X+XX, Center_Y+YY) , Green, 2)
	
    # 원본 이미지에서 ROI 영역만큼 잘라서 roi_img에 담음 
    roi_img = cimg[Center_Y-YY:Center_Y+YY, Center_X-XX:Center_X+XX]

    # roi_img 칼라 이미지를 회색 이미지로 바꾸고 노이즈 제거를 위해 블러링 처리를 함  
    img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Hough Circle 함수를 이용해서 이미지에서 원을 (여러개) 찾음 
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                  param1=40, param2=20, 
                  minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    # 디버깅을 위해서 Canny 처리를 했을때의 모습을 화면에 표시함
    # 위 HoughCircles에서 param1, param2에 사용했던 값을 아래 canny에서 똑같이 적용해야 함. 순서 조심.
    canny = cv2.Canny(blur, 20, 40)
    cv2.imshow('Canny image used by HoughCircles', canny)
    cv2.waitKey(1)

    if circles is not None:
    
        # 정수값으로 바꾸고 발견된 원의 개수를 출력
        circles = np.round(circles[0, :]).astype("int")
        print("\nFound",len(circles),"circles")
        
        # 중심의 Y좌표값 순서대로 소팅해서 따로 저장
        y_circles = sorted(circles, key=lambda circle: circle[1])
 
        # 중심의 X좌표값 순서대로 소팅해서 circles에 다시 저장
        circles = sorted(circles, key=lambda circle: circle[0])
         
        # 발견된 원들에 대해서 루프를 돌면서 하나씩 녹색으로 그리기 
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(cimg, (x+Center_X-XX, y+Center_Y-YY), r, Green, 2)
 
    # 이미지에서 정확하게 3개의 원이 발견됐다면 신호등 찾는 작업을 진행  
    if (circles is not None) and (len(circles)==3):
            
        # 가장 밝은 원을 찾을 때 사용할 변수 선언
        max_mean_value = 0
        max_mean_value_circle = None
        max_mean_value_index = None

        # 발견된 원들에 대해서 루프를 돌면서 하나씩 처리 
 	    # 원의 중심좌표, 반지름. 내부밝기 정보를 구해서 화면에 출력
        for i, (x, y, r) in enumerate(circles):
            roi = img[y-(r//2):y+(r//2),x-(r//2):x+(r//2)]
            # 밝기 값은 반올림해서 10의 자리수로 만들어 사용
            mean_value = round(np.mean(roi),-1)
            print(f"Circle {i} at ({x},{y}), radius={r}: brightness={mean_value}")
			
            # 이번 원의 밝기가 기존 max원보다 밝으면 이번 원을 max원으로 지정  
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                max_mean_value_circle = (x, y, r)
                max_mean_value_index = i
                
            # 원의 밝기를 계산했던 사각형 영역을 빨간색으로 그리기 
            cv2.rectangle(cimg, ((x-(r//2))+Center_X-XX, (y-(r//2))+Center_Y-YY),
                ((x+(r//2))+Center_X-XX, (y+(r//2))+Center_Y-YY), Red, 2)

        # 가장 밝은 원을 찾았으면 그 원의 정보를 출력하고 노란색으로 그리기 
        if max_mean_value_circle is not None:
            (x, y, r) = max_mean_value_circle
            print(f" --- Circle {max_mean_value_index} is the brightest.")
            cv2.circle(cimg, (x+Center_X-XX, y+Center_Y-YY), r, Yellow, 2)
            
        # 신호등 찾기 결과가 표시된 이미지를 화면에 출력
        cv2.imshow('Circles Detected', cimg)
        
        # 제일 위와 제일 아래에 있는 2개 원의 Y좌표값 차이가 크면 안됨 
        vertical_diff = MAX_RADIUS * 2
        if (y_circles[-1][1] - y_circles[0][1]) > vertical_diff:
            print("Circles are scattered vertically!")
            return False
        
        # 제일 왼쪽과 제일 오른쪽에 있는 2개 원의 X좌표값 차이가 크면 안됨 
        horizontal_diff = MAX_RADIUS * 8
        if (circles[-1][0] - circles[0][0]) > horizontal_diff:
            print("Circles are scattered horizontally!")
            return False      
            
        # 원들이 좌우로 너무 붙어 있으면 안됨 
        min_distance = MIN_RADIUS * 3
        for i in range(len(circles) - 1):
            if (circles[i+1][0] - circles[i][0]) < min_distance:
                print("Circles are too close horizontally!")
                return False 
            
        # 3개 중에서 세번째 원이 가장 밝으면 (파란색 신호등) True 리턴 
        if (max_mean_value_index == 2):
            print("Traffic Sign is Blue...!")
            return True
        
        # 첫번째나 두번째 원이 가장 밝으면 (파란색 신호등이 아니면) False 반환 
        else:
            print("Traffic Sign is NOT Blue...!")
            return False

    # 신호등 찾기 결과가 표시된 이미지를 화면에 출력
    cv2.imshow('Circles Detected', cimg)
    
    # 원본 이미지에서 원이 발견되지 않았다면 False 리턴   
    #print("Can't find Traffic Sign...!")
    return False



# AR 태그 데이터 필터링 함수
def filter_ar_data(ar_data, prev_data):
    closest_data = {"ID": None, "DX": None, "DZ": None}
    min_distance = float('inf')

    for i in range(len(ar_data["ID"])):
        ar_id = ar_data["ID"][i]
        dx = ar_data["DX"][i]
        dz = ar_data["DZ"][i]

        # 이전 dx와 dz 값 가져오기
        prev_dx, prev_dz = prev_data.get(ar_id, (None, None))

        # Update dx only if the change is less than 50
        if prev_dx is not None and abs(dx - prev_dx) >= 50:
            dx = prev_dx

        # Calculate the distance
        distance = math.sqrt(dx**2 + dz**2)

        # Update the closest data if the current one is closer
        if distance < min_distance and dz > 5:
            closest_data["ID"] = ar_id
            closest_data["DX"] = dx
            closest_data["DZ"] = dz
            min_distance = distance

    # Update previous data dictionary
    if closest_data["ID"] is not None:
        prev_data[closest_data["ID"]] = (closest_data["DX"], closest_data["DZ"])
    elif len(ar_data["ID"]) > 0:  # If no valid data was found but there were AR tags detected
        last_id = ar_data["ID"][-1]
        closest_data = {
            "ID": last_id,
            "DX": prev_data.get(last_id, (None, None))[0],
            "DZ": prev_data.get(last_id, (None, None))[1]
        }

    return closest_data, prev_data

# 차량 제어 함수
def ar_drive(filtered_data):
    K = 300 * (-1) #커질수록 더 급격하게 회전
    if filtered_data["ID"] is None:
        drive(0, Fix_Speed)
        return
    dx = filtered_data["DX"]
    dz = filtered_data["DZ"]

    target_x = 50 if dx < 0 else -50
    target_z = 10
    
    distance = math.sqrt((dx-target_x)**2 + (dz-target_z)**2)
    
    if distance > 125:
        error = (dx - 27)
    else:
        error = ((dx - target_x) / distance) * K
    print("error : ", error)
    print("distance : ", distance)
    print("x_pos : ", dx)
    
    
    control = ar_pid.pid_control(error)
    control = max(min(int(control), 30), -30)
    print(control)
    drive(int(control), Fix_Speed)
    
def traffic_ar_drive(filtered_data, prev_control):
    K = 500 *(-1) #커질수록 더 급격하게 회전
    if filtered_data["ID"] is None:
        drive(-10, Fix_Speed)
        #print("none")
        return prev_control
    dx = filtered_data["DX"]
    dz = filtered_data["DZ"]

    target_x = 50 if dx < 0 else -50
    target_z = 10
    
    distance = math.sqrt((dx-target_x)**2 + (dz-target_z)**2)
    
    if distance > 150:
        error = dx
    else:
        error = ((dx - target_x - 27) / distance) * K
    print("error : ", error)
    print("distance : ", distance)
    print("x_pos : ", dx)
    
    
    control = ar_pid.pid_control(error)
    control = max(min(int(control), 30), -30)
    print(control)
    drive(int(control), Fix_Speed)
    return prev_control
    
    
    
    
#=============================================
# 라이다 센서를 이용해서 벽까지의 거리를 알아내서
# 벽과 충돌하지 않으며 주행하도록 모터로 토픽을 보내는 함수
#=============================================

def vaild_mean(points):
    points = np.array(points)
    vaild_points = points[np.isfinite(points)]

    if len(vaild_points) > 0:
        return np.min(vaild_points)
        
    else:
        return np.inf

def sensor_drive():
    global new_angle, new_speed
    global safe_distance, angle_factor,error, k
    Fix_Speed = 5

    # 장애물 판단 범위
    left_points = lidar_points[130:180]
    right_points = lidar_points[540:590]

    # 평균값으로 변환
    left_distance = vaild_mean(left_points)
    right_distance = vaild_mean(right_points)
            
    
    #print("left_distance: ", left_distance)
    #print("right_distance: ", right_distance)
    
    # error값이 초기화가 안돼서 조향이 잘 안됐던 거 같음
    # 오른쪽으로 가야 하는데? 이미 누적된 error값이 -21452라면?
    # 다시 +값으로 되려면 21452만큼 더해져야함 (저번에 반응 느렸던 이유)
    error = 0
    
    
    # 평균 값이 inf가 아니고, 거리가 2.0 이하일 때
    if left_distance != np.inf and right_distance != np.inf:    
        if left_distance < 0.2 or right_distance == 0.0:
            error = 30
            #print("왼쪽 가까움 or 오른쪽 0")
        elif right_distance < 0.2 or left_distance == 0.0:
            error = -30
            #print("오른쪽 가까움 or 왼쪽 0")
            
        # 왼쪽으로 가야할 때
        elif left_distance - safe_distance > right_distance:
            error -= 3.0  - right_distance
            #print("1")
            if error > 0:
                error = error*(-1)
                #print("2")
        # 오른쪽으로 가야할 때
        elif right_distance - safe_distance > left_distance:
            error += 3.0 - left_distance
            #print("3")
            if error < 0:
                error = error*(-1)
                #print("4")
        else:
            error = 0   
            #print("5")   
    else: # left_distance 또는 right_distance가 inf인 경우
        if left_distance == np.inf:
            error -= 3.0 - right_distance  # 왼쪽으로 회전
            #print("6")
        elif right_distance == np.inf:
            error += 3.0  - left_distance   # 오른쪽으로 회전
            #print("7")
        
    
    new_angle = int(error * k)
    #new_angle = max(min(new_angle, 80), -80)
    
    #print("angle: ", new_angle)

    new_speed = Fix_Speed
    drive(new_angle, new_speed)




class HSVTracker:
    def __init__(self):
        self.image = None
        self.hsv_image = None
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([179, 255, 255])
        self.running = False
        self.window_name = 'HSV Tracker'
        self.init_ui()

    def get_hsv_on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_value = self.hsv_image[y, x]
            cv2.setTrackbarPos('H Lower', self.window_name, hsv_value[0])
            cv2.setTrackbarPos('S Lower', self.window_name, hsv_value[1])
            cv2.setTrackbarPos('V Lower', self.window_name, hsv_value[2])
            cv2.setTrackbarPos('H Upper', self.window_name, hsv_value[0])
            cv2.setTrackbarPos('S Upper', self.window_name, hsv_value[1])
            cv2.setTrackbarPos('V Upper', self.window_name, hsv_value[2])

    def init_ui(self):
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('H Lower', self.window_name, self.hsv_lower[0], 179, self.nothing)
        cv2.createTrackbar('S Lower', self.window_name, self.hsv_lower[1], 255, self.nothing)
        cv2.createTrackbar('V Lower', self.window_name, self.hsv_lower[2], 255, self.nothing)
        cv2.createTrackbar('H Upper', self.window_name, self.hsv_upper[0], 179, self.nothing)
        cv2.createTrackbar('S Upper', self.window_name, self.hsv_upper[1], 255, self.nothing)
        cv2.createTrackbar('V Upper', self.window_name, self.hsv_upper[2], 255, self.nothing)
        cv2.setMouseCallback(self.window_name, self.get_hsv_on_click)

    def nothing(self, x):
        pass

    def update_hsv_values(self):
        h_lower = cv2.getTrackbarPos('H Lower', self.window_name)
        s_lower = cv2.getTrackbarPos('S Lower', self.window_name)
        v_lower = cv2.getTrackbarPos('V Lower', self.window_name)
        h_upper = cv2.getTrackbarPos('H Upper', self.window_name)
        s_upper = cv2.getTrackbarPos('S Upper', self.window_name)
        v_upper = cv2.getTrackbarPos('V Upper', self.window_name)

        self.hsv_lower = np.array([h_lower, s_lower, v_lower])
        self.hsv_upper = np.array([h_upper, s_upper, v_upper])

    def apply_mask(self):
        mask = cv2.inRange(self.hsv_image, self.hsv_lower, self.hsv_upper)
        result = cv2.bitwise_and(self.image, self.image, mask=mask)
        return result

    def show_combined_image(self):
        result = self.apply_mask()
        combined = np.hstack((self.image, result))
        cv2.imshow(self.window_name, combined)

    def start(self, image_array):
        self.image = image_array
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.running = True
        self.update_hsv_values()
        self.show_combined_image()

    def stop(self):
        self.running = False
        


#=============================================
# 장애물
#=============================================
def vaild_obstacle(points):
    points = np.array(points)
    vaild_points = points[np.isfinite(points)]

    if len(vaild_points) > 0:
        return np.sum(vaild_points < 0.55)
        
    else:
        return np.inf


def obstacle():
    global new_angle, new_speed
    global safe_distance, angle_factor,error, k
    global drive_mode
  
    
    # 장애물 판단 범위
    left_lane_points = lidar_points[35:70] 
    right_lane_points = lidar_points[650:685]
    middle_lane_points = lidar_points[0:1]
    # inf 제거, 카운트
    left_count = vaild_obstacle(left_lane_points)
    right_count = vaild_obstacle(right_lane_points)
    middle_count = vaild_obstacle(middle_lane_points)
    print("Left: ", left_count)
    print("Right: ", right_count)
    print("middle: ", middle_count)
    
    if middle_count > 0 and middle_count != np.inf:
        print("중간 장애물")
        move_car(-100, Fix_Speed, 13) # 왼
        move_car(100, Fix_Speed, 20) # 오
        move_car(-100, Fix_Speed, 8) # 왼
        #drive_mode = 0
    elif left_count != np.inf and right_count != np.inf:
        if left_count > right_count: # 1차선에 장애물 있음
            print("1차선 장애물")
            move_car(20, Fix_Speed, 10) # 오
            move_car(-30, Fix_Speed, 20) # 왼
            move_car(50, Fix_Speed, 8) # 오
            #drive_mode = 0
        elif left_count < right_count: # 2차선에 장애물
            print("2차선 장애물")
            move_car(-30, Fix_Speed, 10.5) # 왼
            move_car(30, Fix_Speed, 20) # 오
            move_car(-50, Fix_Speed, 8) # 왼
            #drive_mode = 0
 
    drive(int(new_angle), Fix_Speed)
            
    new_speed = Fix_Speed
    drive(int(new_angle), new_speed)

def hsv_rubbercon():

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    upper_orange = np.array([179, 255, 255])
    lower_orange = np.array([0, 38, 255])
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    cv2.imshow("rubbercon_hsv", result)
    
#=============================================
# 정지선이 있는지 체크해서 True/False 값을 반환하는 함수
#=============================================
def check_stopline():
    global stopline_num

    # 원본 영상을 화면에 표시
    #cv2.imshow("Original Image", image)
    
    # image(원본이미지)의 특정영역(ROI Area)을 잘라내기
    #기본값300:480, 0:640, 100:120, 200:440
    roi_img = image[270:320, 0:640] 
    cv2.imshow("ROI Image", roi_img)

    # HSV 포맷으로 변환하고 V채널에 대해 범위를 정해서 흑백이진화 이미지로 변환
    hsv_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) 
    upper_white = np.array([255, 255, 255])
    lower_white = np.array([0, 0, 180])
    binary_img = cv2.inRange(hsv_image, lower_white, upper_white)
    #cv2.imshow("Black&White Binary Image", binary_img)

    # 흑백이진화 이미지에서 특정영역을 잘라내서 정지선 체크용 이미지로 만들기
    stopline_check_img = binary_img[0:50, 150:480] 
    #cv2.imshow("Stopline Check Image", stopline_check_img)
    
    # 흑백이진화 이미지를 칼라이미지로 바꾸고 정지선 체크용 이미지 영역을 녹색사각형으로 표시
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (200,100),(440,120),Green,3)
    cv2.imshow('Stopline Check', img)
    cv2.waitKey(1)
    
    # 정지선 체크용 이미지에서 흰색 점의 개수 카운트하기
    stopline_count = cv2.countNonZero(stopline_check_img)
    
    # 사각형 안의 흰색 점이 기준치 이상이면 정지선을 발견한 것으로 한다
    if stopline_count > 1000:
        print("Stopline Found...! -", stopline_num)
        stopline_num = stopline_num + 1
        #cv2.destroyWindow("ROI Image")
        return True
    
    else:
        return False

#=========================================
#  사다리꼴 모양으로 이미지 자르기
#=========================================
def apply_trapezoid_mask(img):
    mask = np.zeros_like(img)
    height, width = img.shape[:2]

    # 사다리꼴 모양의 좌표 정의
    trapezoid = np.array([

        [width * 2//6, height * 1//6],
        [width * 4//6, height * 1//6],
        [width,        height * 5//6],
        [0,            height * 5//6]

    ], np.int32)

    # 사다리꼴 모양으로 마스크 이미지 채우기
    cv2.fillPoly(mask, [trapezoid], 255)

    cv2.imshow("Mask", mask)
    
    # 마스크를 사용하여 이미지 자르기
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

class LaneDetector:
    def __init__(self, camera_processor=None, slide_window_processor=None):
        self.camera_processor = camera_processor
        self.slide_window_processor = slide_window_processor
        self.warped_frame = None
        self.processed_frame = None
        
    def simple_moving_average(self, values):
        return sum(values) / len(values) if values else 0
        
    def set_processor(self, camera, slide_window):
        self.camera_processor = camera
        self.slide_window_processor = slide_window

    def detect(self, frame):
        # 카메라 이미지 업데이트
        self.warped_frame = self.camera_processor.process_image(frame)
        
        if self.warped_frame is not None:
            lane_detected, left_x, right_x, self.processed_frame, flag = self.slide_window_processor.w_slidewindow(self.warped_frame)
            if self.processed_frame is not None:
                # 수평 슬라이드 윈도우 메서드 호출 및 결과 처리
                final_img, steer_theta = self.slide_window_processor.h_slidewindow(self.processed_frame, left_x, right_x, flag)
                return lane_detected, left_x, right_x, self.processed_frame, final_img
            return lane_detected, left_x, right_x, self.processed_frame, None
        return False, None, None, frame, None



#Log 기록하는 함수
#import json
#from datetime import datetime

class Logging:
    def __init__(self, filename='/home/xytron/log.json'):
        self.prev_time = time.time()
        self.filename = filename 
        self.img_path = "/home/xytron/xycar_imgs/img"
        self.warp_path = "/home/xytron/xycar_imgs/warp"
        self.processed_path = "/home/xytron/xycar_imgs/processed"
        self.log_entries = []  # 메모리에 로그 데이터를 저장할 리스트
        self.log_entries.append({
                                "image" : None,
                                "time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "position": (0,0),
                                "direction": 0,
                                "speed": 0,
                                "lidar": [],
                                "ultrasound": [],
                                "AR_Tag": {}
                                })
        self.delete_imgs(self.img_path[:-4])
    
    def delete_imgs(self, path):
        files = os.listdir(path)
        print(path)
        for file in files:
            file_path = os.path.join(path,  file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print("Error : ", e)
    
    def write_log(self, car_speed, car_angle):
        # 로그 항목을 구성하여 메모리 리스트에 추가
        global lane_detector
        if time.time() -  self.prev_time < 0.05:
            return
        else:
            self.prev_time = time.time()
        img_filename = self.img_path + str(len(self.log_entries)) + ".jpg"
        warp_filename = self.warp_path + str(len(self.log_entries)) + ".jpg"
        processed_filename = self.processed_path + str(len(self.log_entries)) + ".jpg"
        log_entry = {
            "image": img_filename,
            "warp" : warp_filename,
            "processed" : processed_filename,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "position": ( self.log_entries[-1]["position"][0] + car_speed*math.cos(car_angle), self.log_entries[-1]["position"][1] + car_speed*math.sin(car_angle)),
            "direction": car_angle,
            "speed": car_speed,
            "lidar": lidar_points,
            "ultrasound": ultra_msg,
            "AR_Tag": {"AR_ID": ar_msg['ID'], "AR_DX": ar_msg['DX'], "AR_DZ": ar_msg['DZ']}
        }  
        try:
            cv2.imwrite(img_filename, compressed_image)
            cv2.imwrite(warp_filename, lane_detector.warped_frame)
            cv2.imwrite(processed_filename, lane_detector.processed_frame)
        except Exception as e:
            pass
            #print("some imgs are not existed : ", e)
        self.log_entries.append(log_entry)

    def save_logs_to_file(self):
        # 프로그램 종료 시 모든 로그를 파일에 저장
        with open(self.filename, 'w') as file:
            json.dump(self.log_entries, file, indent=4)
        print("#### log is saved")


#=============================================
# 실질적인 메인 함수 
#=============================================
def start():
    global safe_distance, angle_factor,error, k
    global motor, ultra_msg, image 
    global new_angle, new_speed
    global CURRENT_ANGLE
    CURRENT_ANGLE = 0
    global logger 
    logger = Logging()
    global lane_detector
    global drive_mode
    lane_detector = LaneDetector()
    
    
    DRIVE = 0
    STARTING_LINE = 1
    TRAFFIC_SIGN = 2
    SENSOR_DRIVE = 3
    LANE_DRIVE = 4
    AR_DRIVE = 5
    PARKING = 7
    FINISH = 9
    OBSTACLE_DRIVE = 10
    HSV = 11
	
    # 처음에 어떤 미션부터 수행할 것인지 여기서 결정합니다. 
    drive_mode = HSV
    cam_exposure(100)  # 카메라의 Exposure 값을 변경
    
    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, compressed_image_callback, queue_size=1)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, ar_callback, queue_size=1 )
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)

    #=========================================
    # 발행자 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("UltraSonic Ready ----------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    rospy.wait_for_message("ar_pose_marker", AlvarMarkers)
    print("AR detector Ready ----------")

    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")
	
    # 일단 차량이 움직이지 않도록 정지상태로 만듭니다.  
    stop_car(10) # 1초 동안 정차
	
    #=========================================
    # 메인 루프 
    #=========================================
    camera_processor = camera_processing.CameraProcessing()
    slide_window_processor = slide_window.SlideWindow(None)
    lane_detector.set_processor(camera_processor, slide_window_processor)
    prev_data = {}
    move_car(0, 0, 1)
    prev_control = 0
    ar_time = time.time()
    tracker = HSVTracker()
    
    while not rospy.is_shutdown():
        #stop_car(50)
        start = time.time()
        while drive_mode == DRIVE:
            new_speed = 20
            lane_detected, left_x, right_x, processed_frame, final_img = lane_detector.detect(image)
            #hsv_rubbercon()
            cv2.waitKey(1)
            
            if lane_detected:
                x_midpoint = (left_x + right_x) // 2 
                new_angle = (x_midpoint - View_Center) / 2
                
                #=========================================
                # new_angle에 이동평균값을 적용
                #=========================================
                angle_avg.add_sample(new_angle)
                new_angle = angle_avg.get_mmed()
                
                #=========================================
                # PID 제어 적용
                #=========================================
                new_angle = int(pid.pid_control(new_angle))

                drive(new_angle, new_speed)  
				
            else:
                # 차선인식이 안됐으면 기존 핸들값을 사용하여 주행합니다. 	
                drive(new_angle, new_speed)
            cv2.imshow("Original Frame", image)
        # 처리된 영상을 화면에 표시
            cv2.imshow("Processed Frame", processed_frame)
            if final_img is not None:
            # h_slidewindow 결과도 화면에 표시
                cv2.imshow("Horizontal Slide Window Result", final_img)
            
            #left_points = lidar_points[0:180] 
            #right_points = lidar_points[629:719]
            #left_distance = vaild_mean(left_points)
            #right_distance = vaild_mean(right_points)
            
    
            #print("left_distance: ", left_distance)
            #print("right_distance: ", right_distance)
            #if left_distance != np.inf and right_distance != np.inf:    
            #    if left_distance < 0.6:
            #        drive_mode = SENSOR_DRIVE

            if cv2.waitKey(1) & 0xFF == ord('q'):  # q를 누르면 루프 탈출
                break
            
        cv2.destroyAllWindows() 
        # ======================================
        # 출발선으로 차량을 이동시킵니다. 
        # AR태그 인식을 통해 AR태그 바로 앞에 가서 멈춥니다.
        # 신호등인식 TRAFFIC_SIGN 모드로 넘어갑니다.  
        # ======================================
        while drive_mode == STARTING_LINE:
            cv2.imshow("image", image)
            # 전방에 AR태그가 보이는지 체크합니다.             
            ar_ID, z_pos, x_pos = check_AR()
            cv2.imshow('AR Detecting', image)

            cv2.waitKey(1)
            filtered_data, prev_data = filter_ar_data(ar_msg, prev_data)
            prev_control = traffic_ar_drive(filtered_data, prev_control)
            
            if (z_pos < 130):  
                # AR태그가 가까워지면 다음 미션으로 넘어갑니다.
                drive_mode = TRAFFIC_SIGN     
                cam_exposure(100)  # 카메라의 Exposure 값을 변경
                stop_car(10)
                print("----- Traffic Sign Detecting... -----")
                # 열려 있는 모든 OpenCV 윈도우 창을 닫습니다. 
                #cv2.destroyAllWindows()
                
        # ======================================
        # 출발선에서 신호등을 찾습니다. 
        # 일단 정차해 있다가 파란색 불이 켜지면 출발합니다.
        # AR_DRIVE 모드로 넘어갑니다.  
        # ======================================
        while drive_mode == TRAFFIC_SIGN:
            Fix_Speed = 5
            # 앞에 있는 신호등에 파란색 불이 켜졌는지 체크합니다.  
            result = check_traffic_sign()
			
            if (result == True):
                # 신호등이 파란불이면 AR_DRIVE 모드로 넘어갑니다.
                drive_mode = AR_DRIVE
                cam_exposure(100)  # 카메라의 Exposure 값을 변경
                print ("----- AR Following Start... -----")
                ar_time = time.time()
                # 열려 있는 모든 OpenCV 윈도우 창을 닫습니다. 
                #cv2.destroyAllWindows()
                
        # ======================================
        # AR태그를 찾고 AR 주행을 시작합니다.
        # ======================================
        retry_count = 0
        while drive_mode == AR_DRIVE:
            # 전방에 AR태그가 보이는지 체크합니다.   
            ar_ID, z_pos, x_pos = check_AR()
            cv2.imshow('AR following', image)
            cv2.waitKey(1)
            filtered_data, prev_data = filter_ar_data(ar_msg, prev_data)
            ar_drive(filtered_data)
            if filtered_data["ID"] is None:
                if time.time() - ar_time > 3:
                    drive_mode = DRIVE
            else:
                ar_time = time.time()
            
        # ======================================
        # 센서로 미로주행을 진행합니다.
        # AR이 보이면 차선인식주행 LANE_DRIVE 모드로 넘어갑니다. 
        # ======================================
        while drive_mode == OBSTACLE_DRIVE:
            # 라이다센서를 이용해서 미로주행을 합니다. 
            obstacle()
            # AR태그를 발견하면 차선주행 모드로 변경합니다. 
        # ======================================
        # 장애물 회피 주행
        # ======================================
        
        while drive_mode == SENSOR_DRIVE:
            # 라이다센서를 이용해서 미로주행을 합니다. 
            sensor_drive()
            
            left_points = lidar_points[0:180] 
            right_points = lidar_points[629:719]
            left_distance = vaild_mean(left_points)
            right_distance = vaild_mean(right_points)
            
            lane_detected, left_x, right_x, processed_frame, final_img = lane_detector.detect(image)
            if lane_detected and left_distance > 0.6:
                    drive_mode = DRIVE
        
        
        while drive_mode == HSV:
            tracker.start(image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # ======================================
        # 주행을 끝냅니다. 
        # ======================================
        if drive_mode == FINISH:
           
            # 차량을 정지시키고 모든 작업을 끝냅니다.
            stop_car(10) # 1초간 정차 
            print ("----- Bye~! -----")
            return            

    logger.save_logs_to_file()
    stop_car(10) # 정차 


#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
