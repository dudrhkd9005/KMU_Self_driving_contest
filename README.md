# 2024 국민대학교 제 7회 자율주행 경진대회 팀 KOBOT
결과: 2등 수상

## 구현내용

### 신호등 인식주행
- check_traffic_sign()
  roi영역을 내의 hsv값을 추출해 신호등의 불빛을 검출 

### AR주행
- ar_drive()
  AR태그를 따라서 주행하는 미션, AR태그를 중심으로 차와의 x, y, z pos 거리를 계산 후 차량을 조향

### 차선주행
- drive()
  angle과, speed 값이 들어가고 angle은 Sliding Window와 PID제어로 도출해낸 조향값이 들어간다.

### 라바콘 주행
- sensor_drive()
  lidar sensor를 이용해 좌우, 앞뒤의 라바콘을 검출, 이후 라바콘사이를 피해 주행한다.

### 장애물 회피주행
- obstacle()
  lidar sensor를 이용해 전방의 차량 장애물을 회피하는 주행을 수행한다.


![Screenshot from 2024-06-29 13-20-46](https://github.com/user-attachments/assets/633b526a-d161-46ce-a140-6ca06a9dc6e5)
