# 융합캡스톤디자인 팀 제네시스
## 주제

YOLOv5 기반 젯슨 나노 교통상황 음성 알림 시스템 : 우회전을 중심으로

## 개요

실시간 영상처리 기술을 활용하여 우회전 가능 여부와 전방 사물에 대한 충돌 가능성 여부를 소리로 알려주는 기술을 개발하고, 젯슨 나노를 이용하여 이에 대한 프로토타입을 제작하고자 한다.  

지난 22년 7월와 23년 1월, 우회전 관련 교통 법규 개정 이후 갑작스럽고 잦은 변화로 인해 많은 운전자들이 적응하기 힘들어했다. 또한 단안 시각장애인의 경우 일반 운전자들에 비해 우측 사각지대가 넓어 우회전 시 더욱 위험하다.  

전체 교통사고 중 약 40%가 교차로에서 발생한다고 하는 국토교통부의 자료도 존재하니, 이러한 문제점들을 해결하기 위해 우회전 가능 여부 알림 시스템을 개발하여 교차로 신호 위반률을 감소시키고자 한다. 또한 가장 사고가 빈번하게 일어나는 오토바이와 보행자를 대상으로 충돌 방지 시스템을 개발하여 교차로 사고율을 낮추며, 나아가 여러 취약 계층 및 일반 운전자들의 사고율 감소를 기대한다. 

## 실행환경 

### 준비물

1. 젯슨 나노 B01  
2. 5V PWM 팬  
3. 20W 어댑터
4. 32GB 이상의 microSD 카드  
5. 키보드  
6. 모니터  
7. 마우스  
8. USB 스피커  
9. USB 카메라

### 소프트웨어 설치

1. Jetson Nano에 JetPack 4.6 설치
2. 램 사용량 저하를 위해 gdm3 제거 및 lightdm 설치
   ```
   sudo apt-get install lightdm
   sudo apt-get purge gdm3
   ```
3. swap 공간 설정
   ```
   sudo apt-get update
   sudo apt-get upgrade
   sudo apt-get install nano
   sudo apt-get install dphys-swapfile

   sudo nano /sbin/dphys-swapfile
   ```
   파일 내의 값들을 아래와 같이 수정 후, [ctrl]+[x], [y], [Enter]를 눌러 저장하고 닫기
   ```
   CONF_SWAPSIZE=4096
   CONF_SWAPFACTOR=2
   CONF_MAXSWAP=4096
   ```
   ```
   sudo nano /etc/dphys-swapfile
   ```
   위와 같은 값으로 편집, [ctrl]+[x], [y], [Enter]를 눌러 저장하고 닫기
   ```
   sudo reboot
   ```
4. OpenCV 4.5.4 with CUDA 설치
   ```
   wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-5-4.sh
   sudo chmod 755 ./OpenCV-4-5-4.sh
   ./OpenCV-4-5-4.sh
   ```
5. SD카드 공간 확보와 스왑 공간 해제
   ```
   rm OpenCV-4-5-4.sh
   sudo /etc/init.d/dphys-swapfile stop
   sudo apt-get remove --purge dphys-swapfile
   sudo rm -rf ~/opencv
   sudo rm -rf ~/opencv_contrib
   ```
6. PyTorch1.8 + torchvision v0.9.0 설치
   ```
   wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
   sudo apt-get install python3-pip libopenblas-base libopenmpi-dev

   pip3 install Cython
   pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

   sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
   git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
   cd torchvision
   export BUILD_VERSION=0.9.0
   python3 setup.py install --user
   cd ../ 
   ```
7. YOLOv5 설치
   ```
   https://github.com/CSID-DGU/2023-1-SCS4031-GENESIS.git
   cd 2023-1-SCS4031-GENESIS/yolov5-python3.6.9-jetson
   ```
   다음 내용 requirements.txt에서 제거
   ```
   numpy>=1.18.5
   opencv-python>=4.1.2
   torch>=1.7.0
   torchvision>=0.8.1
   ```
   이후 requirements.txt 설치
   ```
   python3 -m pip install
   python3 -m pip install -r requirements.txt
   ```
8. detect.py 실행
   ```
   python3 detect.py --source 0 --weights final.pt --img 416
   ```
   
## 결과물

하드웨어 결과물 :  

![image](https://github.com/CSID-DGU/2023-1-SCS4031-GENESIS/assets/101885318/b41f1e62-748c-4240-a8fa-df87914aeceb)  
  
실제 주행 영상을 활용한 시스템 사용 영상 :  

https://youtu.be/Z9FmOT0r8dQ

## 팀원

😀 류강현 : 팀장 및 하드웨어 개발

😀 최다희 : 스프트웨어 개발 및 데이터 전처리

😀 백성욱 : 모델 학습 및 소프트웨어 개발

## 레퍼런스

https://whiteknight3672.tistory.com/316  
