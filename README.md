## DQN 실습

### 6/5 진행상황
 - tran 가능한 것 까지 확인
 - test 코드 구현 필요


### **How to train**
    python main.py                       # 기본 설정으로 실행
    python main.py -rn 4 -bs 64          # 배치사이즈 54로 39999*4 번 학습을 진행시키는 경우
    python main.py --double --dueling    # DoubleDQN과 DuelingDQN 적용하는 경우



## 모두의 연구소 Environment

### **How to use**
    train 코드를 작성하실 때 Sim.py 내에 있는 Simulator class를 이용하시면 됩니다.
    gym, grid world 등 강화학습 예시를 참고하시면서 진행 바랍니다.

### **Make virtual environment using conda**
    
    conda create -n venv
    conda activate venv

### **Prerequisite**
    
    pip install -r requirements.txt
    sudo apt install xvfb
    sudo apt install ffmpeg

### **Files**
train/test 데이터는 최종 목적지에 들어오기 전 가져와야 할 아이템들의 리스트로 이루어져습니다.

- Sim.py : 강화학습 환경
- draw_utils.py : visualization 코드
- factory_order_train.csv : train 데이터
- factory_order_test.csv  : test 데이터
- obstacles.csv : 장애물 위치 좌표
- box.csv : 아이템 위치 좌표


### **Troubleshooting**
    
    * 아래와 같은 pyglet 에러 나오면 주피터 노트북으로 실행하시고,
    pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
    -> xvfb-run -a jupyter notebook

    그후 주피터 노트북에서 train 파일을 실행하시면 됩니다.
    run train.py
