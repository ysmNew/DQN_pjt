from string import ascii_uppercase
#from draw_utils import *
from pyglet.gl import *
import numpy as np
import pandas as pd
import os


local_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


class Simulator:
    def __init__(self, args):
        '''
        height : 그리드 높이
        width : 그리드 너비 
        inds : A ~ Q alphabet list
        '''
        self.args = args
        # Load data
        if self.args.test:
            self.files = pd.read_csv(os.path.join(local_path, "./data/factory_order_test.csv"))
        else:
            self.files = pd.read_csv(os.path.join(local_path, "./data/factory_order_train.csv"))
        self.height = 10
        self.width = 9
        self.inds = list(ascii_uppercase)[:17]
# 아이템 박스에 진입할 수 있는 방향 설정
# (박스 좌표): (들어가는 방향, 나가는 방향)
        self.shelf = {(5, 0): (2,3), (4, 0): (2,3), (3, 0): (2,3), (2, 0): (2,3),
                      (0, 0): (0,1), (0, 1): (0,1), (0, 2): (0,1), (0, 3): (0,1), (0, 4): (0,1),
                      (0, 5): (0,1), (0, 6): (0,1), (0, 7): (0,1), (0, 8): (0,1),
                      (2, 8): (3,2), (3, 8): (3,2), (4, 8): (3,2), (5, 8): (3,2)}

    def set_box(self):
        '''
        아이템들이 있을 위치를 미리 정해놓고 그 위치 좌표들에 아이템이 들어올 수 있으므로 그리드에 2으로 표시한다.
        데이터 파일에서 이번 에피소드 아이템 정보를 받아 가져와야 할 아이템이 있는 좌표만 3으로 표시한다.
        self.local_target에 에이전트가 이번에 방문해야할 좌표들을 저장한다.
        따라서 가져와야하는 아이템 좌표와 end point 좌표(처음 시작했던 좌표로 돌아와야하므로)가 들어가게 된다.

        수정)
        각 그리드를 0~5로 표시
        0: 장애물
        1: 길
        2: 빈 아이템 박스
        3: 미래의 타겟
        4: 현재 타겟
        5: 현재 위치
        '''
        box_data = pd.read_csv(os.path.join(local_path, "./data/box.csv"))

        # 물건이 들어있을 수 있는 경우
        for box in box_data.itertuples(index = True, name ='Pandas'):
            self.grid[getattr(box, "row")][getattr(box, "col")] = 2

        # 물건이 실제 들어있는 경우
        order_item = list(set(self.inds) & set(self.items))
        order_csv = box_data[box_data['item'].isin(order_item)]

        for order_box in order_csv.itertuples(index = True, name ='Pandas'):
            self.grid[getattr(order_box, "row")][getattr(order_box, "col")] = 3
            # local target에 가야 할 위치 좌표 넣기
            self.local_target.append(
                [getattr(order_box, "row"),
                 getattr(order_box, "col")]
                )

# 좌표 기준으로 sort하면 안됨 ㅠㅠ
        #self.local_target.sort()
        self.local_target.append([9,4]) 

        # 알파벳을 Grid에 넣어서 -> grid에 2Dconv 적용 가능

    def set_obstacle(self):
        '''
        장애물이 있어야하는 위치는 미리 obstacles.csv에 정의되어 있다. 이 좌표들을 0으로 표시한다.
        '''
        obstacles_data = pd.read_csv(os.path.join(local_path, "./data/obstacles.csv"))
        for obstacle in obstacles_data.itertuples(index = True, name ='Pandas'):
            self.grid[getattr(obstacle, "row")][getattr(obstacle, "col")] = 0

    def reset(self, epi):
        '''
        reset()은 첫 스텝에서 사용되며 그리드에서 에이전트 위치가 start point에 있게 한다.
        :param epi: episode, 에피소드 마다 가져와야 할 아이템 리스트를 불러올 때 사용
        :return: 초기셋팅 된 히스토리
        :rtype: np.array
        _____________________________________________________________________________________
        items : 이번 에피소드에서 가져와야하는 아이템들
        terminal_location : 현재 에이전트가 찾아가야하는 목적지
        local_target : 한 에피소드에서 찾아가야하는 아이템 좌표, 마지막 엔드 포인트 등의 위치좌표들
        actions: visualization을 위해 에이전트 action을 저장하는 리스트
        curloc : 현재 위치
        '''

        # initial episode parameter setting
        self.epi = epi%39999
        self.items = list(self.files.iloc[self.epi])[0]
        self.cumulative_reward = 0
        self.terminal_location = None
        self.local_target = []
        self.actions = []

        # initial grid setting
        self.grid = np.ones((self.height, self.width), dtype="float16")

        # set information about the gridworld
        self.set_box()
        self.set_obstacle()

        # start point를 grid에 표시
        self.curloc = [9, 4]
        self.grid[int(self.curloc[0])][int(self.curloc[1])] = 5
        
        self.done = False

# 최초 그리드를 4번 쌓아서 히스토리 초기화(4,10,9)
        self.history = np.stack((self.grid,self.grid,self.grid,self.grid,self.grid),axis=0)
 
# 그리드(10,9)가 아니라 히스토리(4,10,9) 상태 반환
        return self.history
    

    def apply_action(self, action, cur_x, cur_y):
        '''
        에이전트가 행한 action대로 현 에이전트의 위치좌표를 바꾼다.
        action은 discrete하며 4가지 up,down,left,right으로 정의된다.
        
        :param x: 에이전트의 현재 x 좌표
        :param y: 에이전트의 현재 y 좌표
        :return: action에 따라 변한 에이전트의 x 좌표, y 좌표
        :rtype: int, int
        '''
        new_x = cur_x
        new_y = cur_y
        # up
        if action == 0:
            new_x = cur_x - 1
        # down
        elif action == 1:
            new_x = cur_x + 1
        # left
        elif action == 2:
            new_y = cur_y - 1
        # right
        else:
            new_y = cur_y + 1

        return int(new_x), int(new_y)

# 거리 계산 함수
    def calculate_distance(self, x, y):
        tar_x, tar_y = self.terminal_location
        dist = abs(x-tar_x)+abs(y-tar_y)
        
        return dist
    
# 리워드 계산 함수
    def move_reward(self, cur_x, cur_y, new_x, new_y):
        cur_dist = self.calculate_distance(cur_x, cur_y)
        new_dist = self.calculate_distance(new_x, new_y)
        print(cur_dist,new_dist, end =' ')
        
        return self.args.forward_reward if cur_dist>new_dist else self.args.backward_reward


    def get_reward(self, cur_x, cur_y, new_x, new_y, out_of_boundary):
        '''
        get_reward함수는 리워드를 계산하는 함수이며, 상황에 따라 에이전트가 action을 옳게 했는지 판단하는 지표가 된다.
        :param new_x: action에 따른 에이전트 새로운 위치좌표 x
        :param new_y: action에 따른 에이전트 새로운 위치좌표 y
        :param out_of_boundary: 에이전트 위치가 그리드 밖이 되지 않도록 제한
        :return: action에 따른 리워드
        :rtype: float
        '''

        # 바깥으로 나가는 경우 (벽에 부딪히는 경우)
        if any(out_of_boundary):
            reward = self.args.obs_reward
        else:
            # 장애물에 부딪히는 경우 + 빈 아이템박스에 들어가려는 경우
            if self.grid[new_x][new_y] in (0,2,3):
                reward = self.args.obs_reward  

            # 현재 목표에 도달한 경우
            elif new_x == self.terminal_location[0] and new_y == self.terminal_location[1]:
                reward = self.args.goal_reward

            # 그냥 움직이는 경우 
            else:
                reward = self.move_reward(cur_x, cur_y, new_x, new_y)

        return reward


    def step(self, action):
        ''' 
        에이전트의 action에 따라 step을 진행한다.
        action에 따라 에이전트 위치를 변환하고, action에 대해 리워드를 받고, 어느 상황에 에피소드가 종료되어야 하는지 등을 판단한다.
        에이전트가 endpoint에 도착하면 gif로 에피소드에서 에이전트의 행동이 저장된다.
        :param action: 에이전트 행동
        :return: # 리턴 수정
            h, 4개의 그리드가 쌓인 히스토리
            action, 에이전트 행동
            reward, 리워드
            cumulative_reward, 누적 리워드
            h_prime, 다음 히스토리
            done, 종료 여부
            #goal_ob_reward, goal까지 아이템을 모두 가지고 돌아오는 finish율 계산을 위한 파라미터
        :rtype: np.array, int, float, float, np.array, bool, bool/str
        (Hint : 시작 위치 (9,4)에서 up말고 다른 action은 전부 장애물이므로 action을 고정하는 것이 좋음)

        goal_ob_reward, 
            평소에는 False
            목표 아이템에 도달했을 때 True
            아이템을 다 먹고 도착지에 도달했을 때 'finish'
        '''

        self.terminal_location = self.local_target[0]
# 현재 목표를 4로 표시
        self.grid[int(self.terminal_location[0])][int(self.terminal_location[1])] = 4
# 현재 그리드를 히스토리에 추가
        h = self.get_history()
        cur_x,cur_y = self.curloc
        self.actions.append((cur_x, cur_y))

        goal_ob_reward = False
        
        new_x, new_y = self.apply_action(action, cur_x, cur_y)

        out_of_boundary = [new_x < 0, new_x >= self.height, new_y < 0, new_y >= self.width]
        
# 허용되지 않은 방향에서 아이템 박스에 진입했을 경우 벽에 부딪힌 것으로 판단
# 현재 (2,0),(2,8),(5,0),(5,8)에서만 작동
        if (new_x, new_y) in self.shelf:
            if action != self.shelf[(new_x, new_y)][0]:
                print(action, '벽으로 못 들어감')
                out_of_boundary.append(True)
# 아이템 박스에서 허용되지 않은 방향으로 나가는 경우 벽에 부딪힌 것으로 판단
# 현재는 반드시 후진해서 나가므로 동작하지 않음 
        if (cur_x,cur_y) in self.shelf:
            if action != self.shelf[(cur_x,cur_y)][1]:
                print(action, '벽으로 못 나감')
                out_of_boundary.append(True)
        
        # 바깥으로 나가는 경우(벽에 부딪힌 경우) 종료
        if any(out_of_boundary):
            print(action, '아이쿠!')
            self.done = True
        else:
            # 장애물에 부딪히는 경우 종료
            if self.grid[new_x][new_y] == 0:
                print(action, '쿵!')
                self.done = True

            # 빈 아이템 박스는 들어가지 않기 ---------> 종료
            elif self.grid[new_x][new_y] in (2,3):
                print(action, '비었어')
                # self.grid[cur_x][cur_y]를 그대로 5로 유지 
                # self.grid[new_x][new_y]를 그대로 2나 3으로 유지
                # self.curloc를 그댈 self.curloc = (cur_x,cur_y)로 유지
                # 대신 reward만 장애물 패널티를 받음 -------------------------> 그냥 죽어라
                self.done = True
            
            # 현재 목표에 도달한 경우, 다음 목표설정
            elif self.grid[new_x][new_y] == 4:
                print(action, self.local_target[0], '찾았다!')
                # end point 일 때
                if (new_x, new_y) == (9,4):
                    print('###########################도착###########################')
                    self.done = True
                else:
                    print('다음 목표는', self.local_target[1])
                    
                self.local_target.remove(self.local_target[0])
                self.grid[cur_x][cur_y] = 1
                self.grid[new_x][new_y] = 5
                goal_ob_reward = True # 액션마스킹에서 사용
                self.curloc = (new_x, new_y)
                
            else:
# 아이템 박스에서 나가는 경우
                if (cur_x,cur_y) in self.shelf:
                    print(action, '나가자!')
                    self.grid[cur_x][cur_y] = 2
                    self.grid[new_x][new_y] = 5
                    self.curloc = (new_x, new_y)
                    
                # 그냥 길에서 움직이는 경우 
                else:
                    print(action, '..')
                    self.grid[cur_x][cur_y] = 1
                    self.grid[new_x][new_y] = 5
                    self.curloc = (new_x, new_y)

        #print(h)
        #input()
                
        reward = self.get_reward(cur_x, cur_y, new_x, new_y, out_of_boundary)
        print('reward:', reward)
        self.cumulative_reward += reward
        h_prime = self.get_history()

# 보류) 상하좌우 확인하고 액션마스킹 하는 기능
        #action_mask = self.mask_action(new_x,new_y)
        #action_mask = [True,True,True,True]

        if self.done == True:
            if [new_x, new_y] == [9, 4]:
                if self.terminal_location == [9, 4]:
                    goal_ob_reward = 'finish'
# 학습중에는 GIF 저장 x, 테스트 파일 확인 할 때만 저장
                if self.args.test:
                    height = 10
                    width = 9 
                    display = Display(visible=False, size=(width, height))
                    display.start()

                    start_point = (9, 4)
                    unit = 50
                    screen_height = height * unit
                    screen_width = width * unit
                    log_path = "./logs"
                    data_path = "./data"
                    render_cls = Render(screen_width, screen_height, unit, start_point, data_path, log_path)
                    for idx, new_pos in enumerate(self.actions):
                        render_cls.update_movement(new_pos, idx+1)
                    
                    render_cls.save_gif(self.epi)
                    render_cls.viewer.close()
                    display.stop()
        
        return h, action, reward, h_prime, self.done, self.cumulative_reward, goal_ob_reward#, action_mask


    def get_history(self):
        # 이번 그리드를 히스토리 사이즈에 맞게 reshape
        new_grid =  np.reshape([self.grid],(1,10,9)) 
        # history의 마지막장을 떼고 new 그리드를 맨 앞에 붙임
        next_history = np.append(new_grid,self.history[:3,:,:], axis=0) # (1+3,10,9)
        return next_history

#--------------------------------------------------------------------------------------#

# 보류)
# 현재 위치의 상하좌우를 확인해서 이동 불가능한 방향의 액션은 마스킹
    def mask_action(self, cur_x, cur_y):
        near = [(cur_x-1, cur_y),(cur_x+1, cur_y),(cur_x, cur_y-1),(cur_x, cur_y+1)]
        action_mask = [False,False,False,False]
        for i, (x,y) in enumerate(near):
            #print((x,y),self.grid[x][y], end = ' ')
            try:
                if self.grid[x][y] not in (0,2,3): 
                    action_mask[i] = True
    # 그리드 밖은 그대로 False
            except: 
                pass
        #print()
        #print([cur_x, cur_y], action_mask)
        #input()
        return action_mask


            
