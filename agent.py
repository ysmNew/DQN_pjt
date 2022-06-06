import torch.optim as optim
from buffer import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, args):
        self.args = args

# dueling으로 모델 결정
        if self.args.dueling:
            self.online = Dueling_Qnet()
            self.target = Dueling_Qnet()
        else:
            self.online = Qnet()
            self.target = Qnet()

        self.online.to(device)      # gpu 
        self.target.to(device)

        self.target_update()        # 타겟 네트워크 동기화

        self.memory = ReplayBuffer(self.args.buffer_limit)
    
    def train(self):
        optimizer = optim.Adam(self.online.parameters(), lr=self.args.learning_rate)
        for i in range(10):
            h, a, r, h_prime, done_mask = self.memory.sample(self.args.batch_size)
            # gpu
            h, a, r, h_prime, done_mask = h.to(device), a.to(device), r.to(device), h_prime.to(device), done_mask.to(device)

            q_out = self.online(h)
            q_a = q_out.gather(1,a)

# double로 타겟 결정
            if self.args.double: 
                # online_net에서 액션 선택
                q_prime_idx = self.online(h_prime).max(1)[1]
                # target_net에서 q값 계산
                q_primes = self.target(h_prime)
                # online_net에서 고른 액션의, target_net의 q값 뽑기
                max_q_prime = q_primes.gather(1,q_prime_idx.unsqueeze(1))
                
            else:
                max_q_prime = self.target(h_prime).max(1)[0].unsqueeze(1)

        # loss 계산
            target = r + self.args.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def target_update(self):
        self.target.load_state_dict(self.online.state_dict())

#obs: np.array
    def sample_action(self, obs, epsilon, action, goal_ob_reward):
        coin = random.random()
        obs = obs.to(device)
        with torch.no_grad():
            out = self.online.forward(obs)
        
        if goal_ob_reward:
            if action == 0: return 1
            if action == 1: return 0
            if action == 2: return 3
            if action == 3: return 2
            
        elif coin < epsilon:
            act = random.randint(0,3)
            #print(out.detach().numpy()[0], act, 'Random!')
            return act
        else : 
            #print(out.detach().numpy()[0], out.argmax().item())
            return out.argmax().item()


#    def sample_action(self, obs, epsilon, action_mask):
#        out = self.forward(obs)
#        coin = random.random()        
#        if coin < epsilon:
# 0 1 2 3 중에서 True인 값 중에서만 랜덤뽑기
#            act_lst = np.array([0,1,2,3])[action_mask]
#            act = random.choice(act_lst)
#            #print(act_lst, action_mask, act, 'Random!')
#            return act
#        else : 
#            out = out[[[action_mask]]]
            #print(out.detach(), out[0], out[0].item(), out.argmax(), out.argmax().item())
            #print(out.detach(), action_mask, out.argmax().item())
            #input()
#            return out.argmax().item()
