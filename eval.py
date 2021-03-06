import argparse
import torch
import os

from Sim import *
from agent import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-tn', '--try_number',      type=int,   default=1)
parser.add_argument('-gr', '--goal_reward',     type=float, default=10.0)
parser.add_argument('-or', '--obs_reward',      type=float, default=-0.5)
parser.add_argument('-mr', '--move_reward',     type=float, default=-0.1)
parser.add_argument('-fr', '--forward_reward',  type=float, default=0.1)
parser.add_argument('-br', '--backward_reward', type=float, default=-0.1)
parser.add_argument('-rn', '--echo',            type=int,   default=1)
parser.add_argument('-bs', '--batch_size',      type=int,   default=256)
parser.add_argument('-bl', '--buffer_limit',    type=int,   default=100000)
parser.add_argument('-sl', '--start_limit',     type=int,   default=50000)
parser.add_argument('-es', '--epsilon',         type=float, default=0.1)
parser.add_argument('-gm', '--gamma',           type=float, default=0.99)
parser.add_argument('-lr', '--learning_rate',   type=float, default=0.0001)
parser.add_argument('-sf', '--sync_freq',       type=int,   default=5000)
parser.add_argument('-lf', '--log_freq',        type=int,   default=200)
parser.add_argument('-md', '--model_path',      type=str,   default='./pt/state_dict_')


# 실행시 test인자를 주면 변수 test에 True가 저장됨
parser.add_argument('--test',    dest='test',    action='store_true')
parser.add_argument('--double',  dest='double',  action='store_true')
parser.add_argument('--dueling', dest='dueling', action='store_true')

args = parser.parse_args()


def main():
    PATH = args.model_path+str(args.try_number)+'.pt'
    #PATH = './pt/temp_40000.pt'
    sim = Simulator(args)
    agent = Agent(args)
    agent.online.load_state_dict(torch.load(PATH))
    
    timestep = 0
    total_reward = 0
    finish_num = 0

    actions_file_name='test_'+str(args.try_number)+'.txt'
    f = open(actions_file_name, 'w')

    for epi in range(1226):
        s = sim.reset(epi)
        epsilon = 0
        done = False

        # 첫번째 액션을 0으로 고정
        s, a, r, s_prime, done, cr, gr = sim.step(0)
        s = s_prime
        
        while not done:
            action = agent.sample_action(torch.from_numpy(s).unsqueeze(0),epsilon,a,gr)
            s, a, r, s_prime, done, cr, gr = sim.step(action)
            s = s_prime
            
            if gr == 'finish':
                finish_num += 1

            if done:
                timestep += len(sim.actions)
                total_reward += cr
                
                break
            
        #print('target:', sim.target_list)
        #print('lenth: ',len(sim.actions), '\n', sim.actions)
        print('Episode : {0:>4}, Timestep : {1:>5}, average Reward : {2:0.3f}, Finish Rate : {3:0.3f}'.format(epi,timestep,total_reward/(epi+1),finish_num/(epi+1)))
        #print('==================================================')
                
        if gr == 'finish':
            f.write(str(epi)+'/'+str(sim.target_list)+'/'+str(cr)+'/'+str(len(sim.actions))+'\n')
            f.write(str(sim.actions)+'\n')


    f.close()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('tensor in', "cuda" if torch.cuda.is_available() else "cpu")
    input()
    
    main()

