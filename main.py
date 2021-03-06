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
    sim = Simulator(args)
    agent = Agent(args)

    #epsilon_limit = 30000*args.echo
    running_loss = 0.0
    cum_reward = 0.0
    avg_len = 0
    finish_num = 0

    for epi in range(len(sim.files)*args.echo):
        ep = epi%39999
        s = sim.reset(ep) 

        done = False

        # 첫번째 액션을 0으로 고정
        s, a, r, s_prime, done, cr, gr = sim.step(0)
        agent.memory.put((s, a, r, s_prime, 1.0)) # done = False, done_mask = 1.0
        s = s_prime

        while not done:
            for _ in range(10):
                action = agent.sample_action(torch.from_numpy(s).unsqueeze(0),args.epsilon,a,gr)
                s, a, r, s_prime, done, cr, gr = sim.step(action)
                done_mask = 0.0 if done else 1.0
                agent.memory.put((s, a, r, s_prime, done_mask))
                s = s_prime
                
                if gr == 'finish':
                    finish_num += 1
                    writer.add_scalar('finish lenth', len(sim.actions), epi)

                if done:
                    cum_reward += cr
                    avg_len += len(sim.actions)
                    print('target:', sim.target_list)
                    print(sim.actions)
                    print('episode: {}, epsilon: {}'.format(epi,args.epsilon))
                    print('finish: {}, lenth: {}, cr: {}'.format(finish_num,len(sim.actions),cr))
                    print('==================================================')
                    break
                
            if agent.memory.size()>args.start_limit:
                loss = agent.train()
                running_loss += loss

                if epi%args.sync_freq == 0:
                    agent.target_update()
                    #input()

                if epi%args.log_freq == 0:
                    writer.add_scalar('cumulative reward', cum_reward/args.log_freq, epi)
                    writer.add_scalar('average training loss', running_loss/args.log_freq, epi)
                    writer.add_scalar('average lenth', avg_len/args.log_freq, epi)
                    cum_reward = 0.0
                    running_loss = 0.0
                    avg_len = 0
                    
            if epi % 40000 == 0 and epi != 0:
                torch.save(agent.online.state_dict(),'./pt/temp_'+str(epi)+'.pt')
                
    torch.save(agent.online.state_dict(),args.model_path+str(args.try_number)+'.pt')


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('tensor in', "cuda" if torch.cuda.is_available() else "cpu")
    input()
    if not os.path.isdir('./pt'):
        os.mkdir('./pt')
    writer = SummaryWriter('./logs/train_'+str(args.try_number))
    
    main()












        
