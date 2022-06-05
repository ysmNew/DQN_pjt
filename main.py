import argparse
import torch

from Sim import *
from agent import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-tn', '--try_number',      type=int,   default=1)
parser.add_argument('-gr', '--goal_reward',     type=float, default=10.0)
parser.add_argument('-or', '--obs_reward',      type=float, default=-0.5)
parser.add_argument('-fr', '--forward_reward',  type=float, default=0.1)
parser.add_argument('-br', '--backward_reward', type=float, default=-0.1)
parser.add_argument('-rn', '--echo',            type=int,   default=1)
parser.add_argument('-bs', '--batch_size',      type=int,   default=256)
parser.add_argument('-bl', '--buffer_limit',    type=int,   default=100000)
parser.add_argument('-sl', '--start_limit',     type=int,   default=50000)
parser.add_argument('-gm', '--gamma',           type=float, default=0.99)
parser.add_argument('-lr', '--learning_rate',   type=float, default=0.0001)
parser.add_argument('-sf', '--sync_freq',       type=int,   default=2000)
parser.add_argument('-lf', '--log_freq',        type=int,   default=200)


# 실행시 test인자를 주면 변수 test에 True가 저장됨
parser.add_argument('--test',    dest='test',    action='store_true')
parser.add_argument('--double',  dest='double',  action='store_true')
parser.add_argument('--dueling', dest='dueling', action='store_true')

args = parser.parse_args()


def main():
    sim = Simulator(args)
    agent = Agent(args)

    # main_args
    epsilon_limit = 30000*args.echo
    running_loss = 0.0
    cum_reward = 0.0
    finish_num = 0

    for epi in range(len(sim.files)*args.echo):
        ep = epi%39999
        h = sim.reset(ep) # 히스토리 [리셋,리셋,리셋,리셋]
        epsilon = max(0.01, 1.0 - (epi/epsilon_limit))
        done = False

        # 첫번째 액션을 0으로 고정
        # h: [1번째,리셋,리셋,리셋], h_prime: [2번째,1번째,리셋,리셋]
        h, a, r, h_prime, done, cr, gr = sim.step(0)
        agent.memory.put((h, a, r, h_prime, 1.0)) # done = False, done_mask = 1.0
        h = h_prime

        while not done:
            for _ in range(10):
                action = agent.sample_action(torch.from_numpy(h).unsqueeze(0),epsilon,a,gr)
                h, a, r, h_prime, done, cr, gr = sim.step(action)
                done_mask = 0.0 if done else 1.0
                agent.memory.put((h, a, r, h_prime, done_mask))
                h = h_prime
                
                if gr == 'finish':
                    finish_num += 1

                if done:
                    print(sim.actions)
                    print('episode: {}, epsilon: {}'.format(epi,epsilon))
                    print('finish: {}, lenth: {}, cr: {}'.format(finish_num,len(sim.actions),cr))
                    print('==================================================')
                    break
                
            if agent.memory.size()>args.start_limit:
                loss = agent.train()
                running_loss += loss
                cum_reward += cr

                if epi%args.sync_freq == 0:
                    agent.target_update()
                    #input()

                if epi%args.log_freq == 0:
                    writer.add_scalar('cumulative reward', cum_reward/args.log_freq, epi)
                    writer.add_scalar('average training loss', running_loss/args.log_freq, epi)
                    cum_reward = 0.0
                    running_loss = 0.0
                    
            if epi % 39999 == 0 and epi != 0:
                torch.save(agent.online.state_dict(),'./pt_temp/'+str(epi)+'.pt')

    torch.save(agent.online.state_dict(),'./pt/state_dict_'+str(args.try_number)+'.pt')


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('tensor in', "cuda" if torch.cuda.is_available() else "cpu")
    input()
    writer = SummaryWriter('./logs/train_'+str(args.try_number))
    
    main()












        
