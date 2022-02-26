import time
import numpy as np

from models.dqn import *


agent = DQNAgent(state_space = WINDOW_SIZE*NUM_COLUMNS, 
                 action_space = NUM_ACTIONS, 
                 dropout = 0,  
                 hidden_size= 256,
                 pretrained = False, 
                 lr = 0.001, 
                 gamma=0.99, 
                 max_mem_size = 30000, 
                 exploration_rate = 0, 
                 exploration_decay = .9995, 
                 exploration_min = 0,  
                 batch_size = 512)

# Animation Loop
game_reward = []
current_reward = 0
counter = 0
countersuper = 0

number_episode = 10

while counter < number_episode:
    
    counter = 0
    state, terminal = agent.reset()
    current_reward = 0
    issue = state.copy()
    while not terminal:

        t1 = time.time()
    
        recom_action = agent.predict(issue)
        
        old_state = issue.copy()
    
        issue, reward, action_chosen, terminal = agent.act(recom_action)
                
        
        '''if terminal or current_reward==-1:
            print(reward)'''
            
        '''print("\nnumber", countersuper)
        print("old_state",old_state)
        print("issue", issue)
        print("reward", reward)
        print("action", action_chosen)
        print("terminal", terminal )'''
        
        agent.remember(old_state, action_chosen, reward, issue, terminal)
        
        
        agent.trading_lessons()

        current_reward += reward
        t2 = time.time()
    # >>> The car has crashed  
   # print("kill")
    game_reward.append(current_reward)
    ind = len(game_reward)
    if ind % REPLACE_TARGET == 0 and ind > REPLACE_TARGET:
            agent.update_params()
    if ind % 100 ==0:                    # print current learning infos every 10 games
        avg = np.mean(game_reward[max(ind-100, 0):ind])
        print("> Game Numer : " + str(ind) + " | Last Game Reward = " + str(current_reward) + " | Average R on 100 last games : " + str(avg) + " | Exploration rate : " + str(agent.get_exploration()) + " | Current FPS : " + str(round(1/(t2-t1))))

