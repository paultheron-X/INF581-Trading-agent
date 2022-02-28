import time
import numpy as np
#import matplotlib.pyplot as plt

from models.dqn import DQNAgent_ds

from config_mods import config_dqn_deepsense as config


agent = DQNAgent_ds()

# Animation Loop
game_reward = []
current_reward = 0
counter = 0
countersuper = 0

num_episode = config['num_episode']

while counter < num_episode:
    
    state, terminal = agent.reset()
    current_reward = 0
    issue = state.copy()
    while not terminal:

        t1 = time.time()
    
        recom_action = agent.predict(issue)
        
        old_state = issue.copy()
    
        issue, reward, action_chosen, terminal = agent.act(recom_action)
        
        agent.remember(old_state, action_chosen, reward, issue, terminal)
        
        agent.trading_lessons()

        current_reward += reward
        t2 = time.time()
    
    counter +=1
    game_reward.append(current_reward)
    ind = len(game_reward)
    if ind % config['replace_target'] == 0 and ind > config['replace_target']:
            agent.update_params()
    if ind % config['training_state'] ==0:                    # print current learning infos every 10 games
        avg = np.mean(game_reward[max(ind-100, 0):ind])
        print("> Game Numer : " + str(ind) + " | Last Game Reward = " + str(current_reward) + " | Average R on 100 last games : " + str(avg) + " | Exploration rate : " + str(agent.get_exploration()) + " | Current FPS : " + str(round(1/(t2-t1))))

#plt.plot(game_reward)
#plt.show()