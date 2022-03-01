import time
import numpy as np
import matplotlib.pyplot as plt

from models.dqn import DQNAgentDeepsense

from config_mods import config_dqn_deepsense as config


agent = DQNAgentDeepsense()

# Animation Loop
game_reward = []
random_game_reward = []
optimal_game_reward = []

normal_game_profit = []
random_game_profit = []
optimal_game_profit = []
    
current_reward = 0
counter = 0
countersuper = 0

num_episode = config['num_episode']

while counter < num_episode:
    
    state, terminal = agent.reset()
    current_reward = 0
    issue = state.copy()
    game_profit = 0
    while not terminal:

        t1 = time.time()
    
        recom_action = agent.predict(issue)
        
        old_state = issue.copy()
    
        issue, reward, action_chosen, terminal = agent.act(recom_action)
        
        agent.remember(old_state, action_chosen, reward, issue, terminal)
        
        agent.trading_lessons()

        current_reward += reward
        t2 = time.time()
        
        if terminal:
            game_profit = reward
            
    
    counter +=1
    
    game_reward.append(current_reward)
    random_reward, optimal_reward = agent.get_reward()
    random_game_reward.append(random_reward)
    optimal_game_reward.append(optimal_reward)
    
    random_profit, optimal_profit = agent.get_profit()
    normal_game_profit.append(game_profit)
    random_game_profit.append(random_profit)
    optimal_game_profit.append(optimal_profit)
    
    random_profit, optimal_profit = agent.get_profit()
    ind = len(game_reward)
    if ind % config['replace_target'] == 0 and ind > config['replace_target']:
            agent.update_params()
    if ind % config['training_state'] ==0:                    # print current learning infos every 10 games
        avg = np.mean(game_reward[max(ind-100, 0):ind])
        avg_random = np.mean(random_game_reward[max(ind-100, 0):ind])
        avg_optimal = np.mean(optimal_game_reward[max(ind-100, 0):ind])
        try :
            FPS = str(round(1/(t2-t1)))
        except:
            FPS = "--"
        print("\n> Game Numer : " + str(ind) + " | Last Game Reward = " + str(current_reward) + " | Average R on 100 last games : " + str(avg) + " | Exploration rate : " + str(agent.get_exploration()) + " | Current FPS : " + str(round(1/(t2-t1))))
        print("     > Last game profit : " + str(game_profit) + " | Last game random profit = " + str(random_profit) + " | Last game optimal profit = " + str(optimal_profit))
        #print("     > Reward comparison to random model : " + str(avg/avg_random) + " | Comparison to optimal model = " + str(avg/avg_optimal))
        print("     > Profit comparison to random model : " + str(game_profit/random_profit) + " | Comparison to optimal model = " + str(game_profit/optimal_profit))

plt.plot(normal_game_profit,random_game_profit,  optimal_game_profit)
plt.savefig('figs/test.png')
