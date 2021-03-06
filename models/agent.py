
from gym.spaces import Discrete

class Agent: 

    def __init__(self, **kwargs):
         self.window_size = kwargs["window_size"]
         self.num_features = kwargs["num_features"]
         self.num_actions = kwargs["num_actions"]

         self.action_space = Discrete(self.num_actions)
         self.state_space = self.window_size * self.num_features
        
         
    def predict(self, state):
        return self.action_space.sample() # Should be overriden

    def learn(self, previous_state, action, next_state, reward, terminal):
        return # Random agent doesn't do anything
    
    def learn_episode(self, episode_num, **kwargs):
        return

    def print_infos(self):
        print("Random agent")
    
    def save_model(self, **kwargs):
        print("Model not saved, method not correctly implemented")

    def load_model(self, **kwargs):
        print("Model not loaded, method not correctly implemented")