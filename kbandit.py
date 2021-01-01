import numpy as np


class KBandit:
    def __init__(self, av, eps=0.1, stationary=True,
                 method="sa", step_size=0.1) -> None:
        self.av = av
        self.naction = av.size
        self.av_estimates = np.zeros(self.naction)
        self.state_track = np.zeros(self.naction)
        self.eps = eps
        self.stationary = stationary
        self.method = method
        self.step_size = step_size
        self.exp_tracks = {
            "rewards": [],
            "optimal": []
        }
    
    def get_reward(self, action, s=1) -> np.array:
        '''Return reward randomly from a normal distrib centered on the action
        real action-value'''
        return np.random.normal(self.av[action], s)
    
    def modify_action_values(self):
        self.av = self.av + np.random.normal(0, 0.01, self.naction)
    
    def sample_average(self, action, reward):
        self.av_estimates[action] = self.av_estimates[action] + \
                (reward - self.av_estimates[action]) / self.state_track[action]
    
    def constant_step(self, action, reward):
        self.av_estimates[action] = self.av_estimates[action] + \
                (reward - self.av_estimates[action]) * self.step_size
    
    def step(self) -> None:
        behaviour = "exploit" if np.random.random() < 1-self.eps else "explore"
        
        if behaviour == "exploit":
            action = np.random.choice(np.flatnonzero(
                    self.av_estimates == self.av_estimates.max()))
        else:
            action = np.random.randint(self.naction)

        reward = self.get_reward(action)
        self.state_track[action] += 1
        if self.method == "sa":
            self.sample_average(action, reward)
        else:
            self.constant_step(action, reward)
        
        if not self.stationary:
            self.modify_action_values()

        # keep track of rewards and optimal pick
        self.exp_tracks["rewards"].append(reward)
        is_optimal = True if action == np.argmax(self.av) else False
        self.exp_tracks["optimal"].append(is_optimal)
        
    def run_bandit(self, nsteps=1000) -> None:
        for i in range(nsteps):
            self.step()