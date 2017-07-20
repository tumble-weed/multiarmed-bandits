import numpy as np

class Bandit:
    def __init__(self,true_mean,initial_est_mean = 0.):
        self.true_mean = true_mean
        self.est_mean = initial_est_mean
        self.N = 0
        pass
    def pull(self):
        return np.random.randn() + self.true_mean
        pass
    def update(self,sample):
        self.N += 1
        N = self.N
        self.est_mean = sample * (1./N) + self.est_mean*((N-1)*1./N)
        pass