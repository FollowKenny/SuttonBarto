import numpy as np
import matplotlib.pyplot as plt
from kbandit import KBandit

class TestBed:
    def __init__(self, n=10, m=0, s=1, nsteps=1000, nrounds=2000) -> None:
        self.nrounds = nrounds
        self.nsteps = nsteps
        self.av = np.random.normal(m, s, n)
        self.test_results = {
            "rewards" : np.empty((nrounds, nsteps, 2)),
            "optimal_percent" : np.empty((nrounds, nsteps, 2))
        }
    
    def do_round(self) -> KBandit:
        bandit_sa = KBandit(self.av, stationary=False)
        bandit_sa.run_bandit(self.nsteps)
        bandit_cs = KBandit(self.av, stationary=False, method="cs")
        bandit_cs.run_bandit(self.nsteps)
        return bandit_sa, bandit_cs
    
    def do_test(self):
        for i in range(self.nrounds):
            bandit = self.do_round()
            for k in range(len(bandit)):
                self.test_results["rewards"][i, :, k] = \
                        bandit[k].exp_tracks["rewards"]
                self.test_results["optimal_percent"][i, :, k] = \
                    [100 * sum(bandit[k].exp_tracks["optimal"][:j+1])/(j+1)
                      for j in range(self.nsteps)]

    def compute_graphs(self):            
        plt.plot(range(self.nsteps), 
                 np.mean(self.test_results["rewards"], axis=0))
        plt.hlines(np.max(self.av), 0, self.nsteps)
        plt.legend(("sample average", "constant step size"))
        plt.xlabel("number of steps")
        plt.ylabel("average rewards on %d rounds" % self.nrounds)
        plt.savefig("rewards.png")
        plt.clf()
        plt.plot(range(self.nsteps),
                 np.mean(self.test_results["optimal_percent"], axis=0),
                 label=("steps", "percentage of optimal action"))
        plt.legend(("sample average", "constant step size"))
        plt.xlabel("number of steps")
        plt.ylabel("average percentage of optimal action on %d rounds" %
                   self.nrounds)
        plt.savefig("optimal.png")
        plt.close()

test = TestBed(nsteps=10000, nrounds=2000)
test.do_test()
test.compute_graphs()
