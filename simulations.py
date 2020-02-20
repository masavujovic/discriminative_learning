
from suffixprefix import SuffixPrefixModel
import numpy as np

class Simulation(SuffixPrefixModel):

    def __init__(self, exemplars, probabilities, categories, number_of_categories, condition, alpha, beta, n_trials):
        SuffixPrefixModel.__init__(self, exemplars, probabilities, categories, number_of_categories, condition, alpha, beta)
        self.n_trials = n_trials

    def simulate(self):
        n = 0
        weights = []
        while n < self.n_trials:
            weights_trial = self.rescorla_wagner()
            weights.append(self.theta.copy())
            n += 1
        weights = np.stack(weights)
        return weights


