import numpy as np
import matplotlib.pyplot as plt

import testmodel
import plots
from simulations import Simulation


class Experiment3:
    def __init__(self, beta, n_trials):
        self.learning_set, self.probabilities, self.categories, self.number_of_categories = self.get_experiment3_setup()
        self.beta = beta
        self.n_trials = n_trials
        self.prefix_weights, self.suffix_weights = self.train_model()

    def get_experiment3_setup(self):
        """Generate a learning set

        learning_set is a matrix with dimensions corresponding to
        the number of exemplars and the number of features
        """
        num_exemplars = 16
        num_features = 40

        learning_set = np.zeros([num_exemplars, num_features])

        # With individual items Exp 1
        learning_set[0, 0] = 1
        learning_set[0, 2] = 1
        learning_set[0, 6] = 1
        learning_set[0, 8] = 1
        learning_set[0, 9] = 1

        learning_set[1, 0] = 1
        learning_set[1, 2] = 1
        learning_set[1, 6] = 1
        learning_set[1, 10] = 1
        learning_set[1, 11] = 1

        learning_set[2, 0] = 1
        learning_set[2, 2] = 1
        learning_set[2, 6] = 1
        learning_set[2, 12] = 1
        learning_set[2, 13] = 1

        learning_set[3, 0] = 1
        learning_set[3, 2] = 1
        learning_set[3, 6] = 1
        learning_set[3, 14] = 1
        learning_set[3, 15] = 1

        learning_set[4, 0] = 1
        learning_set[4, 2] = 1
        learning_set[4, 6] = 1
        learning_set[4, 16] = 1
        learning_set[4, 17] = 1

        learning_set[5, 0] = 1
        learning_set[5, 2] = 1
        learning_set[5, 6] = 1
        learning_set[5, 18] = 1
        learning_set[5, 19] = 1

        learning_set[6, 1] = 1
        learning_set[6, 3] = 1
        learning_set[6, 6] = 1
        learning_set[6, 20] = 1
        learning_set[6, 21] = 1

        learning_set[7, 1] = 1
        learning_set[7, 3] = 1
        learning_set[7, 6] = 1
        learning_set[7, 22] = 1
        learning_set[7, 23] = 1

        learning_set[8, 1] = 1
        learning_set[8, 4] = 1
        learning_set[8, 7] = 1
        learning_set[8, 24] = 1
        learning_set[8, 25] = 1

        learning_set[9, 1] = 1
        learning_set[9, 4] = 1
        learning_set[9, 7] = 1
        learning_set[9, 26] = 1
        learning_set[9, 27] = 1

        learning_set[10, 1] = 1
        learning_set[10, 4] = 1
        learning_set[10, 7] = 1
        learning_set[10, 28] = 1
        learning_set[10, 29] = 1

        learning_set[11, 1] = 1
        learning_set[11, 4] = 1
        learning_set[11, 7] = 1
        learning_set[11, 30] = 1
        learning_set[11, 31] = 1

        learning_set[12, 1] = 1
        learning_set[12, 4] = 1
        learning_set[12, 7] = 1
        learning_set[12, 32] = 1
        learning_set[12, 33] = 1

        learning_set[13, 1] = 1
        learning_set[13, 4] = 1
        learning_set[13, 7] = 1
        learning_set[13, 34] = 1
        learning_set[13, 35] = 1

        learning_set[14, 0] = 1
        learning_set[14, 5] = 1
        learning_set[14, 7] = 1
        learning_set[14, 36] = 1
        learning_set[14, 37] = 1

        learning_set[15, 0] = 1
        learning_set[15, 5] = 1
        learning_set[15, 7] = 1
        learning_set[15, 38] = 1
        learning_set[15, 39] = 1

        # probability of each exemplar being chosen, divided by number of categories
        probabilities = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                                 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]) / 2.

        categories = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        number_of_categories = 2

        return learning_set, probabilities, categories, number_of_categories

    def train_model(self):
        prefix_weights = Simulation(self.learning_set, self.probabilities, self.categories, self.number_of_categories,
                                    "prefix", self.beta, self.n_trials).simulate()
        suffix_weights = Simulation(self.learning_set, self.probabilities, self.categories, self.number_of_categories,
                                    "suffix", self.beta, self.n_trials).simulate()
        return prefix_weights, suffix_weights

    def test_model(self, test_features, test_trials):
        """Test the model as follows:
            Do a single run of the model.
            Do an additional 499 runs and average the weights across runs (testmodel.averaged_weights).
            For a test item, sum up the weights of its features for correct and incorerct affix (testmodel.test).

            Return:
                weights: for both conditions, the raw weight sum for the correct affix and the incorrect affix
                         for given test item

            Arguments:
                test_features -- list of features of the test item
                test_trials -- list of trial numbers at which to test the model
        """
        prefix_weights, suffix_weights = self.train_model()

        prefix_averaged_weights = testmodel\
            .averaged_weights(prefix_weights, self.learning_set, self.probabilities, self.categories,
                              self.number_of_categories, "prefix", self.beta, self.n_trials)
        suffix_averaged_weights = testmodel\
            .averaged_weights(suffix_weights, self.learning_set, self.probabilities, self.categories,
                              self.number_of_categories, "suffix", self.beta, self.n_trials)

        # Test prefix
        prefix_correct_weights = prefix_averaged_weights[:, 0]
        prefix_incorrect_weights = prefix_averaged_weights[:, 1]
        prefix_results = testmodel.sums(test_features, test_trials, prefix_correct_weights, prefix_incorrect_weights)

        # Test suffix
        suffix_correct_weights = suffix_averaged_weights[:, 0]
        suffix_incorrect_weights = suffix_averaged_weights[:, 1]
        suffix_results = testmodel.sums(test_features, test_trials, suffix_correct_weights, suffix_incorrect_weights)

        sum_weights = np.concatenate([prefix_results, suffix_results], axis=1)
        return sum_weights

    def plot_weights(self):
        """Plot weights over time from a single run.
        Figure 4.2 in thesis and Figure 2 in paper."""
        plots.exp3_plt_weights(self.prefix_weights, 1)
        plots.exp3_plt_weights(self.suffix_weights, 2)
        plt.show()


class Results(Experiment3):
    def __init__(self, beta, n_trials):
        super().__init__(beta, n_trials)

    def test_raw_weights(self):
        test_features_hf = [0, 2, 6]
        test_trials_hf = [6999]
        w_hf = self.test_model(test_features_hf, test_trials_hf)

        test_features_lf = [1, 3, 6]
        test_trials_lf = [6999]
        w_lf = self.test_model(test_features_lf, test_trials_lf)
        return w_hf, w_lf

    def get_results(self):
        w_hf, w_lf = self.test_raw_weights()
        prob_correct_hf = testmodel.luces_choice(w_hf)
        prob_correct_lf = testmodel.luces_choice(w_lf)
        return w_hf, w_lf, prob_correct_hf, prob_correct_lf

    def plot_thesis(self):
        w_hf, w_lf, prob_correct_hf, prob_correct_lf = self.get_results()

        plots.plt_sum(w_hf, -0.5, 4)  # Figure 4.3 panel A in thesis
        plt.show()

        plots.plt_sum(w_lf, -0.1, 1.6)  # Figure 4.3 panel B in thesis
        plt.show()

        plots.plt_luce(prob_correct_hf)  # Figure 4.3 panel C in thesis
        plt.show()

        plots.plt_luce(prob_correct_lf)  # Figure 4.3 panel D in thesis
        plt.show()

    def plot_paper(self):
        _, _, prob_correct_hf, prob_correct_lf = self.get_results()
        self.plot_weights()  # Figure 2 paper
        plt.show()

        p = np.stack([prob_correct_hf, prob_correct_lf])
        plots.plt_paper(p)  # Figure 3 paper
        plt.show()
