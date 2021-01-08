import numpy as np
import matplotlib.pyplot as plt

import testmodel
import plots
from simulations import Simulation


def get_experiment3_setup():
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


learning_set, probabilities, categories, number_of_categories = get_experiment3_setup()
beta = 0.01
n_trials = 7000

p = Simulation(learning_set, probabilities, categories, number_of_categories, "prefix", beta, n_trials)
prefix_weights = p.simulate()

s = Simulation(learning_set, probabilities, categories, number_of_categories, "suffix", beta, n_trials)
suffix_weights = s.simulate()


def exp3_test_raw(test_features, test_trials):
    """Test the model as follows:
    Do a single run of the model. 
    Do an additional 499 runs and average the weights across runs (testmodel.averaged_weights).
    For a test item, sum up the weights of its features for the correct and the incorerct affix (testmodel.test).

    Return:
        weights: for both conditions, the raw weight sum for the correct affix and the incorrect affix 
                 for given test item
    
    Arguments:
        test_features -- list of features of the test item
        test_trials -- list of trial numbers at which to test the model
    """
    prefix_averaged_weights = testmodel.averaged_weights(prefix_weights, learning_set, probabilities, categories,
                                                         number_of_categories, "prefix", beta, n_trials)
    suffix_averaged_weights = testmodel.averaged_weights(suffix_weights, learning_set, probabilities, categories,
                                                         number_of_categories, "suffix", beta, n_trials)

    # Prefix
    prefix_correct_weights = prefix_averaged_weights[:, 0]
    prefix_incorrect_weights = prefix_averaged_weights[:, 1]
    prefix_results = testmodel.sums(test_features, test_trials, prefix_correct_weights, prefix_incorrect_weights)

    # Suffix
    suffix_correct_weights = suffix_averaged_weights[:,0]
    suffix_incorrect_weights = suffix_averaged_weights[:,1]
    suffix_results = testmodel.sums(test_features, test_trials, suffix_correct_weights, suffix_incorrect_weights)
    
    sum_weights = np.concatenate([prefix_results, suffix_results], axis=1)
    return sum_weights


def exp3_weights_fig():
    # Experiment 3 plot weights: Figure 4.2 in thesis
    plots.exp3_plt_weights(prefix_weights, 1)
    plt.grid(b=None)
    plots.exp3_plt_weights(suffix_weights, 2)
    plt.grid(b=None)
    plt.tight_layout()
    plt.savefig('model_weights.png', quality=95, dpi=300)


def exp3_raw_weights_hf():
    test_features = [0, 2, 6]
    test_trials = [6999]
    w_hf = exp3_test_raw(test_features, test_trials)
    return w_hf


def exp3_raw_weights_lf():
    test_features = [1, 3, 6]
    test_trials = [6999]
    w_lf = exp3_test_raw(test_features, test_trials)
    return w_lf

def get_plots_thesis():
    exp3_weights_fig()

    w_hf = exp3_raw_weights_hf()
    w_lf = exp3_raw_weights_lf()

    # Experiment 3 plot sum of raw weights at test: Figure 4.3 panel A in thesis
    plots.plt_sum(w_hf, -0.5, 4)
    plt.show()

    # Experiment 3 plot Luce's choice axiom results: Figure 4.3 panel C in thesis
    prob_correct_hf = testmodel.luces_choice(w_hf)
    plots.plt_luce(prob_correct_hf)
    plt.show()

    softmax_hf = testmodel.softmax(w_hf)
    print(softmax_hf)

    # Experiment 3 plot sum of raw weights at test: Figure 4.3 panel B in thesis
    plots.plt_sum(w_lf, -0.1, 1.6)
    plt.show()

    # Experiment 3 plot Luce's choice axiom results: Figure 4.3 panel D in thesis
    prob_correct_lf = testmodel.luces_choice(w_lf)
    plots.plt_luce(prob_correct_lf)
    plt.show()


def get_plots_paper():
    exp3_weights_fig()
    plt.show()

    w_hf = exp3_raw_weights_hf()
    w_lf = exp3_raw_weights_lf()

    probs_exp3 = np.stack([testmodel.luces_choice(w_hf), testmodel.luces_choice(w_lf)])
    plots.plt_paper(probs_exp3)
    plt.show()

