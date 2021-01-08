import numpy as np
import matplotlib.pyplot as plt

import testmodel
import plots
from simulations import Simulation

# import experiment_1
import experiment_3


# def exp1_alpha():
#     num_exemplars = 8
#     num_features = 18
#
#     alpha = np.zeros([num_exemplars, num_features])
#
#     # e.g. learning_set[0,0] = 1 means that exemplar 1 has feature 1 present
#     alpha[:,0] = 0.1
#     alpha[:,1] = 0.1
#
#     alpha[: ,2] = 0.0
#     alpha[: ,3] = 0.0
#     alpha[: ,4] = 0.0
#     alpha[: ,5] = 0.0
#     alpha[: ,6] = 0.0
#     alpha[: ,7] = 0.0
#     alpha[: ,8] = 0.0
#     alpha[: ,9] = 0.0
#     alpha[: ,10] = 0.0
#     alpha[: ,11] = 0.0
#     alpha[: ,12] = 0.0
#     alpha[: ,13] = 0.0
#     alpha[: ,14] = 0.0
#     alpha[: ,15] = 0.0
#     alpha[: ,16] = 0.0
#     alpha[: ,17] = 0.0
#     return alpha


def get_experiment1_setup():
    """Generate a learning set

    learning_set is a matrix with dimensions correspondiing to
    the number of exemplars and the number of features
    """
    num_exemplars = 2
    num_features = 2

    learning_set = np.zeros([num_exemplars, num_features])

    # e.g. learning_set[0,0] = 1 means that exemplar 1 has feature 1 present
    learning_set[0, 0] = 1
    learning_set[1, 1] = 1

    # probability of each exemplar being chosen, divided by number of categories
    probabilities = np.array([1, 1]) / 2.
    categories = np.array([0, 1])

    number_of_categories = 2

    return learning_set, probabilities, categories, number_of_categories


def exp1_follow_up():
    # alpha = exp1_alpha()
    alpha = 0.1
    learning_set, probabilities, categories, number_of_categories = get_experiment1_setup()
    beta = 0.1
    n_trials = 2000
    
    p = Simulation(learning_set, probabilities, categories, number_of_categories, "prefix", alpha, beta, n_trials)
    prefix_weights = p.simulate()
    
    s = Simulation(learning_set, probabilities, categories, number_of_categories, "suffix", alpha, beta, n_trials)
    suffix_weights = s.simulate()
    
    # Figure 5.1 in thesis
    plots.exp1_plt_weights(prefix_weights, 1)
    plots.exp1_plt_weights(suffix_weights, 2)
    plt.show()

    # Experiment 1 plot raw weights at test: Figure 5.2 panel A in thesis
    # test_features = [0]
    # test_trials = [49, 99, 499, 999, 1499, 1999]
    # labels = ["50", "100", "500", "1000", "1500", "2000"]
    # prefix_averaged_weights = testmodel.averaged_weights(prefix_weights, learning_set, probabilities, categories, number_of_categories, "prefix", alpha, beta, n_trials)
    # suffix_averaged_weights = testmodel.averaged_weights(suffix_weights, learning_set, probabilities, categories, number_of_categories, "suffix", alpha, beta, n_trials)
    #
    # # Prefix
    # prefix_correct_weights = prefix_averaged_weights[:,0]
    # prefix_incorrect_weights = prefix_averaged_weights[:,1]
    # prefix_results = testmodel.sums(test_features, test_trials, prefix_correct_weights, prefix_incorrect_weights)
    #
    # # Suffix
    # suffix_correct_weights = suffix_averaged_weights[:,0]
    # suffix_incorrect_weights = suffix_averaged_weights[:,1]
    # suffix_results = testmodel.sums(test_features, test_trials, suffix_correct_weights, suffix_incorrect_weights)
    #
    # plots.plt_sum_intervals(prefix_results, 1, len(test_trials), labels)
    # plots.plt_sum_intervals(suffix_results, 2, len(test_trials), labels)
    # plt.show()
    #
    # # Plot Luce's choice axiom results: Figure 5.2 panel B in thesis
    # prefix_500 = prefix_results[:, 2]
    # suffix_500 = suffix_results[:, 2]
    # sum_weights = [prefix_500, suffix_500]
    # sum_weights = np.transpose(np.stack(sum_weights))
    #
    # prob_correct = testmodel.luces_choice(sum_weights)
    # plots.plt_luce(prob_correct)
    # plt.show()


def softmax_vs_luce():
    # Plot Luce's choice vs Softmax (Figure 5.3 in thesis)
    # w_exp1 = experiment_1.exp1_raw_weights()
    # prob_correct_exp1 = testmodel.luces_choice(w_exp1)
    # softmax_exp1 = testmodel.softmax(w_exp1)
    # probs_exp1 = np.stack([prob_correct_exp1, softmax_exp1])
    # plots.plt_softmax_luce(probs_exp1, 1)

    w_hf = experiment_3.exp3_raw_weights_hf()
    prob_correct_hf = testmodel.luces_choice(w_hf)
    softmax_hf = testmodel.softmax(w_hf)
    probs_exp3_hf = np.stack([prob_correct_hf, softmax_hf])

    w_lf = experiment_3.exp3_raw_weights_lf()
    prob_correct_lf = testmodel.luces_choice(w_lf)
    softmax_lf = testmodel.softmax(w_lf)
    probs_exp3_lf = np.stack([prob_correct_lf, softmax_lf])

    softmax_exp3 = np.stack([softmax_hf, softmax_lf])
    luces_exp3 = np.stack([prob_correct_hf, prob_correct_lf])
    plots.plt_softmax_luce(luces_exp3, 1)
    plots.plt_softmax_luce(softmax_exp3, 2)

    plt.show()


if __name__ == '__main__':
    exp1_follow_up()
