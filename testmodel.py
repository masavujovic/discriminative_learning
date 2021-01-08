import numpy as np
from simulations import Simulation
import sys
np.set_printoptions(threshold=sys.maxsize)


def averaged_weights(weights_init, learning_set, probabilities, categories, number_of_categories, condition, beta,
                     n_trials):
    """Compute average weights over 500 runs of the model

    Arguments:
    weights_init -- weights after a single run of the model 
    learning set --  trials on which the model is trained (array)
    probabilities -- probabilities of each trial in learning_set (vector)
    categories --  all categories (e.g., [0, 1] - affix1 and affix2) (vector)
    number_of_categories -- the number of cateogries (affixes) (integer)
    condition -- "suffix" or "prefix" (string)
    alpha -- cue saliency (integer or array)
    beta -- learning rate (integer)
    n_trials -- number of trials (integer)
    """
    i = 0
    while i < 499:
        if condition == "prefix":
            network = Simulation(learning_set, probabilities, categories, number_of_categories, "prefix",
                                 beta, n_trials)
            new_weights = network.simulate()
        else:
            network = Simulation(learning_set, probabilities, categories, number_of_categories, "suffix",
                                 beta, n_trials)
            new_weights = network.simulate()
        weights_init += new_weights
        i += 1
    weights_avg = (weights_init)/499.0
    return weights_avg


def sums(test_features, test_trials, correct, incorrect):
    """Sum the correct and incorrect affix weights for a test item.

    Keyword arguments:
        test_features -- features corresponding to the test item (list)
        test_trials -- trials at which the model is to be tested (list)
        correct -- weights for all the features for the correct affix throughout learning
        incorrect -- weights for all the features for the incorrect affix throughout learning
    """
    all_correct = np.zeros(len(test_trials))
    all_incorrect = np.zeros(len(test_trials))

    for i in range(len(test_trials)):
        total_correct = 0
        total_incorrect = 0
        for j in test_features:
            total_correct += correct[test_trials[i], j]
            total_incorrect += incorrect[test_trials[i], j]
        all_correct[i] = total_correct
        all_incorrect[i] = total_incorrect

    all_total = [all_correct, all_incorrect]
    all_total = np.stack(all_total)
    return all_total


def luces_choice(sums):
    """Compute Luce's choice axiom.

    Return:
        prob_correct -- list of probabilities of choosing the correct affix in each condition

    Arguments:
        sums -- sums of weights for the correct and incorrect affix in each condition
    """
    prefix_correct = sums[0, 0]
    prefix_incorrect = sums[1, 0]
    suffix_correct = sums[0, 1]
    suffix_incorrect = sums[1, 1]
    prefix_prob_correct = prefix_correct/(prefix_correct + prefix_incorrect)
    suffix_prob_correct = suffix_correct/(suffix_correct + suffix_incorrect)
    prob_correct = [prefix_prob_correct, suffix_prob_correct]
    return prob_correct


def softmax(sums):
    prefix_correct = sums[0, 0]
    prefix_incorrect = sums[1, 0]
    suffix_correct = sums[0, 1]
    suffix_incorrect = sums[1, 1]
    prefix_prob_correct = np.exp(prefix_correct)/(np.exp(prefix_correct) + np.exp(prefix_incorrect))
    suffix_prob_correct = np.exp(suffix_correct)/(np.exp(suffix_correct) + np.exp(suffix_incorrect))
    prob_correct = [prefix_prob_correct, suffix_prob_correct]
    return prob_correct
