import numpy as np
import matplotlib.pyplot as plt

import testmodel
import plots
from simulations import Simulation


def get_experiment1_setup():
    """Generate a learning set 
    
    learning_set is a matrix with dimensions correspondiing to 
    the number of exemplars and the number of features
    """
    num_exemplars = 8
    num_features = 18
    
    learning_set = np.zeros([num_exemplars, num_features])
    
    # e.g. learning_set[0,0] = 1 means that exemplar 1 has feature 1 present
    learning_set[0,0] = 1
    learning_set[0,2] = 1
    learning_set[0,5] = 1
    learning_set[0,6] = 1
    learning_set[0,8] = 1
    learning_set[0,10] = 1
    learning_set[0,15] = 1

    learning_set[1,0] = 1
    learning_set[1,3] = 1
    learning_set[1,5] = 1
    learning_set[1,7] = 1
    learning_set[1,8] = 1
    learning_set[1,12] = 1
    learning_set[1,14] = 1

    learning_set[2,0] = 1
    learning_set[2,2] = 1
    learning_set[2,4] = 1
    learning_set[2,6] = 1
    learning_set[2,8] = 1
    learning_set[2,11] = 1
    learning_set[2,16] = 1

    learning_set[3,0] = 1
    learning_set[3,3] = 1
    learning_set[3,4] = 1
    learning_set[3,7] = 1
    learning_set[3,8] = 1
    learning_set[3,13] = 1
    learning_set[3,17] = 1

    learning_set[4,1] = 1
    learning_set[4,3] = 1
    learning_set[4,5] = 1
    learning_set[4,6] = 1
    learning_set[4,9] = 1
    learning_set[4,10] = 1
    learning_set[4,17] = 1

    learning_set[5,1] = 1
    learning_set[5,3] = 1
    learning_set[5,4] = 1
    learning_set[5,6] = 1
    learning_set[5,9] = 1
    learning_set[5,12] = 1
    learning_set[5,16] = 1

    learning_set[6,1] = 1
    learning_set[6,2] = 1
    learning_set[6,5] = 1
    learning_set[6,7] = 1
    learning_set[6,9] = 1
    learning_set[6,11] = 1
    learning_set[6,14] = 1

    learning_set[7,1] = 1
    learning_set[7,2] = 1
    learning_set[7,4] = 1
    learning_set[7,7] = 1
    learning_set[7,9] = 1
    learning_set[7,13] = 1
    learning_set[7,15] = 1

    # probability of each exemplar being chosen, divided by number of categories
    probabilities = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])/2.
    categories = np.array([0,0,0,0,1,1,1,1])
    
    number_of_categories = 2

    return(learning_set, probabilities, categories, number_of_categories)

learning_set, probabilities, categories, number_of_categories = get_experiment1_setup()
alpha = 0.1
beta = 0.1
n_trials = 2000

p = Simulation(learning_set, probabilities, categories, number_of_categories, "prefix", alpha, beta, n_trials)
prefix_weights = p.simulate()

s = Simulation(learning_set, probabilities, categories, number_of_categories, "suffix", alpha, beta, n_trials)
suffix_weights = s.simulate()

def exp1_test_raw(test_features, test_trials):
    """Test the model as follows:
    Do a single run of the model. 
    Do an additional 499 runs and average the weights across runs (testmodel.averaged_weights).
    For a test item, sum up the weights of its features for the correct and the incorerct affix (testmodel.test).

    Return:
        weights: for both conditions, the raw weight sum for the correct affix and the incorrect affix 
                 for given test item
    
    Keywrod arguments:
        test_features -- list of features of the test item
        test_trials -- list of trial numbers at which to test the model
    """
    prefix_averaged_weights = testmodel.averaged_weights(prefix_weights, learning_set, probabilities, categories, number_of_categories, "prefix", alpha, beta, n_trials)
    suffix_averaged_weights = testmodel.averaged_weights(suffix_weights, learning_set, probabilities, categories, number_of_categories, "suffix", alpha, beta, n_trials)

    # Prefix
    prefix_correct_weights = prefix_averaged_weights[:,0]
    prefix_incorrect_weights = prefix_averaged_weights[:,1]
    prefix_results = testmodel.sums(test_features, test_trials, prefix_correct_weights, prefix_incorrect_weights)

    # Suffix
    suffix_correct_weights = suffix_averaged_weights[:,0]
    suffix_incorrect_weights = suffix_averaged_weights[:,1]
    suffix_results = testmodel.sums(test_features, test_trials, suffix_correct_weights, suffix_incorrect_weights)
    
    sum_weights = np.concatenate([prefix_results, suffix_results], axis = 1)
    return sum_weights

def exp1_weights_fig():
    # Experiment 1 plot weights: Figure 3.2 in thesis
    plots.exp1_plt_weights(prefix_weights, 1)
    plots.exp1_plt_weights(suffix_weights, 2)
    plt.show()

def exp1_sum_fig():
    # Experiment 1 plot raw weights at test: Figure 3.3 panel A in thesis
    plots.plt_sum(w, -0.1, 4.1)
    plt.show()

def exp1_luces_fig():
    # Experiment 1 plot Luce's choice axiom results: Figure 3.3 panel B in thesis
    prob_correct = testmodel.luces_choice(w)
    plots.plt_luce(prob_correct)
    plt.show()

def exp1_raw_weights():
    test_features = [0, 2, 5, 6, 8, 10, 16]
    test_trials = [1999]
    w = exp1_test_raw(test_features, test_trials)
    return w

if __name__ == '__main__':
    exp1_weights_fig()
    exp1_sum_fig()
    exp1_luces_fig()