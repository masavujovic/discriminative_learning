import numpy as np
import matplotlib.pyplot as plt

import testmodel
import plots
from simulations import Simulation


def get_experiment3_setup():
    """Generate a learning set 
    
    learning_set is a matrix with dimensions correspondiing to 
    the number of exemplars and the number of features
    """
    num_exemplars = 4
    num_features = 6
    
    learning_set = np.zeros([num_exemplars, num_features])
    
    # e.g. learning_set[0,0] = 1 means that exemplar 1 has feature 1 present
    learning_set[0,0] = 1
    learning_set[0,2] = 1
    
    learning_set[1,1] = 1
    learning_set[1,3] = 1

    learning_set[2,1] = 1
    learning_set[2,4] = 1
    
    learning_set[3,0] = 1
    learning_set[3,5] = 1
    
    # probability of each exemplar being chosen, divided by number of categories

    probabilities = np.array([0.75, 0.25, 0.75, 0.25])/2.
    categories = np.array([0,0,1,1])
    
    number_of_categories = 2

    return(learning_set, probabilities, categories, number_of_categories)

learning_set, probabilities, categories, number_of_categories = get_experiment3_setup()
alpha = 0.1
beta = 0.1
n_trials = 7000

p = Simulation(learning_set, probabilities, categories, number_of_categories, "prefix", alpha, beta, n_trials)
prefix_weights = p.simulate()

s = Simulation(learning_set, probabilities, categories, number_of_categories, "suffix", alpha, beta, n_trials)
suffix_weights = s.simulate()

def exp3_test_raw(test_features, test_trials):
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

def exp3_weights_fig():
    # Experiment 3 plot weights: Figure 4.2 in thesis
    plots.exp3_plt_weights(prefix_weights, 1)
    plots.exp3_plt_weights(suffix_weights, 2)
    plt.show()

def exp3_raw_weights_hf():
    test_features = [0, 2]
    test_trials = [6999]
    w_hf = exp3_test_raw(test_features, test_trials)
    return w_hf

def exp3_raw_weights_lf():
    test_features = [1, 3]
    test_trials = [6999]
    w_lf = exp3_test_raw(test_features, test_trials)
    return w_lf

if __name__ == '__main__':
    exp3_weights_fig()

    w_hf = exp3_raw_weights_hf()
    w_lf = exp3_raw_weights_lf()

    # Experiment 3 plot sum of raw weights at test: Figure 4.3 panel A in thesis
    plots.plt_sum(w_hf, -0.1, 1.6)
    plt.show()

    # Experiment 3 plot Luce's choice axiom results: Figure 4.3 panel C in thesis
    prob_correct_hf = testmodel.luces_choice(w_hf)
    plots.plt_luce(prob_correct_hf)
    plt.show()

    # Experiment 3 plot sum of raw weights at test: Figure 4.3 panel B in thesis
    plots.plt_sum(w_lf, -0.1, 1.6)
    plt.show()

    # Experiment 3 plot Luce's choice axiom results: Figure 4.3 panel D in thesis
    prob_correct_lf = testmodel.luces_choice(w_lf)
    plots.plt_luce(prob_correct_lf)
    plt.show()