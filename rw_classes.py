import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

class RescorlaWagnerNetwork:
	def __init__(self, exemplars, probabilities, categories, number_of_categories, network_type):
		self.exemplars = exemplars
		self.probabilities = probabilities
		self.categories = categories
		self.number_of_categories = number_of_categories
		self.network_type = network_type
		self.theta = np.zeros([self.number_of_categories, self.exemplars.shape[1]])

	def trial(self):
		"""Choose stimuli for single learning trial.
		Choose a category choice with probability p,
		and an exemplar from that category.
		Choose a random category label.
    	"""
		choice = np.random.choice(np.arange(0, self.exemplars.shape[0]), p=self.probabilities)
		exemplar = self.exemplars[choice, :]
		label = random.randint(0, self.number_of_categories - 1)
		return choice, label

	def outcome_present(self, choice, label):
		"""Determine if category matches label.
		
		Keyword arguments:
		choice -- randomly chosen category
		label -- randomly chosen label
		"""
		if self.network_type=='fl':
			outcome = 1. if label==self.categories[choice] else 0.
		else:
			outcome = self.exemplars[choice,:]
		return outcome

	def do_trial(self, alpha, beta):
		"""Run a single learning trial
		
		For a randomly chosen category and label, determine
		whether they match.
		
		Sum the weights for all the features associated with the label.
		Update weights via gradient descent.
		
		Keyword arguments:
		alpha -- hyperparameter (learning rate)
		beta -- hyperparameter (cue saliency)
		"""
		choice, label = self.trial()
		outcome = self.outcome_present(choice, label)
		
 		if self.network_type=='fl':
			v_total = np.dot(self.theta[label,:], np.transpose(self.exemplars[choice,:]))
			if type(beta) != float:
				self.theta[label,:] += alpha*beta[label,:]*(outcome - v_total)*self.exemplars[choice,:]
			else:
				self.theta[label,:] += alpha*beta*(outcome - v_total)*self.exemplars[choice,:]
		else:
			cat_of_choice = self.categories[choice]
			cues = np.zeros([self.number_of_categories])
			cues[cat_of_choice] = 1
			v_total = np.dot(cues, self.theta)
			if type(beta) != float:
				self.theta[cat_of_choice,:] += alpha*beta[cat_of_choice,:]*(outcome - v_total)
			else:
				self.theta[cat_of_choice,:] += alpha*beta*(outcome - v_total)

# Experiment 1

def experiment_1():
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

# Suffix simulation
def exp1_suffix(beta, n_max):
	learning_set, probabilities, categories, number_of_categories = experiment_1()
	exp1_suffix = RescorlaWagnerNetwork(learning_set, probabilities, categories, number_of_categories, 'fl')
	
	n = 0
	suffix_weights = []
	while n < n_max:
		exp1_suffix.do_trial(0.1, beta)
		suffix_weights.append(exp1_suffix.theta.copy())
		n += 1
	suffix_weights = np.stack(suffix_weights)
	return suffix_weights

# Prefix simulation
def exp1_prefix(beta, n_max):
	learning_set, probabilities, categories, number_of_categories = experiment_1()
	exp1_prefix = RescorlaWagnerNetwork(learning_set, probabilities, categories, number_of_categories, 'lf')

	n = 0
	prefix_weights = []
	while n < n_max:
		exp1_prefix.do_trial(0.1, beta)
		prefix_weights.append(exp1_prefix.theta.copy())
		n += 1
	prefix_weights = np.stack(prefix_weights)
	return prefix_weights

def exp1_plt(weights, j):
	ax = plt.subplot(1, 2, j)
	correct_cat_weights = weights[:, 0]
	x = np.arange(weights.shape[0])
	names_list = ["shape1", "shape2", "non-discrim sem", "non-discrim sem", \
		"non-discrim sem", "non-discrim sem", "non-discrim sem", \
			"non-discrim sem", "vowel1", "vowel2", "non-discrim phon", \
				"non-discrim phon", "non-discrim phon", "non-discrim phon", \
					"non-discrim phon", "non-discrim phon", "non-discrim phon", "non-discrim phon"]
	for k, w in enumerate(correct_cat_weights.transpose()):
		non_discrim_sem = [2, 3, 4, 5, 6, 7]
		non_discrim_phon = [10, 11, 12, 13, 14, 15, 16, 17]
		discrim_sem = [0, 1]
		if k in non_discrim_sem:
			plt.plot(x, w, label = names_list[k], color = "orange", lw=2)
		elif k in non_discrim_phon:
			plt.plot(x, w, label = names_list[k], color = "purple", lw=2)
		elif k == 0:
			plt.plot(x, w, label = names_list[k], color = "red", lw=2)
		elif k ==1:
			plt.plot(x, w, label = names_list[k], color = "green", lw=2)
	#plt.legend()
	if j == 1:
		plt.ylabel("Associative strength", fontsize = 16)
	plt.xlabel("Trial", fontsize = 16)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	handles, labels = plt.gca().get_legend_handles_labels()
	i = 1
	while i < len(labels):
		if labels[i] in labels[:i]:
			del(labels[i])
			del(handles[i])
		else:
			i += 1
	if j == 2:
		ax.legend(handles, labels, loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
		plt.subplots_adjust(right=0.8)
		ax.set_title("Suffix", fontsize = 16)
	else:
		ax.set_title("Prefix", fontsize = 16)
	plt.grid(True)
	plt.ylim(-0.25,1.01)
	

def exp1_averaged_weights(weights_init, condition):
	weights_init = weights_init
	i = 0
	while i < 500:
		if condition == "lf":
			new_weights = exp1_prefix(0.1, 2000)
		else:
			new_weights = exp1_suffix(0.1, 2000)
		weights_init += new_weights
		i += 1
	weights_avg = (weights_init)/500.0
	return weights_avg

def test(test_features, test_trials, correct, incorrect):
	all_correct = np.zeros(len(test_trials))
	all_incorrect = np.zeros(len(test_trials))

	for i in xrange(len(test_trials)):
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

def exp1_test():
	prefix_weights = exp1_prefix(0.1, 2000)
	suffix_weights = exp1_suffix(0.1, 2000)
	prefix_avg = exp1_averaged_weights(prefix_weights, "lf")
	suffix_avg = exp1_averaged_weights(suffix_weights, "fl")

	test_features = [0, 2, 5, 6, 8, 10, 15]
	test_trials = [49, 99, 499, 999, 1999]

	# Prefix
	prefix_correct_weights = prefix_avg[:,0]
	prefix_incorrect_weights = prefix_avg[:,1]
	prefix_results = test(test_features, test_trials, prefix_correct_weights, prefix_incorrect_weights)
	#print(prefix_results)

	# Suffix
	suffix_correct_weights = suffix_avg[:,0]
	suffix_incorrect_weights = suffix_avg[:,1]
	suffix_results = test(test_features, test_trials, suffix_correct_weights, suffix_incorrect_weights)

	exp1_plot_test(prefix_results, 1)
	exp1_plot_test(suffix_results, 2)
	plt.show()
	#return (prefix_results, suffix_results)
	return prefix_avg

def exp1_plot_test(weights, k):
	ax = plt.subplot(1, 2, k)
	ind = np.arange(5)
	width = 0.5
	r1 = np.arange(5)
	r2 = [x + width for x in r1]

	plt.bar(r1, weights[0], width = width, color = "grey", edgecolor = "black", label = "Correct affix", align = "edge")
	plt.bar(r2, weights[1], width = width, color = "white", edgecolor = "black", hatch = "\\", label = "Incorrect affix", align = "edge")
	plt.xticks([r + width for r in range(len(weights[0]))], ['50', '100', '500', '1000', '2000'])

	plt.xlabel("Trial", fontsize = 16)
	#plt.grid(True)
	if k == 2:
		ax.legend(loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
		plt.subplots_adjust(right=0.9)
		ax.set_title("Suffix", fontsize = 16)
	else:
		plt.ylabel("Sum of weights for exemplar X", fontsize = 16)
		ax.set_title("Prefix", fontsize = 16)


# Follow-up: Salience

def exp1_salience():
	num_exemplars = 8
	num_features = 18
	
	beta = np.zeros([num_exemplars, num_features])
	
	# e.g. learning_set[0,0] = 1 means that exemplar 1 has feature 1 present
	beta[:, 0] = 0.1
	beta[: ,1] = 0.1
	beta[:, 2] = 0.01
	beta[:, 3] = 0.01
	beta[:, 4] = 0.01
	beta[:, 5] = 0.01
	beta[:, 6] = 0.01
	beta[:, 7] = 0.01

	beta[:, 8] = 0.0001
	beta[:, 9] = 0.0001

	beta[:, 10] = 0.00001
	beta[:, 11] = 0.00001
	beta[:, 12] = 0.00001
	beta[:, 13] = 0.00001
	beta[:, 14] = 0.00001
	beta[:, 15] = 0.00001
	beta[:, 16] = 0.00001
	beta[:, 17] = 0.00001

	return beta


def plot_exp1_salience():
	beta = exp1_salience()
	prefix_weights = exp1_prefix(beta, 2000)
	prefix_exp1_plot = exp1_plt(prefix_weights, 1)

	suffix_weights = exp1_suffix(beta, 2000)
	suffix_exp1_plot = exp1_plt(suffix_weights, 2)
	plt.show()

exp1_test()




