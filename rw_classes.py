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
def exp1_suffix(experiment, beta, n_max):
	if experiment == "experiment1":
		learning_set, probabilities, categories, number_of_categories = experiment_1()
	else:
		learning_set, probabilities, categories, number_of_categories = experiment_2()
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
def exp1_prefix(experiment, beta, n_max):
	if experiment == "experiment1":
		learning_set, probabilities, categories, number_of_categories = experiment_1()
	else:
		learning_set, probabilities, categories, number_of_categories = experiment_2()
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
	

def exp1_averaged_weights(experiment, weights_init, condition, salience, n_max):
	weights_init = weights_init
	i = 0
	while i < 500:
		if condition == "lf":
			new_weights = exp1_prefix(experiment, salience, n_max)
		else:
			new_weights = exp1_suffix(experiment, salience, n_max)
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

def exp1_test(experiment, salience, n_trials, test_features, test_trials):
	prefix_weights = exp1_prefix(experiment, salience, n_trials)
	suffix_weights = exp1_suffix(experiment, salience, n_trials)
	prefix_avg = exp1_averaged_weights(experiment, prefix_weights, "lf", salience, n_trials)
	suffix_avg = exp1_averaged_weights(experiment, suffix_weights, "fl", salience, n_trials)

	#test_features = [0, 2, 5, 6, 8, 10, 15]
	#test_trials = [49, 99, 499, 999, 1999]

	# Prefix
	prefix_correct_weights = prefix_avg[:,0]
	prefix_incorrect_weights = prefix_avg[:,1]
	prefix_results = test(test_features, test_trials, prefix_correct_weights, prefix_incorrect_weights)

	# Suffix
	suffix_correct_weights = suffix_avg[:,0]
	suffix_incorrect_weights = suffix_avg[:,1]
	suffix_results = test(test_features, test_trials, suffix_correct_weights, suffix_incorrect_weights)

	trials = len(test_trials)

	weights = np.concatenate([prefix_results, suffix_results], axis=1)
	#exp1_plot_test(prefix_results, 1, trials)
	#exp1_plot_test(suffix_results, 2, trials)
	plot_masas_plot(weights)
	plt.show()
	return prefix_avg

def plot_masas_plot(weights):
	print(weights)
	width = 0.3
	r1 = np.arange(2)
	r2 = [x + width for x in r1]

	plt.bar(r1, weights[0], width = width, color = "grey", edgecolor = "black", label = "Correct affix")
	plt.bar(r2, weights[1], width = width, color = "white", edgecolor = "black", hatch = "\\", label = "Incorrect affix",)
	plt.xticks([r + width for r in range(len(weights[0]))], ["Prefix", "Suffix"])
	plt.tick_params(axis = "x", which = "major", labelsize = 16)
	#plt.grid(True)
	plt.legend(loc = 1, bbox_to_anchor = (1, 1), prop = {'size':14}, ncol=1)
	plt.subplots_adjust(right=0.8)
	plt.ylabel("Sum of raw weights for exemplar X", fontsize = 16)


def plot_axiom(probs):
	width = 0.3

	plt.bar(0, probs[0], width = width, color = "grey", edgecolor = "black")
	plt.bar(0.5, probs[1], width = width, color = "grey", edgecolor = "black")

	plt.xticks([0.15, 0.65], ["Prefix", "Suffix"])
	plt.tick_params(axis = "x", which = "major", labelsize = 16)

	plt.ylabel("Probability of correct affix", fontsize = 16)
	plt.axhline(y = 0.5, color = "black", linestyle = "dashed")
	plt.show()

def exp1_plot_test(weights, k, trials):
	ax = plt.subplot(1, 2, k)
	width = 0.3
	r1 = np.arange(trials)
	r2 = [x + width for x in r1]

	plt.bar(r1, weights[0], width = width, color = "grey", edgecolor = "black", label = "Correct affix")
	plt.bar(r2, weights[1], width = width, color = "white", edgecolor = "black", hatch = "\\", label = "Incorrect affix",)
	plt.xticks([r + width for r in range(len(weights[0]))])

	plt.xlabel("Trial", fontsize = 16)
	#plt.grid(True)
	if k == 2:
		ax.legend(loc = "upper left", bbox_to_anchor = (1, 0.5), prop = {'size':12}, ncol=1)
		plt.subplots_adjust(right=0.8)
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
	beta[:, 0] = 0.2
	beta[: ,1] = 0.2
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



# Experiment 2
def experiment_2():
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

def exp2_plt(weights, j):
	ax = plt.subplot(1, 2, j)
	correct_cat_weights = weights[:, 0]
	x = np.arange(weights.shape[0])
	names_list = ["shape1", "shape2", "HF discrim", "LF discrim", \
		"HF discrim opposite", "LF discrim opposite"]
	for k, w in enumerate(correct_cat_weights.transpose()):
		plt.plot(x, w, label = names_list[k], lw=2)
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
	plt.ylim(-0.5, 1)


if __name__ == '__main__':
	# prefix_weights = exp1_prefix("experiment2", 0.1, 7000)
	# suffix_weights = exp1_suffix("experiment2", 0.1, 7000)

	# #exp2_plt(prefix_weights, 1)
	# #exp2_plt(suffix_weights, 2)
	# #plt.show()

	test_features = [0, 2, 5, 6, 8, 10, 16]
	test_trials = [1999]
	#exp1_test("experiment1", 0.1, 2000, test_features, test_trials)
	#plot_axiom([0.667, 1])
	a = [0.75, 0.45]
	b = [0.96 -0.042]

	print(np.exp(a) / np.sum(np.exp(a)))
	print(np.exp(b) / np.sum(np.exp(b)))