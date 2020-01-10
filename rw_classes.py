import numpy as np
import random
import matplotlib.pyplot as plt
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
			self.theta[label,:] += alpha*beta*(outcome - v_total)*self.exemplars[choice,:]
		else:
			cat_of_choice = self.categories[choice]
			cues = np.zeros([self.number_of_categories])
			cues[cat_of_choice] = 1
			v_total = np.dot(cues, self.theta)
			self.theta[cat_of_choice,:] += alpha*beta*(outcome - v_total)

def simulation():
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

	categories = np.array([0,0,1,1])
	
	# probability of each exemplar being chosen, divided by number of categories
	probabilities = np.array([0.75, 0.25, 0.75, 0.25])/2.

	number_of_categories = 2

	# this is where we call the function (all the arguments explained above; the last string means suffix or prefix) 
	ntwrk = RescorlaWagnerNetwork(learning_set, probabilities, categories, number_of_categories, 'lf')
	
	n = 0

	with open('thetas.csv', 'w') as csvfile: 
		writer = csv.writer(csvfile, delimiter = ' ')
		while n < 5000:
			ntwrk.do_trial(0.1, 0.1)
			n += 1
			writer.writerow(ntwrk.theta[0])
	print ntwrk.theta

simulation()





