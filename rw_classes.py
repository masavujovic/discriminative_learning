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
		# on a single trial, choose a category with probability p (HF 75%, LF 25% for example)
		choice = np.random.choice(np.arange(0, self.exemplars.shape[0]), p=self.probabilities)
		# then choose an exemplar from that category
		exemplar = self.exemplars[choice, :]
		# then choose a category label at random
		label = random.randint(0, self.number_of_categories - 1)
		return choice, label

	def outcome_present(self, choice, label):
		# if the randomly chosen label (argument "label") matches the actual category 
		# of the exemplar (argument "exemplar") --> outcome = 1 (lambda)
		# this means outcome was present (remember that at each trial weights get updated for all outcomes, present and otherwise)
		if self.network_type=='fl':
			outcome = 1. if label==self.categories[choice] else 0.
		else:
			outcome = self.exemplars[choice,:]
		return outcome

	def do_trial(self, alpha, beta):
		# randomly chose a category and a label
		choice, label = self.trial()
		# determine if they match (if outcome was "present")
		outcome = self.outcome_present(choice, label)
		
		# if netowrk is fl (feature-label) -- suffixing
 		if self.network_type=='fl':
 			# v_total: sum all the weights for all the features for a given label (dot product)
			v_total = np.dot(self.theta[label,:], np.transpose(self.exemplars[choice,:]))
			# update theta (weight vector) = multiply alpha, beta, lambda (outcome), v_total, and the features vector
			# which is self.exemplars[choice,:] -- each exemplar is a vector of 0 and 1 (0 - exemplar does not have that feature; 1 - exemplar has that feature)
			self.theta[label,:] += alpha*beta*(outcome - v_total)*self.exemplars[choice,:]
		else:
			# same but swapped around (prefix)
			cat_of_choice = self.categories[choice]
			cues = np.zeros([self.number_of_categories])
			cues[cat_of_choice] = 1
			v_total = np.dot(cues, self.theta)
			self.theta[cat_of_choice,:] += alpha*beta*(outcome - v_total)

def simulation():
	num_exemplars = 4
	num_features = 6
	# cat is a matrix with dimensions corresponding to number of exemplars
	# and number of features 
	cat = np.zeros([num_exemplars, num_features])
	
	# e.g. cat[0,0] means that exemplar 1 has feature 1 present
	cat[0,0] = 1
	cat[0,2] = 1
	cat[1,1] = 1
	cat[1,3] = 1
	cat[2,1] = 1
	cat[2,4] = 1
	cat[3,0] = 1
	cat[3,5] = 1

	# as many as there are examplars 
	categories = np.array([0,0,1,1])
	
	# probs of each exemplar appearing (type frequency manipulation) divided by number of categories
	probabilities = np.array([0.75, 0.25, 0.75, 0.25])/2.

	number_of_categories = 2

	# this is where we call the function (all the arguments explained above; the last string means suffix or prefix) 
	ntwrk = RescorlaWagnerNetwork(cat, probabilities, categories, number_of_categories, 'lf')
	
	n = 0

	with open('thetas.csv', 'w') as csvfile: 
		writer = csv.writer(csvfile, delimiter = ' ')
		# n total number of trials
		while n < 5000:
			ntwrk.do_trial(0.1, 0.1)
			n += 1
			writer.writerow(ntwrk.theta[0])
	print ntwrk.theta

simulation()





