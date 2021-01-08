import numpy as np
import random


class SuffixPrefixModel:
    def __init__(self, exemplars, probabilities, categories, number_of_categories, condition, beta):
        self.exemplars = exemplars
        self.probabilities = probabilities
        self.categories = categories
        self.number_of_categories = number_of_categories
        self.condition = condition
        self.beta = beta
        self.theta = np.zeros([self.number_of_categories, self.exemplars.shape[1]])

    def trial(self):
        """Choose stimuli for a single learning trial
        choice: category chosen randomly with probability p
        exemplar: exemplar from category choice
        label: label chosen from category at random
        """
        choice = np.random.choice(np.arange(0, self.exemplars.shape[0]), p=self.probabilities)
        exemplar = self.exemplars[choice, :]
        label = random.randint(0, self.number_of_categories - 1)
        return choice, label

    def outcome_present(self, choice, label):
        """Determine if category matches label

        Keyword arguments:
        choice -- randomly chosen category
        label -- randomly chosen label
        """
        if self.condition == 'suffix':
            outcome = 1. if label == self.categories[choice] else 0.
        else:
            outcome = self.exemplars[choice, :]
        return outcome

    def rescorla_wagner(self):
        """Run a single learning trial

        For a randomly chosen category and label, determine
        whether they match.

        Sum the weights for all the features associated with the label.
        Update weights via gradient descent.
        """
        choice, label = self.trial()
        outcome = self.outcome_present(choice, label)

        if self.condition == 'suffix':
            v_total = np.dot(self.theta[label, :], np.transpose(self.exemplars[choice, :]))
            self.theta[label, :] += self.beta * (outcome - v_total) * self.exemplars[choice, :]
        else:
            cat_of_choice = self.categories[choice]
            cues = np.zeros([self.number_of_categories])
            cues[cat_of_choice] = 1
            v_total = np.dot(cues, self.theta)
            self.theta[cat_of_choice, :] += self.beta * (outcome - v_total)