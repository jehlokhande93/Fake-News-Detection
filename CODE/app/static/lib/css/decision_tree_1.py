from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        #self.tree = {}
		tree = {}
		tree['sv'] = None
		tree['sf'] = None 
		tree['left'] = {}
		tree['right'] = {}
		tree['leaf_val'] = None
		
        pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
		
		# for loop to iterate through each column, outside of this function
		# distinct values (set(new_words))
		# of distinct values len(set(new_words))
		# for categorical, 
		# also iterate through the split values for each attribute
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
		
		info_gain = 0
		best = {}
		best['IG'] = 0
		
		zipped = (zip(*X))
		sf_column = list(zipped[sf])
		sf_range = [min(sf_column), max(sf_column)]
		sv_list = (np.random.uniform(sf_range[0],sf_range[1],25))

		for sf in range(0, len(X)-1): # for all split attributes
		
			for val in sv_list: # for split values 
		
				[X_left, X_right, y_left, y_right] = partition_classes(X, y, sf, sv)
			
				cur_y = [y_left, y_right]
				prev_y = y
				
				info_gain = information_gain(prev_y, cur_y)
				if (info_gain > best['IG']):
					best['IG'] = info_gain
					best['sf'] = sf
					best['sv'] = val
				
		# after choosing the best split feature and the best split value,
		# partition the classes and set the .left = X_left and the .right = X_right 
		[X_left, X_right, y_left, y_right] = partition_classes(X, y, sf, sv)
		
		# The recursively call learn on X_left and X_right
		left = addNode(X_left, y_left)
		right = addNode(X_right, y_right)
				
		self.tree['left'] = left
		self.tree['right'] = right
		self.tree['sv'] = sv
		self.tree['sf'] = sf
		
		
		
        pass


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
		# Traversing the tree based on the recoed values and 
		traverse = self.tree
		
		classification = 0
        return classification
