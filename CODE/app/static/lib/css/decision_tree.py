from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        #self.tree = {}
		self.tree = {}
		self.tree['sv'] = None
		self.tree['sf'] = None 
		self.tree['left'] = {}
		self.tree['right'] = {}
		self.tree['leaf_val'] = None


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
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
		
        # starting with the 1st sf
        def addNode(X,y): 
            n = DecisionTree()
            leaf_size = 3
            sameX = all(X[0] == item for item in X)
            sameY = all(y[0] == item for item in y)
			
            # If the number of points left is smaller than the leaf size
            if(len(X) <= leaf_size):
                if(len(X) == 0):
                    n.tree['leaf_val'] = int(round(np.mean(y),0))
                    return(n.tree)
                else:
                    n.tree['leaf_val'] = int(round(np.mean(y),0))
                    return(n.tree)
                    
            # if all X values are the same, or all Y values are the same        
            # if this is the case, stop bc otherwise there is divergence in 
            # classification logic
            
            elif(sameX or sameY):
                n.tree['leaf_val'] = int(round(np.mean(y),0))
                return(n.tree)
            
            # Otherwise, a node has not been reached
            else:
    
                sf = 0
                sv = X[0][sf]
                
                info_gain = 0
                best = {}
                best['IG'] = 0
                best['sf'] = sf
                best['sv'] = sv
                
                    
                [X_left, X_right, y_left, y_right] = partition_classes(X, y, sf, sv)
                zipped = (zip(*X))
                    
                
                for sf in range(0, len(X[0])):
                    sf_column = list(zipped[sf])
                    sf_range = [min(sf_column), max(sf_column)]
                    if(np.issubdtype(type(sf_column[0]), np.number)):
                        NUMERIC = True
                    else:
                        NUMERIC = False
                        
                    if(NUMERIC):
                        sv_list = (np.random.uniform(sf_range[0],sf_range[1],25))
                    else:
                        sv_list = list(set(sf_column))
                    
                    for val in sv_list:
                        if(NUMERIC):
                            val = round(val,2)
                        if(not((len(X_right)==0 and len(X_left)==0))):    
                            [X_left, X_right, y_left, y_right] = partition_classes(X, y, sf, val)
                        
                            cur_y = [y_left, y_right]
                            prev_y = y

                            info_gain = information_gain(prev_y, cur_y)
                            if (info_gain > best['IG']):
                                best['IG'] = info_gain
                                best['sf'] = sf
                                best['sv'] = val
                    print("IN sf FOR LOOP")

                [X_left, X_right, y_left, y_right] = partition_classes(X, y, best['sf'], best['sv'])
        		
                
                n.tree['sv'] = best['sv']
                n.tree['sf'] = best['sf']   
        		# The recursively call learn on X_left and X_right
                left = addNode(X_left, y_left)
                right = addNode(X_right, y_right)
        				
                n.tree['left'] = left
                n.tree['right'] = right

                return(n.tree)
        self.tree = addNode(X,y)
        return(self.tree)    


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
		# Traversing the tree based on the recoed values and 
        traverse = self.tree
        while(traverse['leaf_val'] == None):
            if(record[traverse['sf']] <= traverse['sv']):        
                traverse = traverse['left']
            else:
                traverse = traverse['right']
        output = traverse['leaf_val']
        return output
