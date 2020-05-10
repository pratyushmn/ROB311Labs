import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS


def dt_entropy(goal, examples):
    """
    Compute entropy over discrete random varialbe for decision trees.
    Utility function to compute the entropy (wich is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """
    # INSERT YOUR CODE HERE.
    entropy = 0.0
    
    num_classes = len(goal[1])
    counts = np.zeros(num_classes)
    total_examples = (examples.shape)[0]
    last_column = (examples.shape)[1] - 1

    for row in examples:
        counts[row[last_column]] += 1

    for count in counts:
        if count == 0:
            pass
        else:
            entropy -= (count/total_examples)*(np.log2(count/total_examples))

    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    return entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    cond_entropy = 0.0

    num_classes = len(goal[1])
    num_vals = len(attribute[1])

    joint_counts = np.zeros((num_vals, num_classes))
    x_counts = np.zeros(num_vals)
    total_examples = (examples.shape)[0]
    last_column = (examples.shape)[1] - 1

    for row in examples:
        x_counts[row[col_idx]] += 1
        joint_counts[row[col_idx], row[last_column]] += 1

    x_probs = x_counts/total_examples
    joint_probs = joint_counts/total_examples

    for n in range(num_vals):
        tot = 0.0
        for k in range(num_classes):
            if joint_probs[n, k] == 0:
                pass
            elif x_probs[n] == 0: 
                tot += joint_probs[n, k]*np.log2(joint_probs[n, k])
            else:
                tot += joint_probs[n, k]*(np.log2(joint_probs[n, k]) - np.log2(x_probs[n]))
            
        cond_entropy -= tot

    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """
    # INSERT YOUR CODE HERE.
    info_gain = 0.0

    entropy = dt_entropy(goal, examples)
    cond_entropy = dt_cond_entropy(attribute, col_idx, goal, examples)

    info_gain = entropy - cond_entropy

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples):
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    intrinsic_info = 0.0

    d = len(attribute[1])
    num_ex = examples.shape[0]
    count_vals = np.zeros(d) # number of examples with each attribute value

    for row in examples:
        count_vals[row[col_idx]] += 1

    for k in range(d):
        if count_vals[k] == 0:
            pass
        else:
            intrinsic_info -= (count_vals[k]/num_ex)*np.log2(count_vals[k]/num_ex)

    return intrinsic_info


def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Avoid NaN examples by treating 0.0/0.0 = 0.0
    gain_ratio = 0.0

    if dt_intrinsic_info(attribute, col_idx, examples) == 0:
        return gain_ratio
    else:
        gain_ratio = dt_info_gain(attribute, col_idx, goal, examples)/dt_intrinsic_info(attribute, col_idx, examples)

    return gain_ratio


def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.

    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """
    # YOUR CODE GOES HERE
    node = None
    
    # 1. Do any examples reach this point?
    if examples.size == 0:
        node = TreeNode(parent, goal, parent.examples, True, plurality_value(goal, parent.examples))
    else:
        # 2. Or do all examples have the same class/label? If so, we're done!
        initial_class = examples[0][-1]
        isLeaf = True
        for row in examples:
            if row[-1] != initial_class:
                isLeaf = False

        if isLeaf:
            node = TreeNode(parent, goal, examples, True, initial_class)
        else:
            # 3. No attributes left? Choose the majority class/label.
            if len(attributes) == 0:
                node = TreeNode(parent, goal, examples, True, plurality_value(goal, examples))

            # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!
            else:
                # Best score?
                best_score = -1
                best_index = -1
                for a in range(len(attributes)):
                    new_score = score_fun(attributes[a], a, goal, examples)
                    if new_score > best_score:
                        best_score = new_score
                        best_index = a

                # NOTE: to pass the Autolab tests, when breaking ties you should always select the attribute with the smallest (i.e.
                # leftmost) column index!

                # Create a new internal node using the best attribute, something like:
                # node = TreeNode(parent, attributes[best_index], examples, False, 0)

                node = TreeNode(parent, attributes[best_index], examples, False, -1) # label is -1 since it doesn't matter

                # Now, recurse down each branch (operating on a subset of examples below).
                # You should append to node.branches in this recursion

                new_atts = attributes.copy()
                del new_atts[best_index] # new_atts has all remaining attributes
                

                for i in range(len(attributes[best_index][1])): # go through each possible value of the attribute
                    remove_examples = [] # a list that holds all the rows of the examples array to remove (since they have the current value of the attribute)
                    
                    for j in range(examples.shape[0]):
                        if examples[j][best_index] != i:
                            remove_examples.append(j)

                    ex = np.delete(examples, remove_examples, 0) # an array to hold the remaining examples
                    new_examples = np.delete(ex, best_index, 1)

                    node.branches.append(learn_decision_tree(node, new_atts, goal, new_examples, score_fun)) 

    return node


def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """
    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root

if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])

    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(None, attributes, goal, examples, dt_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
