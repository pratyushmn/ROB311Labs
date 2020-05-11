# mdp_cleaning_task.py: Project 4
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

###
# Imports
###

import numpy as np
from mdp_env import mdp_env

class cleaning_env(mdp_env):
    """
        cleaning_env class stores all the cleaning enviroment related functions and attributes.
        This is a grid world environment with two terminal states and 6 adjoining states.

        Attributes
        ------------------
            states_names:  List of names associated with the Unique state IDs
            action_names:  List of names associated with the Unique action IDs
            transition_model:  Matrix of size (SxSxA) specifying all of the
                               transition probabilities
    """

    def __init__(self, \
                 states = (0, 1, 2, 3, 4, 5), \
                 terminal = (0, 5), \
                 actions = (0, 1), \
                 rewards = (4, 1, 0, 0, 0, 5)):
        """
            INIT function
        """
        # Init the super class
        super().__init__(states, actions, terminal, rewards)

        # Init owner attributes
        self.state_names = ("0", "1", "2", "3", "4", "5")
        self.action_names = ("L", "R")
        self.transition_model = []

    def init_stochatic_model(self, get_transition_model):
        """
        init_stochatic_model method initializes the transition probability table for the the
        cleaning task.

        Inputs
        -------------
            get_transition_model: User Implemented method that defines the transition model
            for the cleaning environment
        """
        self.transition_model = get_transition_model(self)

    def print_env(self):
        print("\n------------- Environment -------------- ")
        print_string = ""
        for i in range(len(self.states)):
            print_string = print_string + "\t" + self.state_names[i] + ","
            if self.states[i] in self.terminal:
                print_string = print_string + "T"
        print(print_string)

    def print_transition_model(self):
        print("\n------------- Transition Model -------------- ")
        print("rows -> from-state | columns -> to-state")

        print_string = " \t0\t\t1\t\t2\t\t3\t\t4\t\t5\n"
        for i in range(len(self.states)):
            print_string += str(i)
            for j in range(len(self.states)):
                print_string += "\t" + str(self.transition_model[i, j, 0])
                for k in range(1, len(self.actions)):
                    print_string += ", " + str(self.transition_model[i, j, k])
            print_string += "\n"
        print(print_string)