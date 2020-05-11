# part1_2.py: Project 4 Part 1 script
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
from mdp_agent import mdp_agent

from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
	"""
	value_iteration method implements VALUE ITERATION MDP solver,
	shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
	for any given mdp environment.

	Inputs
	---------------
		agent: The MDP solving agent (mdp_agent)
		env:   The MDP environment (mdp_env)
		eps:   Max error allowed in the utility of a state
		max_iter: Max iterations for the algorithm

	Outputs
	---------------
		policy: A list/array of actions for each state
				(Terminal states can have any random action)
		<agent>  Implicitly, you are populating the utlity matrix of
				the mdp agent. Do not return this function.
	"""
	policy = np.empty_like(env.states)
	agent.utility = np.zeros([len(env.states), 1])

	## START: Student code
	iters = 0
	max_delta = np.inf

	while iters < max_iter and (max_delta >= (eps*(1 - agent.gamma))/agent.gamma): # keep updating until the "error" between updated values and old values is below the cutoff, or until max_iter
		max_delta = 0

		# need to update the utility of each state, as well as the policy for each state
		for state in env.states:
			currVal = agent.utility[state, 0] # current/old utility of the state

			action_utilities = np.zeros(len(env.actions)) # utility of taking each individual action

			# finding the utility of each possible action by updating the action_utilities array
			for action in env.actions:
				for next_state in env.states:
					action_utilities[action] += env.transition_model[state, next_state, action]*agent.utility[next_state, 0]
			
			newVal = env.rewards[state] + agent.gamma*np.max(action_utilities) # calculating the new utility value for this state 

			if abs(newVal - currVal) > max_delta:
				max_delta = abs(newVal - currVal)

			agent.utility[state, 0] =  newVal # updating the utility function of the agent
			policy[state] = np.argmax(action_utilities) # the policy is to take the action which maximizes the expected utility of the next state

		iters += 1

	## END Student code
	return policy
