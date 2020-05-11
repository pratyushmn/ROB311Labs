# part2.py: Project 4 Part 2 script
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


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
	"""
	policy_iteration method implements POLICY ITERATION MDP solver,
	shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
	for any given mdp environment.

	Inputs-
		agent: The MDP solving agent (mdp_agent)
		env:   The MDP environment (mdp_env)
		max_iter: Max iterations for the algorithm

	Outputs -
		policy: A list/array of actions for each state
				(Terminal states can have any random action)
		<agent>  Implicitly, you are populating the utlity matrix of
				the mdp agent. Do not return this function.
	"""
	np.random.seed(1) # TODO: Remove this

	# policy = np.random.randint(len(env.actions), size=(len(env.states), 1))
	policy = np.random.randint(len(env.actions), size=len(env.states))
	agent.utility = np.zeros([len(env.states), 1])

	## START: Student code
	unchanged = False # variable to check whether the policy has been updated since the last iteration
	iters = 0

	while not unchanged and iters < max_iter: # keep updating until the policy no longer changes, or we have reached the maximum number of iterations - whichever comes first
		unchanged = True # assume the policy will not be changed unless proven otherwise
		policy_evaluation(env, agent, policy, max_iter) # update the agent's utility estimates given the current policy

		# go through each state and see if the optimal policy for that state has changed given the new utility estimates for each state
		for state in env.states:
			all_action_vals = np.zeros(len(env.actions)) # create an array to store the expected utility of taking each action given the current state

			# compute the expected utility of each action given the current state
			for action in env.actions:
				for next_state in env.states:
					all_action_vals[action] += env.transition_model[state, next_state, action]*agent.utility[next_state, 0]

			# if the action that maximizes utility is no longer equal to the action given by the policy, update the policy and note the fact that there has been at least one change this iteration
			if np.argmax(all_action_vals) != policy[state]:
				policy[state] = np.argmax(all_action_vals)
				unchanged = False
		
		iters += 1

	## END: Student code

	return policy


def policy_evaluation(env: mdp_env, agent: mdp_agent, policy: np.ndarray, max_iter: int, max_change=0.001):
	"""
	a method that evaluates the utilities for each state under the input policy using the simplified Bellman equation
	"""
	max_delta = np.inf # initialize as high value
	iters = 0

	# keep updating utility estimates till they converge or till we've updated max_iter times
	while max_delta > max_change and iters < max_iter:
		max_delta = 0

		# go through each state and find its new utility value
		for state in env.states:
			new_val = 0

			# add the utility of all the next states as modified based on the probability of reaching the state, as well as the agent's gamma value (ie. how much it values future rewards)
			for next_state in env.states:
				new_val += agent.gamma*env.transition_model[state, next_state, policy[state]]*agent.utility[next_state]

			new_val += env.rewards[state] # finally, add the reward of being in that state

			# check if the new estimate is further from the current estimate than for any of the other states
			if abs(agent.utility[state] - new_val) > max_delta:
				max_delta = abs(agent.utility[state] - new_val)
			
			agent.utility[state] = new_val # update the utility estimate of that state with the new calculated value
		
		iters += 1
