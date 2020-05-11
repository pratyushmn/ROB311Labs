import random
import gym
import math
import numpy as np

"""
Agent Description: Simple implementation of Monte-Carlo Policy Gradient algorithm (REINFORCE)

TODO: Insert a 10-15 line description (80 characters wide) of the algorithm
that you implemented in your agent. If you used a reference paper or book,
you may add additional lines to cite this reference.

This agent uses the Monte-Carlo Policy Gradient (REINFORCE) algorithm to learn.
The "policy" is a mapping from input states to actions (more accurately, probability
distributions over actions). Here, the policy is implemented with a matrix of
weights which is multiplied with the input state vector, and then converted into
a probability distribution using the softmax function. The "gradient" aspect refers
to the use of gradient descent to update the weights after each episode; modifying 
the policy so that it improves after every episode. Updates are made by calculating
the gradients for all actions made under a specific policy during an episode,
discounting the rewards gathered by each action in the episode (shown in class),
and then incrementing the weights by multiplying those two values for each action
(as well as the learning rate). In order to prevent exploding gradients, I "clipped"
the max/min update values to a range in between +/- epsilon, and this greatly
improved performance. Another change I made to improve performance was to
modify the reward function - as the cartpole environment aims to minimize the
pole angle without allowing the cart to move offscreen, I constructed a new function
that directly incentivizes the agent to keep the pole upright and not venture far
off to either side of the screen.

References: 
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce
https://youtu.be/S_gwYj1Q-44 
https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d (for calculating gradient)

"""

class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        #----- TODO: Add your code here. -----

        # Store observation space and action space, in addition to values of hyperparameters such as epsilon, gamma, and learning rae (based on trial and error)
        self.observation_space = 4
        self.action_space = 2
        self.gamma = 0.99
        self.epsilon = 0.1
        self.lr = 3e-4

        # important variables
        self.weights = np.random.rand(4, 2) # weights to be updated via gradient descent
        self.grads = [] # gradients for each action made in the episode
        self.reward_memory = [] # received rewards for each action made in the episode (before discounting)

    def policy(self, state):
        """Maps an input state to a probability distribution over possible actions using matrix multiplication and softmax"""
        x = state.dot(self.weights)
        return self.softmax(x)

    def softmax(self, x):
        """Returns the softmax of a vector x"""
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator
        return softmax

    def compute_grad(self, pi_s, a):
        """Compute gradient of pi(s, a)."""
        j = pi_s.reshape(-1,1)
        jacobian = np.diagflat(j) - np.dot(j, j.T)
        
        return jacobian[a, :] # for policy gradient, I need only the gradient for the chosen action, not for all actions
    
    def action(self, state):
        """Choose an action from set of possible actions."""
        #----- TODO: Add your code here. -----
        
        # predict an action given the current policy
        probs = self.policy(state)
        action = np.random.choice(self.action_space, p=probs)
        
        # compute gradient
        dpi_s_a = self.compute_grad(probs, action)
        dlog_pi = dpi_s_a/probs[action]
        grad = np.transpose(state[np.newaxis, :]).dot(dlog_pi[np.newaxis,:])
        
        # save gradient for later use
        self.grads.append(grad)

        return action

    def discount_rewards(self):
        """Discount the rewards for this episode using gamma."""
        discounted_rewards = []

        # the discounted reward for each action (G_t) is calculated by looking at all future rewards and discounting them based on how many timesteps away
        for t in range(len(self.reward_memory)):
            G_t = 0
            discount = 1
            
            # calculating the value of each individual G_t according to 
            for j in range(t, len(self.reward_memory)):
                G_t += self.reward_memory[j]*discount
                discount *= self.gamma

            discounted_rewards.append(G_t)

        return discounted_rewards

    def reset(self):
        """Reset the agent, if desired."""
        #----- TODO: Add your code here. -----

        # before carrying out gradient descent, the rewards need to be properly discounted
        rewards = self.discount_rewards() 

        #  gradient descent
        for i in range(len(self.grads)):
            update = self.lr*rewards[i]*self.grads[i] 
            update = np.clip(update, -1*self.epsilon, self.epsilon) #  inspired by gradient clipping/proximal policy optimization; intended to prevent large changes to the policy that may cause it to overshoot and become worse
            self.weights += update
        
        # clearing memory of the past episode
        self.grads.clear()
        self.reward_memory.clear()

    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        #----- TODO: Add your code here. -----

        # instead of using the given reward, I created a new reward function based on the pole angle
        angle = state_next[2]
        position = state_next[0]
        new_reward = 1/(abs(angle) + np.nextafter(0, 1)) # to ensure that I don't divide by 0, the nextafter function gives the smallest possible np number
        
        #  penalize moving too far towards the edge of the screen
        if abs(position) > 2.2:
            new_reward/= abs(position)

        new_reward *= reward
        
        #  save the current reward for the action (will be updated later)
        self.reward_memory.append(new_reward)
