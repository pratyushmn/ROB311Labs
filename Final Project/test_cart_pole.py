import gym
import numpy as np

from rl_agent import CartPoleAgent

# Test harness used to evaluate a CartPoleAgent.
SUCCESS_REWARD = 265
SUCCESS_STREAK = 100
MAX_EPISODES = 1000
MAX_STEPS = 5000

def run_cart_pole():
    """
    Run instances of cart-pole gym and tally scores.
    
    The function runs up to 1,000 episodes and returns when the 'success' 
    criterion for the OpenAI cart-pole task (v0) is met: an average reward
    of 195 or more over 100 consective episodes.
    """
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_STEPS

    # Create an instance of the agent.
    cp_agent = CartPoleAgent(env.observation_space, env.action_space)
    avg_reward, win_streak = (0, 0)
    rewards = []

    for episode in range(MAX_EPISODES):
        state = env.reset()

        # Reset the agent, if desired.
        cp_agent.reset()
        episode_reward = 0
    
        # The total number of steps is limited (to avoid long-running loops).
        for steps in range(MAX_STEPS):
            # env.render()

            # Ask the agent for the next action and step accordingly.
            action = cp_agent.action(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward

            # Update any information inside the agent, if desired.
            cp_agent.update(state, action, reward, state_next, terminal)
            episode_reward += reward # Total reward for this episode.
            state = state_next

            if terminal:
                # Update average reward.
                if episode < SUCCESS_STREAK:
                    rewards.append(episode_reward)
                    avg_reward = float(sum(rewards))/(episode + 1)
                else:
                    # Last set of epsiodes only (moving average)...
                    rewards.append(episode_reward)
                    rewards.pop(0)
                    avg_reward = float(sum(rewards))/SUCCESS_STREAK

                # Print some stats.
                print("Episode: " + str(episode) + \
                        ", Reward: " + str(episode_reward) + \
                        ", Avg. Reward: " + str(avg_reward) + \
                        ", Current Win Streak: " + str(win_streak))

                # Is the agent on a winning streak?
                if episode_reward >= SUCCESS_REWARD:
                    win_streak += 1
                else:
                    win_streak = 0
                break

        #print(rewards)

        # Has the agent succeeded?
        if win_streak == SUCCESS_STREAK and avg_reward >= SUCCESS_REWARD:
            # return episode + 1, avg_reward 
            return (1, episode + 1)

        if episode > 900 and win_streak == 0:
            return (0, 1000)

    # Worst case, agent did not meet criterion, so bail out.
    # return episode + 1, avg_reward
    return (0, episode + 1)

if __name__ == "__main__":
    num_successes = 0
    episode_solve = []
    for i in range(20):
        solved, num_eps = run_cart_pole()
        num_successes += solved
        episode_solve.append(num_eps)
        print("--------------------------")
        print("Solved {} runs out of {}".format(num_successes, i+1))

    print("--------------------------")
    # print("Episodes to solve: " + str(episodes) + ", Achieved Avg. Reward: " + str(achieved_avg_reward))
    print("Solved {} runs out of 20".format(num_successes))
    print(episode_solve)

