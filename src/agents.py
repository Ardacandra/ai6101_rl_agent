import numpy as np
import math

class RandomAgent:
    def __init__(self, env, num_episodes):
        self.action_space = [1, 2, 3, 4]
        self.env = env
        self.num_episodes = num_episodes

    def act(self):
        """Returns a random choice of the available actions"""
        return np.random.choice(self.action_space)

    def learn(self):
        rewards = []
        
        for _ in range(self.num_episodes):
            cumulative_reward = 0 # Initialise values of each game
            state = self.env.reset()
            done = False
            while not done: # Run until game terminated
                action = self.act() 
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += reward
                state = next_state
            rewards.append(cumulative_reward)

        return rewards
    
class RLAgent:
    def __init__(
            self, env, num_episodes, n_training_phase=10, alpha=0.1, gamma=0.99, 
            exp_method="epsilon_greedy", epsilon=0.1,
            c=2.0, tau=0.5

        ):
        self.action_space = env.action_space
        self.q_table = dict() # Store all Q-values in a dictionary
        self.count_table = dict() # Store all action counts in a dictionary (for UCB)
        # Loop through all possible grid spaces, create sub-dictionary for each
        for agent_x in range(env.world_height):
            for agent_y in range(env.world_width):
                for box_x in range(env.world_height):
                    for box_y in range(env.world_width):
                        # Populate sub-dictionary with zero values for possible moves
                        self.q_table[(agent_x, agent_y, box_x, box_y)] = {k: 0 for k in self.action_space}
                        self.count_table[(agent_x, agent_y, box_x, box_y)] = {k: 0 for k in self.action_space}

        self.env = env
        self.num_episodes = num_episodes
        self.n_training_phase = n_training_phase # additional parameter to get fall-off rate per phase
        self.alpha = alpha
        self.gamma = gamma
        self.exp_method = exp_method # additional parameter to select exploration method
        self.epsilon = epsilon
        self.c = c # additional parameter for UCB
        self.tau = tau # additional parameter for boltzmann

        self.timesteps = 1 # to keep track of time steps for UCB
        
    def act(self, state, is_training = True):
        # """Returns the (epsilon-greedy) optimal action from Q-Value table."""
        """ Returns the optimal action based on the exploration method specified."""
        ### ASSIGNMENT START
        q_values_of_state = self.q_table[state]
        count_values_of_state = self.count_table[state]

        if self.exp_method == "epsilon_greedy":
            if np.random.uniform(0,1) < self.epsilon and is_training:
                action = self.action_space[np.random.randint(0, len(self.action_space))]
            else:       
                maxValue = max(q_values_of_state.values())
                action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        elif self.exp_method == "ucb":
            if is_training:
                q_array = np.array([v for k, v in q_values_of_state.items()])
                n_array = np.array([v for k, v in count_values_of_state.items()])
                ucb_values = q_array + self.c * np.sqrt(np.log(self.timesteps + 1) / (n_array + 1e-5))
                ucb_dict = {k: v for k, v in zip(self.action_space, ucb_values)}
                maxValue = max(ucb_dict.values()) 
                action = np.random.choice([k for k, v in ucb_dict.items() if v == maxValue])
            else :
                maxValue = max(q_values_of_state.values())
                action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        elif self.exp_method == "boltzmann":
            if is_training:
                z = np.array([q_values_of_state[a] for a in self.action_space]) / self.tau
                prefs = np.exp(z - np.max(z)) # substract max for numerical stability
                action_probs = prefs / np.sum(prefs)
                action = np.random.choice(self.action_space, p=action_probs)
            else:        
                maxValue = max(q_values_of_state.values())
                action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        else :
            #if no exploration method specified, select random action
            action = self.action_space[np.random.randint(0, len(self.action_space))]
        
        return action
    
        ### ASSIGNMENT END

    def learn(self):
        """Updates Q-values iteratively."""
        ### ASSIGNMENT START

        rewards = [] #to store the reward of each episode
        cliff_fall_frac = [] #to store the fraction of times agent/box fell off each cliff per phase

        phase_length = math.ceil(self.num_episodes / self.n_training_phase)
        total_episode_count = 0

        for phase in range(self.n_training_phase):
            phase_episode_count = 0

            #create a dict to monitor the number of times agent/box fell off each cliff cell
            cliff_fall_dict = {}
            for x in range(self.env.world_height):
                for y in range(self.env.world_width):
                    cliff_fall_dict[(x, y)] = 0

            for _ in range(phase_length): 
                # stop training if num_episodes is reached
                if total_episode_count >= self.num_episodes:
                    break

                cumulative_reward = 0 # Initialise values of each game
                state = self.env.reset()
                done = False

                while not done: # Run until game terminated
                    #Applying Q-Learning Algorithm

                    #choose A from S using policy derived from Q
                    action = self.act(state, is_training=True)

                    #take action A, observe R and S'
                    next_state, reward, done, info = self.env.step(action)

                    #update Q(S, A)
                    q_original = self.q_table[state][action]
                    q_next_state = max(self.q_table[next_state].values())
                    self.q_table[state][action] = q_original + self.alpha * (reward + (self.gamma * q_next_state) - q_original)
                    
                    #update counts and time step for UCB
                    self.count_table[state][action] += 1
                    self.timesteps += 1

                    #update cumulative reward and current state
                    cumulative_reward += reward
                    state = next_state

                rewards.append(cumulative_reward)

                #check if agent/box fell off cliff and update dict
                if self.env._check_off_cliff(self.env.agent_pos):
                    cliff_fall_dict[tuple(self.env.agent_pos)] += 1
                if self.env._check_off_cliff(self.env.box_pos):
                    cliff_fall_dict[tuple(self.env.box_pos)] += 1
                
                total_episode_count += 1
                phase_episode_count += 1

            if phase_episode_count>0:
                cliff_fall_frac.append({cell: cliff_fall_dict[cell]/phase_episode_count for cell in cliff_fall_dict})

        return rewards, cliff_fall_frac
    
        ### ASSIGNMENT END