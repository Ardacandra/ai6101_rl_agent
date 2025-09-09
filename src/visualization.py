import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize(q_table):
    v_table = {}
    policy = {}
    for key, v in q_table.items():
        ### ASSIGNMENT START

        v_table[key] = max(v.values()) # get the maximum Q value among all possible actions
        policy[key] = max(v, key=v.get) # get the action that has the highest Q value

        ### ASSIGNMENT END
    state_num = len(q_table.keys())
    print(f"State space: {state_num}")
    
    # Print the largest 20 state values in v_table and the corresponding policy
    for k, val in sorted(v_table.items(), key=lambda x: x[1], reverse=True)[:20]:
      print("v_table", k, val / state_num)
      print("policy", k, policy[k])

def plot_state_grid(world_state, val_to_int, cmap, title, output_path):
    """
    Plot the state grid of the environment.

    Args:
        world_state : 2D array showing current state
        val_to_int : dictionary to map state values to integers
        cmap : colormap for each state
        title : plot title
        output_path : path to save the plot
    """
    #map state values to integers
    world_state_int = np.vectorize(val_to_int.get)(world_state)

    #plot the state matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(world_state_int, cmap=cmap)
    
    #plot grids
    plt.xticks(np.arange(-0.5, world_state.shape[1], 1), minor=True)
    plt.yticks(np.arange(-0.5, world_state.shape[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=1)

    #overlay entity labels
    for i in range(world_state.shape[0]):
        for j in range(world_state.shape[1]):
            plt.text(j, i, world_state[i][j], ha="center", va="center", color="black", fontsize=14)
    
    plt.title(title)
    plt.savefig(output_path)
    plt.close()