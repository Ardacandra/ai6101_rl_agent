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