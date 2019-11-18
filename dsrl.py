
import random
import pandas as pd
from env import *

actions = ['up', 'down', 'right', 'left']


def choose_action(s_alg, state, agent_pos, model, s_prob):
    # print("\nPREVIOUS MODEL - CHOOSE ACTION\n", model)
    zero = False
    a_v_list = []
    d = {}
    obj_list = create_obj_list(state)
    rel_list = relation_obj_list(obj_list, agent_pos)
    new_state = rel_list

    for obj in new_state: # FOR ALL OBJECTS SEEN
        tp_n_c = str(obj.tp) # GET THE TYPE FROM THE NEW STATE
        s_n_c = str(obj.loc) # GET THE LOCATION FROM THE NEW STATE
        if tp_n_c not in model.columns:
            # print("tp_n_c not in model.columns", tp_n_c)
            model[tp_n_c] = 0
        if s_n_c not in model.index:
            #print("s_n_c not in model.index", s_n_c)
            m_index = pd.MultiIndex(levels=[[s_n_c], actions],
                                    labels=[[0, 0, 0, 0], [0, 1, 2, 3]],
                                    names=['state', 'actions'])
            df_zero = pd.DataFrame(index=m_index)
            model = model.append(df_zero)
            model = model.fillna(0)
        Qts_a = model[tp_n_c].loc[s_n_c]
        # print("Qts_a - ", Qts_a)
        if s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near": # Calculate the distance
            s_n_c_abs = [int(s) for s in s_n_c if s.isdigit()]  # s_n_c_abs = state_new_absolute_distance
            distance = np.sqrt(s_n_c_abs[0]**2 + s_n_c_abs[1]**2)
            # print("distance",distance)
            Qts_a = Qts_a.divide(distance*distance, axis=0)
        a_v = [(value, key) for value, key in Qts_a.items()]
        # print("Qts_a - NEW", Qts_a)
        a_v_list.append(a_v) # Append Q-value

    # Sum the values of all Qs into a single Q
    for element in a_v_list:
        for a in element:
            act = a[0] # Action
            val = a[1] # Value
            d[act] = d.get(act, 0) + val # Sum values for each Q


    if d != {}: # BE CAREFUL THIS IS A DICT (argmax does not work as usual)
        inverse = [(value, key) for key, value in d.items()] # CALCULATE ALL KEYS
        n_action = max(inverse)[1] # Choose the max argument

        if max(d.values()) == 0: zero = True
    else:
        n_action = "down"


    x = random.random()  # E greedy exploration
    if x < s_prob:
        n_action = random.choice(actions)
        print_action = 'Random Act (Prob):'
    elif zero == True:
        n_action = random.choice(actions)
        print_action = 'Random Act (Zero):'
    else:
        print_action = 'Chosen Act:'
    # print("\nNEW MODEL - CHOOSE ACTION\n", model)
    return n_action, model, print_action


# region CREATE OBJ_LIST FROM STATE AND RELATIONSHIP LIST BETWEEN AGENT AND OBJECTS
''' CREATE obj_list - FROM env '''
def create_obj_list(env):
    obj_list_fun = []
    tp_list = []
    loc_list = []
    env = env.transpose()
    h_max = env.shape[0]
    # print("h_max", h_max)
    v_max = env.shape[1]
    # print("v_max",v_max)
    for h in range(1, (h_max - 1)):
        for v in range(1, (v_max - 1)):
            if env[h][v] != 0:
                tp_list.append(env[h][v])
                loc_list.append((h, v))
    for i in range(len(loc_list)):
        tp = tp_list[i]
        loc = loc_list[i]
        obj = Obj(tp, loc)
        obj_list_fun.append(obj)
    return obj_list_fun

''' CREATE A RELATIONSHIP LIST BETWEEN AGENT AND OBJECTS - FROM obj_list '''
def relation_obj_list(obj_list, agent_pos):
    rel_list = []
    xA = agent_pos[0]
    yA = agent_pos[1]
    # print("xA", xA)
    # print("yA", yA)
    for obj in obj_list:
        xB = obj.loc[0]
        yB = obj.loc[1]
        x = xA - xB
        y = yA - yB
        loc_dif = (x, y)
        # loc_dif = (x[0], y[0])
        tp = obj.tp
        obj = Obj(tp, loc_dif)
        rel_list.append(obj)
    return rel_list
# endregion



def learn(s_alg, model, state_t, state_t1, agent_t_pos, agent_t1_pos, reward, action_t, end_game, net_conf, alfa, gamma):
    # print("\nPREVIOUS MODEL - LEARN\n", model)
    batch_loss = 0
    max_value = 0

    obj_list = create_obj_list(state_t)
    rel_list = relation_obj_list(obj_list, agent_t_pos)
    old_state = rel_list

    obj_list = create_obj_list(state_t1)
    rel_list = relation_obj_list(obj_list, agent_t1_pos)
    new_state = rel_list

    for i in range(len(old_state)):
        # Check all items in old state
        obj_prev = old_state[i]
        tp_prev = str(obj_prev.tp)
        s_prev = str(obj_prev.loc)
        # Check all items in new state
        obj_new = new_state[i]
        tp_new = str(obj_new.tp)
        s_new = str(obj_new.loc)

        if tp_new not in model.columns: # If type is new, then add type
            model[tp_new] = 0
        if s_new not in model.index: # If state is new, then add state
            m_index = pd.MultiIndex(levels=[[s_new], actions],
                                    labels=[[0, 0, 0, 0], [0, 1, 2, 3]],
                                    names=['state', 'actions'])
            df_zero = pd.DataFrame(index=m_index)
            model = model.append(df_zero)
            model = model.fillna(0)

        max_value = max(model[tp_new].loc[s_new])
        if s_alg == "DSRL": # THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL
            if end_game == False:
                Q_v = model[tp_prev].loc[s_prev, action_t]
                model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value) - Q_v)
            else:
                model[tp_prev].loc[s_prev, action_t] = reward

        elif s_alg == "DSRL_dist": # THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL
            if reward != 0:
                s_p_c = [int(s) for s in s_prev if s.isdigit()]
                if s_p_c[0] < 2 and s_p_c[1] < 2:
                    # EDITIONG DELETE
                    if end_game == False:
                        Q_v = model[tp_prev].loc[s_prev, action_t]
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value) - Q_v)
                    else:
                        model[tp_prev].loc[s_prev, action_t] = reward
            else:
                if end_game == False:
                    Q_v = model[tp_prev].loc[s_prev, action_t]
                    model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value) - Q_v)
                else:
                    model[tp_prev].loc[s_prev, action_t] = reward

        elif s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near": # THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL
            max_value_positive = max(model[tp_new].loc[s_new])
            if reward != 0:
                s_p_c = [int(s) for s in s_prev if s.isdigit()]  # s_p_c = state_previous_absolute_distance
                if s_p_c[0] < 2 and s_p_c[1] < 2: # IF IT IS CLOSE BY, THEN UPDATE ONLY THE CLOSE ONE:
                    if reward < 0 and tp_new == "180": # IF REWARD IS NEGATIVE and NEW OBJECT IS NEGATIVE UPDATE ONLY NEGATIVE TYPE:
                        if end_game == False:
                            Q_v = model[tp_prev].loc[s_prev, action_t]
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                        else:
                            model[tp_prev].loc[s_prev, action_t] = reward
                    elif reward > 0 and tp_new == "60":  # IF REWARD IS POSITIVE and NEW OBJECT IS POSITIVE UPDATE ONLY POSITIVE TYPE:
                        if end_game == False:
                            Q_v = model[tp_prev].loc[s_prev, action_t]
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                        else:
                            model[tp_prev].loc[s_prev, action_t] = reward
            # IF reward is zero
            else:
                if end_game == False:
                    Q_v = model[tp_prev].loc[s_prev, action_t]
                    if tp_prev == "180": # IF THE PREVIOUS OBJECT WAS NEGATIVE
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                    elif tp_prev == "60": # IF THE PREVIOUS OBJECT WAS POSITIVE
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                else:
                    model[tp_prev].loc[s_prev, action_t] = reward

        elif s_alg == "DSRL_dist_type_near_propNeg": # I try to solve this with max and min, but it did not work very well(THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL)
            max_value_positive = max(model[tp_new].loc[s_new])
            min_value_negative = min(model[tp_new].loc[s_new])
            if reward != 0:
                s_p_c = [int(s) for s in s_prev if s.isdigit()]  # s_p_c = state_previous_absolute_distance
                if s_p_c[0] < 2 and s_p_c[1] < 2: # IF IT IS CLOSE BY, THEN UPDATE ONLY THE CLOSE ONE:
                    if reward < 0 and tp_new == "180": # IF REWARD IS NEGATIVE and NEW OBJECT IS NEGATIVE UPDATE ONLY NEGATIVE TYPE:
                        if end_game == False:
                            Q_v = model[tp_prev].loc[s_prev, action_t]
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * min_value_negative) - Q_v)
                        else:
                            model[tp_prev].loc[s_prev, action_t] = reward
                    elif reward > 0 and tp_new == "60":  # IF REWARD IS POSITIVE and NEW OBJECT IS POSITIVE UPDATE ONLY POSITIVE TYPE:
                        if end_game == False:
                            Q_v = model[tp_prev].loc[s_prev, action_t]
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                        else:
                            model[tp_prev].loc[s_prev, action_t] = reward
            # IF reward is zero
            else:
                if end_game == False:
                    Q_v = model[tp_prev].loc[s_prev, action_t]
                    if tp_prev == "180": # IF THE PREVIOUS OBJECT WAS NEGATIVE
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * min_value_negative) - Q_v)
                    elif tp_prev == "60": # IF THE PREVIOUS OBJECT WAS POSITIVE
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                else:
                    model[tp_prev].loc[s_prev, action_t] = reward

        elif s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
            max_value_positive = max(model[tp_new].loc[s_new])

            # Find the object that the agent interacted with:
            # This means that the agents has to know that the object which interacted with
            # After finding it, he has to assign the value to that object.
            # This means that I have to find the type and the state of this object that has now x=zero y=zero

            # print("obj_new.loc[0]\n", obj_new.loc[0])
            # print("obj_new.loc[1]\n", obj_new.loc[1])
            # print("action_t\n", action_t)
            # print("s_prev\n", s_prev)

            if obj_new.loc[0] == 0 and obj_new.loc[1] == 0:
                tp_to_update = tp_new
                # print("tp_new\n", tp_new)
                if action_t == "up":
                    s_prev_to_update = str((0,1))
                elif action_t == "down":
                    s_prev_to_update = str((0,-1))
                elif action_t == "right":
                    s_prev_to_update = str((-1,0))
                elif action_t == "left":
                    s_prev_to_update = str((1,0))
                # print("s_prev_to_update\n", s_prev_to_update)
                if end_game == False:
                    Q_v = model[tp_to_update].loc[s_prev_to_update, action_t]
                    model[tp_to_update].loc[s_prev_to_update, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                else:
                    model[tp_to_update].loc[s_prev_to_update, action_t] = reward

            if reward == 0:
                if end_game == False:
                    Q_v = model[tp_prev].loc[s_prev, action_t]
                    model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                else:
                    model[tp_prev].loc[s_prev, action_t] = reward


    # print("\nNEW MODEL - LEARN\n", model)
    return model, batch_loss
