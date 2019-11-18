# coding=utf-8
# # ------------------------------------------------------------------------------------------------- #
# This code can run 2 different models of Reinforcement Learning:
# SRL (DSRL), SRL+CS(DSRL_object_near) and some other variations of SRL
# The setting for each run can be set at the end of the code
# It can load and save the models in Excel form
# There are some pre-defined environments, but you can create your own
# Press P to stop
# -------------------------------------------------------------------------------------------------- #


import sys
import time
from time import sleep
import os
from env import *
from dsrl import *
from display import *
from model import *
import pprint

# region PANDAS DEFINITION
pd.set_option('display.max_columns', None)
pd.set_option('display.large_repr', 'info')
desired_width = 180
pd.set_option('display.width', desired_width)
pd.set_option('precision', 4)
# endregion

np.random.seed(123)  # For reproducibility
pygame.init()  # Pygame initialialization
pp = pprint.PrettyPrinter(indent=4)
actions_dict = {'up':0, 'down':1, 'right':2, 'left':3}
p_keys = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]

# region REWARDS
negative_reward = 10  # Negative Reward
positive_reward = 1  # Positive Reward
step_reward = 0  # Reward received by each step
# endregion

alfa = 1 # Learning Rate
gamma = 0.9 # Temporal Discount Factor

''' PROGRAM START '''

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def run(s_env, s_alg, s_learn, s_load, s_print, s_auto, s_episode, s_cond_to_end, s_server, s_net_comb_param, s_load_path, s_prob, s_sample, s_save):
    net_conf = {"N_actions": n_actions,
                "Max_memory": max_memory_list[s_net_comb_param],
                "Hidden_size": hidden_size_list[s_net_comb_param],
                "Batch_size": batch_size_list[s_net_comb_param],
                "Optimizer": optimizer_list[0]}
    begin = time.time()
    begin_time = time.strftime('%X %x')
    print("\n\n --- BEGINING --- s_sample: %s \n begin_time: %s \n" % (s_sample, begin_time))

    df_score = pd.DataFrame()
    df_percent_list = pd.DataFrame()
    df_loss_list = pd.DataFrame()
    df_time_sample = pd.DataFrame()
    avg_last_score_list = []

    if s_server == False: screen = pygame.display.set_mode((400 + 37 * 5, 330 + 37 * 5))

    score_list_best = [0]
    for sample in list(range(1, s_sample+1)):
        experiment_configurations = (sample, s_env, s_alg, s_episode, s_learn, s_load, s_print, s_auto, s_cond_to_end, s_server, s_net_comb_param, s_prob)
        print("\n - START - "
              "\n sample: %s"
              "\n s_env: %s"
              "\n s_alg: %s"
              "\n s_episode: %s"
              "\n s_learn: %s"
              "\n s_load: %s"
              "\n s_print: %s"
              "\n s_auto: %s"
              "\n s_cond_to_end: %s"
              "\n s_server: %s"
              "\n s_net_comb_param: %s"
              "\n s_prob: %s" % experiment_configurations)

        start = time.time()
        start_time = time.strftime('%X %x')
        print("\nStart time: ", start_time)
        negativo_list, positivo_list, agent, wall_list, h_max, v_max = environment_conf(s_env)

        env_dim = [h_max, v_max]
        if s_load == True:
            try:
                model, op_conf = load_model(s_alg, __location__ + s_load_path)
            except:
                print("DID NOT FIND THE FILE")
        else:
            model, op_conf = create_model(s_alg, env_dim, net_conf)

        # region INITIALIZE VARIABLES 1
        percent_list = []
        score = 0
        score_list = []
        episodes = 0
        episodes_list = []
        steps = 0
        steps_list = []
        batch_loss = 0
        loss_list = []
        # endregion

        while (episodes < s_episode):  # max_episodes
            negativo_list, positivo_list, agent, wall_list, h_max, v_max = environment_conf(s_env)
            # region INITIALIZE VARIABLES 2
            episodes += 1
            episodes_list.append(episodes)
            max_steps = 10
            steps_list.append(steps)
            steps = 0
            act_list = []
            last_move = False
            action_chosen = ""
            encountered = 0
            pos_collected = 0
            prob = s_prob
            # endregion

            if s_server == False:
                # region DRAW SCREEN
                screen.fill(white)
                show_Alg(s_alg, screen)
                show_Samples(sample, screen)
                show_Level(episodes, screen)
                show_Score(score, screen)
                show_Steps(steps, screen)
                show_Percent(percent_list[-10:], screen)
                show_Steps_list(steps_list[-30:], screen)
                show_Act_List(act_list[-20:], screen)
                show_Action(action_chosen, screen)
                show_Env(s_env, screen)
                draw_objects(agent, positivo_list, negativo_list, wall_list, screen)
                pygame.display.flip()
                # endregion

            while (True):  # max_steps or condition to finish
                sleep(speed)
                ''' EVENT HANDLE '''
                key_pressed = False
                set_action = False
                while (s_server == False):
                    for event in pygame.event.get():
                        # QUIT GAME
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        # ADD OR DELETE WALL
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            pass
                        # PRESS A KEY
                        if event.type == pygame.KEYDOWN:
                            # SAVE AND QUIT - KEY P
                            if event.key == pygame.K_p:
                                pygame.quit()
                                sys.exit()

                            # MOVE - SPACE BAR
                            if event.key == pygame.K_SPACE:
                                key_pressed = True
                                break
                            # MOVE - ARROW KEYS
                            if event.key in p_keys:
                                key_pressed = True
                                set_action = True
                                if event.key == pygame.K_w:  # North # add_act('↑') ⇦ ⇨ ⇧ ⇩
                                    key_action = "up"
                                if event.key == pygame.K_s:  # South # add_act('↓') ⬅  ➡  ⬆ ⬇
                                    key_action = "down"
                                if event.key == pygame.K_d:  # West # add_act('→')
                                    key_action = "right"
                                if event.key == pygame.K_a:  # East # add_act('←')
                                    key_action = "left"
                                break
                    # Run game if key is preseed or automatic is selected
                    if key_pressed or s_auto:
                        break
                # BREAK IF IT WAS THE LAST MOVE
                if last_move == True:
                    break
                # RUN_GAME
                steps += 1
                ''' OLD STATE - S 1 - 1'''
                state_t = update_state(h_max, v_max, agent, positivo_list, negativo_list, wall_list)
                agent_t = agent.pos
                ''' CHOOSE ACTION - AGENT ACT - 2'''
                action_chosen, model, print_action = choose_action(s_alg, state_t, agent_t, model, prob)

                if set_action: action_chosen = key_action

                ''' CHANGE THE WORLD - UP_ENV - 3'''
                agent.try_move(action_chosen, wall_list)
                act_list.append(action_chosen)
                if s_print: print(print_action, action_chosen)

                ''' NEW STATE - S2 - 4'''
                state_t1 = update_state(h_max, v_max, agent, positivo_list, negativo_list, wall_list)
                agent_t1 = agent.pos
                if s_print:
                    print('\n>>>>   Level: ' + str(episodes) + ' |  Step: ' + str(
                        steps) + ' |  New_agent_pos: ' + str(agent.pos) + '  <<<<')

                ''' GET REWARD - 5 '''
                # region GET REWARD AND DELETE COLLECTED OBJECT
                prev_score = score
                score += step_reward

                for positivo in positivo_list:
                    if agent.pos == positivo.pos:
                        encountered += 1
                        pos_collected += 1
                        score += positive_reward
                        positivo = Positivo('./images/positivo', agent.pos[0], agent.pos[1])
                        positivo_list.remove(positivo)
                        if s_print == True and s_server == False:
                            print('                                 Hit the Positivo')
                for negativo in negativo_list:
                    if agent.pos == negativo.pos:
                        encountered += 1
                        score -= negative_reward
                        negativo = Negativo('./images/negativo', agent.pos[0], agent.pos[1])
                        negativo_list.remove(negativo)
                        if s_print == True and s_server == False:
                            print('                                 Hit the Negativo')

                new_score = score
                score_list.append(score)
                reward = new_score - prev_score
                # endregion

                ''' LEARN - 6 '''
                # CONDITION TO FINISH THE Episode
                if s_cond_to_end == 'max_steps':
                    if steps == max_steps:
                        last_move = True
                elif s_cond_to_end == 'coll_all' or steps > max_steps:
                    if len(positivo_list) == 0 and len(negativo_list) == 0 or steps > max_steps:
                        last_move = True
                elif s_cond_to_end == 'only_positive' or steps > max_steps:
                    if len(positivo_list) == 0 or steps > max_steps:
                        last_move = True
                elif s_cond_to_end == 'only_negative' or steps > max_steps:
                    if len(negativo_list) == 0 or steps > max_steps:
                        last_move = True

                # LEARN
                if s_learn == True:
                    action_t = action_chosen
                    if last_move == False:
                        ''' LEARN '''
                        model, batch_loss = learn(s_alg, model, state_t, state_t1, agent_t, agent_t1, reward, action_t, False, net_conf, alfa, gamma)
                    else:
                        ''' LEARN FINAL '''
                        model, batch_loss = learn(s_alg, model, state_t, state_t1, agent_t, agent_t1, reward, action_t, True, net_conf, alfa, gamma)

                if s_server == False:
                    # region DRAW SCREEN
                    screen.fill(white)
                    show_Alg(s_alg, screen)
                    show_Samples(sample, screen)
                    show_Level(episodes, screen)
                    show_Score(score, screen)
                    show_Steps(steps, screen)
                    show_Percent(percent_list[-10:], screen)
                    show_Steps_list(steps_list[-30:], screen)
                    show_Act_List(act_list[-20:], screen)
                    show_Action(action_chosen, screen)
                    show_Env(s_env, screen)

                    draw_objects(agent, positivo_list, negativo_list, wall_list, screen)
                    pygame.display.flip()
                    # endregion

            try:
                percent = pos_collected / encountered
            except ZeroDivisionError:
                percent = 0
            percent_list.append(percent)
            loss_list.append(batch_loss)
            print("Episode: ", episodes)

        # region TIME 1
        print("Start time: ", start_time)
        end = time.time()
        end_time = time.strftime('%X %x')
        print("End time: ", end_time)
        time_elapsed = end - start
        print("Time elapsed: ", time_elapsed)
        # endregion

        '''GET THE BEST MODEL'''
        if max(score_list) > max(score_list_best):
            best_model = model
            score_list_best = score_list

        # region MAKE LIST OF THE RESULTS
        avg_last_score_list.append(score_list[-1])

        score_list_df = pd.DataFrame({'Score': score_list})
        percent_list_df = pd.DataFrame({'Percent': percent_list})
        loss_list_df = pd.DataFrame({'Batch_loss': loss_list})
        time_sample_df = pd.DataFrame({'Time': [time_elapsed]})

        df_score = pd.concat([df_score, score_list_df], ignore_index=True, axis=1)
        df_percent_list = pd.concat([df_percent_list, percent_list_df], ignore_index=True, axis=1)
        df_loss_list = pd.concat([df_loss_list, loss_list_df], ignore_index=True, axis=1)
        df_time_sample = pd.concat([df_time_sample, time_sample_df], ignore_index=True, axis=1)
        # endregion

    if s_save == True:
        # region PATH TO SAVE
        save_path_core = __location__ + "/Results/"
        if s_learn == True: save_path = save_path_core + "Train/Env_" + str(s_env) + "/Train_Env_" + str(s_env) + "_" + s_alg
        else: save_path = save_path_core + "Test/Env_" + str(s_env) + "/Test_Env_" + str(s_env) + "_" + s_alg

        # convert begin_time to string and format it
        time_path = begin_time.replace(" ", "   ")
        time_path = time_path.replace(":", " ")
        time_path = time_path.replace("/", "-")
        # append to the save path
        save_path = save_path + "   " + time_path

        if s_load == True:
            load_path = " loaded_with " + s_load_path.replace("/", "_")
            save_path = save_path + load_path

        # If it doesnt find the path, then create a new path
        if not os.path.exists(os.path.dirname(save_path)):
            try:
                os.makedirs(os.path.dirname(save_path))
            except OSError as exc:  # Guard against race condition
                print("ERROR when saving the File")
        # endregion
        print("save_path: ", save_path)

        # region SAVE ALL
        # IF IT IS NOT DQN NULL NET CONF. VALUES
        op_conf = [0, 0, 0, 0, 0, 0]
        net_conf = {"N_actions":0, "Max_memory":0, "Hidden_size":0, "Batch_size":0, "Optimizer":"none"}

        avg_last_score = np.average(avg_last_score_list)
        config_list = pd.concat([pd.Series({'Run_Conf': "A"}),
                                 pd.Series({'Env_conf': s_env}),
                                 pd.Series({'Algort': s_alg}),
                                 pd.Series({'Learn': s_learn}),
                                 pd.Series({'Load': s_load}),
                                 pd.Series({'Samples': s_sample}),
                                 pd.Series({'Episode': s_episode}),
                                 pd.Series({'Max_steps': max_steps}),
                                 pd.Series({'s_cond_to_end': s_cond_to_end}),
                                 pd.Series({'Auto': s_auto}),
                                 pd.Series({'Server': s_server}),
                                 pd.Series({'Print': s_print}),
                                 pd.Series({'MODEL CONF': ""}),
                                 pd.Series({'alfa': alfa}),
                                 pd.Series({'gamma': gamma}),
                                 pd.Series({'Prob': Prob}),
                                 pd.Series({'N_actions': net_conf["N_actions"]}),
                                 pd.Series({'Max_memory': net_conf["Max_memory"]}),
                                 pd.Series({'Hidden_size': net_conf["Hidden_size"]}),
                                 pd.Series({'Batch_size': net_conf["Batch_size"]}),
                                 pd.Series({'Optimizer': net_conf["Optimizer"]}),
                                 pd.Series({'lr': op_conf[0]}),
                                 pd.Series({'beta_1': op_conf[1]}),
                                 pd.Series({'beta_2': op_conf[2]}),
                                 pd.Series({'epsilon': op_conf[3]}),
                                 pd.Series({'decay': op_conf[4]}),
                                 pd.Series({'rho': op_conf[5]}),
                                 pd.Series({'': ""}),
                                 pd.Series({'AVG SCORE': avg_last_score})])
        config_list = config_list.to_frame()

        if s_print: print("\nconfig_list:\n", config_list)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(save_path + ".xlsx", engine='xlsxwriter')

        # SAVING CONFIG:
        config_list.to_excel(writer, sheet_name='Run_Conf', header=False)
        worksheet = writer.sheets['Run_Conf']
        worksheet.set_column('A:B', 15)
        # SAVING SCORE:
        df_score_mean = df_score.mean(axis=1)
        df_score.insert(0, "Avg " + str(s_sample), df_score_mean)
        df_score.to_excel(writer, sheet_name='Score')
        worksheet = writer.sheets['Score']
        worksheet.write(0, 0, "Score")
        # SAVING PERCENT:
        df_percent_list_mean = df_percent_list.mean(axis=1)
        df_percent_list.insert(0, "Avg " + str(s_sample), df_percent_list_mean)
        df_percent_list.to_excel(writer, sheet_name='Percent')
        worksheet = writer.sheets['Percent']
        worksheet.write(0, 0, "Percent")
        # SAVING LOSS:
        df_loss_list.to_excel(writer, sheet_name='Loss')
        worksheet = writer.sheets['Loss']
        worksheet.write(0, 0, "Loss")
        # SAVING TIME:
        df_time_sample.to_excel(writer, sheet_name='Time')
        worksheet = writer.sheets['Time']
        worksheet.write(0, 0, "Time")

        # SAVING BEST MODEL (out of # Samples):
        if s_alg == "DSRL" or s_alg == "DSRL_dist" or s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
            # SAVING MODEL CONFIGURATIONSؤ
            best_model.to_excel(writer, sheet_name='model')
            # CONDITIONAL COLOR
            worksheet = writer.sheets['model']
            for x in range(2, 700, 4):
                cell = "C" + str(x) + ":D" + str(x + 3)
                worksheet.conditional_format(cell, {'type': '3_color_scale'})
            # CELL SIZE
            worksheet = writer.sheets['model']
            worksheet.set_column('A:A', 50)


        writer.save()
        # endregion

    print("\n - END - "
          "\n sample: %s"
          "\n s_env: %s"
          "\n s_alg: %s"
          "\n s_episode: %s"
          "\n s_learn: %s"
          "\n s_load: %s"
          "\n s_print: %s"
          "\n s_auto: %s"
          "\n s_cond_to_end: %s"
          "\n s_server: %s"
          "\n s_net_comb_param: %s"
          "\n s_prob: %s" % experiment_configurations)

    # region TIME 2
    print("\n\nBegin time: ", begin_time)
    finish = time.time()
    finish_time = time.strftime('%X %x')
    print("Final time: ", finish_time)
    total_time = finish - begin
    print("Total time: ", total_time)
    # endregion

    return

if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------- #
    ''' SELECT PARAMETERS TO RUN THE SOFTWARE '''
    Env = 11
    Alg_list = ["DSRL",
                "DSRL_object_near",
                "DSRL_dist",
                "DSRL_dist_type",
                "DSRL_dist_type_near",
                "DSRL_dist_type_near_propNeg",
                "DSRL_object"]
    Alg = Alg_list[3] # Select the algorithm to be used
    Learn = True # To update its knowledge
    Load = False # To load a learned model
    Load_path = "/Results/Train/Env_1/Train_Env_1_DQN_4   00 33 03   01-05-18"

    Samples = 2 # Usually 10 samples
    Print = False # Print some info in the terminal
    Auto = True # Agent moves Automatic or if False it moves by pressing the Spacebar key
    Server = False # If running in the server since
    Prob = 0 # Probability to make a random move (exploration rate)
    Cond_to_end = "only_positive" # Choose from below (there are 4)
    Save = True # Save the model
    speed = 0 # seconds per frame

    Episodes = 1000 # Usually 1000 or 100

    max_memory_list =  [100,    100,    100,    300, 300,   300,    900, 900, 900]
    hidden_size_list = [5,      10,     15,     5,   10,    15,     5,   10,  15]
    batch_size_list =  [32,     32,     32,     32,  32,    32,     32,  32,  32]
    optimizer_list = ["adam", "rms_opt"]
    n_actions = 4  # [move_up, move_down, move_left, move_right]
    # endregion
    Net_comb_param = 4

    run(Env, Alg, Learn, Load, Print, Auto, Episodes, Cond_to_end, Server, Net_comb_param, Load_path, Prob, Samples, Save)

