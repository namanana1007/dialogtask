import os,sys
pathx = os.path.dirname(os.path.abspath(__file__))
print(("pathx:",pathx))
sys.path.append(pathx)

import argparse, json, copy, os,sys
#import cPickle as pickle  /python2
import pickle

from .deep_dialog.dialog_system import DialogManager, text_to_dict
from .deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN, RequestInformSlotAgent
from .deep_dialog.usersims import RealUser, RuleSimulator, RuleRestaurantSimulator, RuleTaxiSimulator

from .deep_dialog import dialog_config
from .deep_dialog.dialog_config import *

from .deep_dialog.nlu import nlu
from .deep_dialog.nlg import nlg

global dialog_manager
global first_time

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""

""" load action """
def load_actions(sys_req_slots, sys_inf_slots):
    dialog_config.feasible_actions = [
        {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
    ]

    for slot in sys_inf_slots:
        dialog_config.feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

    for slot in sys_req_slots:
        dialog_config.feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})

def init_dialog_manager():
    params = {
      "simulation_epoch_size": 50, 
      "slot_err_mode": 0, 
      "diaact_nl_pairs": "./dialogweb/deep_dialog/data_taxi/sim_dia_act_nl_pairs.json", 
      "save_check_point": 10, 
      "episodes": 500, 
      "predict_mode": False, 
      "cmd_input_mode": 0, 
      "goal_file_path": "./dialogweb/deep_dialog/data_taxi/user_goals_first.v4_1.p", 
      "max_turn": 30, 
      "experience_replay_pool_size": 1000, 
      "write_model_dir": "./dialogweb/deep_dialog/checkpoints/", 
      "usr": 0, 
      "auto_suggest": 0, 
      "run_mode": 0, 
      "trained_model_path": "./dialogweb/deep_dialog/checkpoints/taxi/nl/dqn/agt_13_419_500_0.19000.p", 
      "success_rate_threshold": 0.3, 
      "nlu_model_path": "./dialogweb/deep_dialog/models/nlu/taxi/lstm_[1532583523.63]_88_99_400_0.998_n.p", 
      "epsilon": 0, 
      "batch_size": 16, 
      "learning_phase": "all", 
      "nlg_model_path": "./dialogweb/deep_dialog/models/nlg/taxi/lstm_tanh_[1532457558.95]_95_99_194_0.985_n.p", 
      "act_set": "./dialogweb/deep_dialog/data_taxi/dia_acts.txt", 
      "slot_err_prob": 0.0, 
      "warm_start": 1, 
      "warm_start_epochs": 100, 
      "dict_path": "./dialogweb/deep_dialog/data_taxi/slot_dict.v1_1.p", 
      "intent_err_prob": 0.0, 
      "split_fold": 5, 
      "slot_set": "./dialogweb/deep_dialog/data_taxi/taxi_slots.txt", 
      "act_level": 0, 
      "dqn_hidden_size": 80, 
      "agt": 11, 
      "kb_path": "./dialogweb/deep_dialog/data_taxi/taxi.kb.1k.v1_1.p", 
      "gamma": 0.9
    }

    max_turn = params['max_turn']
    num_episodes = params['episodes']

    agt = params['agt']
    usr = params['usr']

    dict_path = params['dict_path']
    goal_file_path = params['goal_file_path']

    # load the user goals from .p file
    all_goal_set = pickle.load(open(goal_file_path, 'rb'))

    # split goal set
    split_fold = params.get('split_fold', 5)
    goal_set = {'train':[], 'valid':[], 'test':[], 'all':[]}
    for u_goal_id, u_goal in enumerate(all_goal_set):
        if u_goal_id % split_fold == 1: goal_set['test'].append(u_goal)
        else: goal_set['train'].append(u_goal)
        goal_set['all'].append(u_goal)
    # end split goal set

    kb_path = params['kb_path']
    kb = pickle.load(open(kb_path, 'rb'))

    act_set = text_to_dict(params['act_set'])
    slot_set = text_to_dict(params['slot_set'])

    ################################################################################
    # a movie dictionary for user simulator - slot:possible values
    ################################################################################
    movie_dictionary = pickle.load(open(dict_path, 'rb'))

    dialog_config.run_mode = params['run_mode']
    dialog_config.auto_suggest = params['auto_suggest']

    ################################################################################
    #   Parameters for Agents
    ################################################################################
    agent_params = {}
    agent_params['max_turn'] = max_turn
    agent_params['epsilon'] = params['epsilon']
    agent_params['agent_run_mode'] = params['run_mode']
    agent_params['agent_act_level'] = params['act_level']

    agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
    agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
    agent_params['batch_size'] = params['batch_size']
    agent_params['gamma'] = params['gamma']
    agent_params['predict_mode'] = params['predict_mode']
    agent_params['trained_model_path'] = params['trained_model_path']
    agent_params['warm_start'] = params['warm_start']
    agent_params['cmd_input_mode'] = params['cmd_input_mode']
    

    if agt == 0:
        agent = AgentCmd(kb, act_set, slot_set, agent_params)
    elif agt == 1:
        agent = InformAgent(kb, act_set, slot_set, agent_params)
    elif agt == 2:
        agent = RequestAllAgent(kb, act_set, slot_set, agent_params)
    elif agt == 3:
        agent = RandomAgent(kb, act_set, slot_set, agent_params)
    elif agt == 4:
        #agent = EchoAgent(kb, act_set, slot_set, agent_params)
        agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, movie_request_slots, movie_inform_slots)
    elif agt == 5: # movie request rule agent
        agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, movie_request_slots)
    elif agt == 6: # restaurant request rule agent
        agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, restaurant_request_slots)
    elif agt == 7: # taxi request agent
        agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, taxi_request_slots)
    elif agt == 8: # taxi request-inform rule agent
        agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, taxi_request_slots, taxi_inform_slots)
    elif agt == 9: # DQN agent for movie domain
        agent = AgentDQN(kb, act_set, slot_set, agent_params)
        agent.initialize_config(movie_request_slots, movie_inform_slots)
    elif agt == 10: # restaurant request-inform rule agent
        agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, restaurant_request_slots, restaurant_inform_slots)
    elif agt == 11: # taxi request-inform-cost rule agent
        agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, taxi_request_slots, taxi_inform_cost_slots)
    elif agt == 12: # DQN agent for restaurant domain
        load_actions(dialog_config.restaurant_sys_request_slots, dialog_config.restaurant_sys_inform_slots)
        agent = AgentDQN(kb, act_set, slot_set, agent_params)
        agent.initialize_config(restaurant_request_slots, restaurant_inform_slots)
    elif agt == 13: # DQN agent for taxi domain
        load_actions(dialog_config.taxi_sys_request_slots, dialog_config.taxi_sys_inform_slots)
        agent = AgentDQN(kb, act_set, slot_set, agent_params)
        agent.initialize_config(taxi_request_slots, taxi_inform_slots)
        
    ################################################################################
    #    Add your agent here
    ################################################################################
    else:
        pass

    ################################################################################
    #   Parameters for User Simulators
    ################################################################################
    usersim_params = {}
    usersim_params['max_turn'] = max_turn
    usersim_params['slot_err_probability'] = params['slot_err_prob']
    usersim_params['slot_err_mode'] = params['slot_err_mode']
    usersim_params['intent_err_probability'] = params['intent_err_prob']
    usersim_params['simulator_run_mode'] = params['run_mode']
    usersim_params['simulator_act_level'] = params['act_level']
    usersim_params['learning_phase'] = params['learning_phase'] 

    if usr == 0:# real user
        user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    elif usr == 1: # movie simulator
        user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    elif usr == 2: # restaurant simulator
        user_sim = RuleRestaurantSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    elif usr == 3: # taxi simulator
        user_sim = RuleTaxiSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params) 

    ################################################################################
    #    Add your user simulator here
    ################################################################################
    else:
        pass

    ################################################################################
    # load trained NLG model
    ################################################################################
    nlg_model_path = params['nlg_model_path']
    diaact_nl_pairs = params['diaact_nl_pairs']
    nlg_model = nlg()
    nlg_model.load_nlg_model(nlg_model_path)
    nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs) # load nlg templates 

    agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)

    ################################################################################
    # load trained NLU model
    ################################################################################
    nlu_model_path = params['nlu_model_path']
    nlu_model = nlu()
    nlu_model.load_nlu_model(nlu_model_path)

    agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)

    ################################################################################
    # Dialog Manager
    ################################################################################
    global dialog_manager
    dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, kb)
    global first_time
    first_time = True

def get_first_time_value():
    global first_time
    return first_time

def set_first_time_value(first_time_value):
    global first_time
    first_time = first_time_value

def reset_dialog_manager(command):
    global dialog_manager
    global first_time
    first_time = False
    dialog_manager.initialize_episode(command)
    episode_over,ansstr = dialog_manager.next_turn_qian()
    if episode_over:
        first_time = True
    return ansstr, episode_over

def next_dialos_manager(command):
    global dialog_manager
    global first_time
    episode_over, reward = dialog_manager.next_turn_hou(command)
    episode_over,ansstr = dialog_manager.next_turn_qian()
    if episode_over:
        first_time = True
    return ansstr, episode_over
