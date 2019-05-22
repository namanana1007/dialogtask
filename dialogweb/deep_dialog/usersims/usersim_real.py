from .usersim import UserSimulator
from deep_dialog import dialog_config

class RealUser(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        
        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        
        self.learning_phase = params['learning_phase']

    def initialize_episode(self,command):
        """ Initialize a new episode (dialog)"""

#        print "initialize episode"
        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
#        self.goal =  random.choice(self.start_set)
#        self.goal['request_slots']['ticket'] = 'UNK'
#        episode_over, user_action = self._sample_action()
#        assert (episode_over != 1),' but we just started'
#        return user_action
#        print "Turn", self.state['turn'], "user:",
#        command = raw_input()
        return self.generate_diaact_from_nl(command)

    def end_or_not(self, system_action):
        self.episode_over = False
        sys_act = system_action['diaact']
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        if (self.max_turn > 0 and self.state['turn'] + 2 > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            if sys_act == 'closing' or sys_act == "thanks":
                self.episode_over = True
        return self.episode_over, self.dialog_status

    def next(self, command, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        
        sys_act = system_action['diaact']
        
        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
#            self.state['history_slots'].update(self.state['inform_slots'])
#            self.state['inform_slots'].clear()

#            if sys_act == "inform":
#                self.response_inform(system_action)
#            elif sys_act == "multiple_choice":
#                self.response_multiple_choice(system_action)
#            elif sys_act == "request":
#                self.response_request(system_action) 
#            elif sys_act == "thanks":
#                self.response_thanks(system_action)
#            elif sys_act == "confirm_answer":
#                self.response_confirm_answer(system_action)
#            elif sys_act == "closing":
#                self.episode_over = True
#                self.state['diaact'] = "thanks"
            if sys_act == 'closing' or sys_act == "thanks":
                self.episode_over = True
                command = ""
#            else:
#                print "Turn", self.state['turn'], "user:",
#                command = raw_input()
#        self.corrupt(self.state)
#        
#        response_action = {}
#        response_action['diaact'] = self.state['diaact']
#        response_action['inform_slots'] = self.state['inform_slots']
#        response_action['request_slots'] = self.state['request_slots']
#        response_action['turn'] = self.state['turn']
#        response_action['nl'] = ""
#        
#        # add NL to dia_act
#        self.add_nl_to_action(response_action)                       
        #return response_action, self.episode_over, self.dialog_status
        return self.generate_diaact_from_nl(command), self.episode_over, self.dialog_status

    def generate_diaact_from_nl(self, string):
        """ Generate Dia_Act Form with NLU """
        sample_action = {}
        sample_action['diaact'] = {}
        sample_action['inform_slots'] = {}
        sample_action['request_slots'] = {}
        sample_action['turn'] = self.state['turn']
        
        if len(string) > 0:
            sample_action.update(self.nlu_model.generate_dia_act(string))
        
        sample_action['nl'] = string 
        return sample_action
