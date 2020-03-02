import zmq
import json
import numpy as np

import gym
from gym import spaces


class SimioPickDontMoveEnv(gym.Env):
    def __init__(self, num_locations=8, num_pickers=1, num_agvs=2, max_order_qty=100, log_output=False, log_end_episode_only=False):
        '''
        This env interface uses simplified strategy where AGVs are not told where to go, only pickers.
        '''
        self.num_pickers = num_pickers
        self.num_agvs = num_agvs
        self.num_locations = num_locations

        all_action_spaces = []
        all_pickers_observation_space = ()
        all_agvs_observation_space = ()
        
        self.picker_valid_actions = 1+num_agvs  # do nothing, serve agv1, serve agv2,... serve agv N
        picker_action_space = [self.picker_valid_actions]
        picker_observation_space = ()
        picker_observation_space += ( spaces.Discrete(3+num_locations), ) # Picker Location
        for x in range(num_pickers):
            all_action_spaces.extend(picker_action_space)
            all_pickers_observation_space += picker_observation_space
        
        self.agv_valid_actions = num_locations
        agv_action_space = [self.agv_valid_actions] 
        agv_observation_space = ()
        agv_observation_space += ( spaces.Discrete(3+num_locations), ) # AGV Location
        agv_observation_space += ( spaces.MultiDiscrete([max_order_qty for z in range(num_locations)]), ) #AGV1 RemainingOrderQty
        for x in range(num_agvs):
            all_action_spaces.extend(agv_action_space)
            all_agvs_observation_space += agv_observation_space

        
            
            
        self.action_space = spaces.MultiDiscrete(all_action_spaces)
        self.observation_space = spaces.Tuple(all_pickers_observation_space + all_agvs_observation_space) #  + orderpool_observation_space
        
        self.picker_action_space = spaces.MultiDiscrete(picker_action_space)
        self.picker_observation_space = spaces.Tuple(picker_observation_space)
        self.agv_action_space = spaces.MultiDiscrete(agv_action_space)
        self.agv_observation_space = spaces.Tuple(agv_observation_space)
        
        self.log_output = log_output
        self.log_end_episode_only = log_end_episode_only
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        #self.socket.setsockopt(zmq.RCVTIMEO, 15000)
        self.socket.connect("tcp://localhost:5002")

    def flatten_state(self, state):
        flattened = []
        for key, value in state.items():
            if isinstance(value, list):
                flattened.extend(value)
            else:
                flattened.append(value)
        return flattened

    def parse_message(self, message):
        status = message['Status']
        episode_num = message['EpisodeNumber']
        reward = message['Reward']

        state_variables = {}
        state_variables['PickerLocation'] = message['PickerLocation']
        state_variables['AGV1Location'] = message['AGV1Location']
        state_variables['AGV1RemainingOrderQuantities'] = message['AGV1RemainingOrderQuantities']
        state_variables['AGV2Location'] = message['AGV2Location']
        state_variables['AGV2RemainingOrderQuantities'] = message['AGV2RemainingOrderQuantities']
        
        
        done = True if status > 0.1 else False # factoring in floating point error
        
        if self.log_output:
            if self.log_end_episode_only:
                if done:
                    print("Status: ", status, ", EpisodeNum: ", episode_num, ", Reward: ", reward, ", States: ", state_variables)
            else:
                print("Status: ", status, ", EpisodeNum: ", episode_num, ", Reward: ", reward, ", States: ", state_variables)
            
        
        return (self.flatten_state(state_variables), reward, done, None)
    
    def reset(self):
        message = self.socket.recv_json()
        state, reward, done, _ = self.parse_message(message)
        return state


        
    def steprandom(self, log=False):
        return self.step(self.action_space.sample(), log=log)

    
    def step(self, action, log=False):
        '''
        Takes in an action as a np.array with picker action, agv action.
        '''
        
        assert len(action) == len(self.action_space.nvec)
        
        #for a in action:
        #    assert a < env.action_space.nvec[a]
        
        # if isinstance(action, list):
        action = [int(x) for x in action]
        
        
        # Check action is valid
        # assert self.action_space.contains(action)
        # action = action.tolist() # cast from Int32 to int, otherwise json cant serialize :S

        # print(action)
        #print(type(action))

        # TODO: Update this code so it can support a variable number of pickers/AGVs
        senddict = {
            'PickerAction': action[0],
            'AGV1Action': action[1],
            'AGV2Action': action[2]
        }
        self.socket.send_json(senddict)
        if log:
            print("action: ", senddict)
        
        message = self.socket.recv_json()
        return self.parse_message(message)

    def finalize(self):
        self.socket.send_json({'IsNoOp': True})
