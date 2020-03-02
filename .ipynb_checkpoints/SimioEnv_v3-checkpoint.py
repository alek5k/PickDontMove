import zmq
import json
import numpy as np

import gym
from gym import spaces

def flatten_list(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten_list(x))
    else:
        result.append(xs)
    return result


class SimioPickDontMoveEnv(gym.Env):
    def __init__(self, num_locations=8, max_order_qty=20, log_output=False, log_end_episode_only=False):
        ''' 
        This env interface uses simplified strategy where AGVs are not told where to go, only pickers.
        '''

        self.num_locations = num_locations
        
        
        self.picker_actions = 1 + num_agvs # picker: do nothing, serve agv1, serve agv2,... serve agv N
        self.agv_actions = num_locations # agv: location 1, location 2... location L
        
        action_space = [self.picker_actions]*num_pickers + [self.agv_actions]*num_agvs

        self.picker_observations = 3 + num_locations# picker: idle, pickup, delivery, location 1... location L
        self.agv_observations = 3 + num_locations # agv: idle, pickup, delivery, location 1...location L
        # agv 1 remaining order qty... agv N remaining order qty
        state_space = [self.picker_observations]*num_pickers + [self.agv_observations]*num_agvs + [max_order_qty]*num_locations*num_agvs
        

        self.action_space = spaces.MultiDiscrete(action_space)
        self.observation_space = spaces.MultiDiscrete(state_space)
        
        self.log_output = log_output
        self.log_end_episode_only = log_end_episode_only
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        #self.socket.setsockopt(zmq.RCVTIMEO, 15000)
        self.socket.connect("tcp://localhost:5002")


    def parse_message(self, message):
        status = message['Status']
        episode_num = message['EpisodeNumber']
        reward = message['Reward']

        state_variables = []
        state_variables.append(message['VehicleTypeRequesting'])
        state_variables.append(message['VehicleID'])
        state_variables.append(message['VehicleCurrentLocation'])
        state_variables.append(message['VehicleRemainingOrderQuantities'])
        state_variables = flatten_list(state_variables)
        
        assert(len(state_variables) == len(self.observation_space.nvec))
        
        done = True if status > 0.0001 else False # factoring in floating point error
        
        if self.log_output:
            if self.log_end_episode_only:
                if done:
                    print("Status: ", status, ", EpisodeNum: ", episode_num, ", Reward: ", reward, ", States: ", state_variables)
            else:
                print("Status: ", status, ", EpisodeNum: ", episode_num, ", Reward: ", reward, ", States: ", state_variables)
            
        
        return (state_variables, reward, done, None)
    
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
        # assert len(action) == len(self.action_space.nvec)
        # 
        # #for a in action:
        # #    assert a < env.action_space.nvec[a]
        # 
        # # if isinstance(action, list):
        # action = [int(x) for x in action]
        # 
        # 
        # # Check action is valid
        # # assert self.action_space.contains(action)
        # # action = action.tolist() # cast from Int32 to int, otherwise json cant serialize :S
# 
        # # print(action)
        # #print(type(action))
# 
        # # TODO: Update this code so it can support a variable number of pickers/AGVs
        senddict = {
            'VehicleAction': action,
        }
        self.socket.send_json(senddict)
        if log:
            print("Sent action: ", senddict)
        
        message = self.socket.recv_json()
        return self.parse_message(message)

    def finalize(self):
        self.socket.send_json({'IsNoOp': True})
