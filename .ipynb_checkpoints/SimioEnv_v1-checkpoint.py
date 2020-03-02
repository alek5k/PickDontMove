import zmq
import json
import numpy as np

import gym
from gym import spaces

class SimioPickDontMoveEnv(gym.Env):
    def __init__(self, num_locations=8, num_pickers=1, num_agvs=2, order_window_size=5, max_order_qty=100, log_output=False, log_end_episode_only=False):
        
        self.num_pickers = num_pickers
        self.num_agvs = num_agvs
        self.num_locations = num_locations
        # self.locations = ["PreparationStation", "DeliveryStation", "Bay"] + ["Location"+str(x+1) for x in range(num_locations)]

        all_action_spaces = []
        all_pickers_observation_space = ()
        all_agvs_observation_space = ()
        
        
        self.picker_valid_actions = 5+num_locations
        picker_action_space = [self.picker_valid_actions]
        picker_observation_space = ()
        picker_observation_space += ( spaces.Discrete(3), ) # Picker Current Action
        picker_observation_space += ( spaces.Discrete(3+num_locations), ) # Picker Location
        for x in range(num_pickers):
            all_action_spaces.extend(picker_action_space)
            all_pickers_observation_space += picker_observation_space
        
        self.agv_valid_actions = 4+num_locations
        agv_action_space = [self.agv_valid_actions] # Action      ASSIGNORDER [4+num_locations, 5]
        agv_observation_space = ()
        agv_observation_space += ( spaces.Discrete(3), ) # AGV Current Action
        agv_observation_space += ( spaces.Discrete(3+num_locations), ) # AGV Location
        agv_observation_space += ( spaces.MultiDiscrete([max_order_qty for z in range(num_locations)]), ) #AGV1 RemainingOrderQty
        for x in range(num_agvs):
            all_action_spaces.extend(agv_action_space)
            all_agvs_observation_space += agv_observation_space

        # orderpool_observation_space = ()
        # for x in range(order_window_size):
        #     orderpool_observation_space += (spaces.MultiDiscrete([max_order_qty for z in range(num_locations)]),)

        self.action_space = spaces.MultiDiscrete(all_action_spaces)
        self.observation_space = spaces.Tuple(all_pickers_observation_space + all_agvs_observation_space) #  + orderpool_observation_space
        
        self.picker_action_space = spaces.MultiDiscrete(picker_action_space)
        self.agv_action_space = spaces.MultiDiscrete(agv_action_space)
        self.picker_observation_space = spaces.Tuple(picker_observation_space)
        self.agv_observation_space = spaces.Tuple(agv_observation_space)
        
        # print(self.action_space)
        # print(self.observation_space)
        # self.observation_space = spaces.Dict({
        #     'pickers': spaces.Tuple(picker_observation_space),
        #     'agvs': spaces.Tuple(agv_observation_space),
        #     'orders': spaces.Tuple(orderpool_observation_space)
        # })

        # self.nA = self.action_space.n
        # self.nS = self.observation_space.n

        self.log_output = log_output
        self.log_end_episode_only = log_end_episode_only
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        #self.socket.setsockopt(zmq.RCVTIMEO, 15000)
        self.socket.connect("tcp://localhost:5000")

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
        state_variables['PickerCurrentAction'] = message['PickerCurrentAction']
        state_variables['AGV1Location'] = message['AGV1Location']
        state_variables['AGV1CurrentAction'] = message['AGV1CurrentAction']
        state_variables['AGV1RemainingOrderQuantities'] = message['AGV1RemainingOrderQuantities']
        state_variables['AGV2Location'] = message['AGV2Location']
        state_variables['AGV2CurrentAction'] = message['AGV2CurrentAction']
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


        
    def steprandom(self):
        return self.step(self.action_space.sample())

    
    def step(self, action):
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
        self.socket.send_json({
            'PickerAction': action[0],
            'AGV1Action': action[1],
            'AGV2Action': action[2]
        })
        message = self.socket.recv_json()
        return self.parse_message(message)

    def finalize(self):
        self.socket.send_json({'IsNoOp': True})
