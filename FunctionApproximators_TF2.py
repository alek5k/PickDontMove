from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np

## RELEVANT MODEL FUNCTIONS
# - fit
# - evaluate
# - predict
# - train_on_batch

class ValueEstimator:
    def __init__(self, input_shape, hidden_size=30, learning_rate=0.01):
        '''
        input_shape: input shape of the first layer. this should be the shape of the observation space.
        learning_rate: learning rate used in the optimizer
        '''
        self.model = self._build_model(input_shape, hidden_size, learning_rate)
        
    def predict(self, state):
        '''
        state: an observation of the state space
        '''
        return self.model.predict(state)
                                  
    def update(self, state, target):
        '''
        state: an observation of the state space
        target: value the output of the value function is compared to for loss
        '''
        # # we could call evaluate just to see what kind of loss we would get
        # # loss = self.model.evaluate(reshaped_flat_observation, td_target, batch_size=1)
        # history = self.model.fit(state, target, batch_size=1, verbose=0)
        history = self.model.train_on_batch(state, target)
        return history

    def _build_layers(self, inputs, hidden_size):
        '''
        Takes in input layer and builds all the intermediate layers up to and including the output
        '''
        x = layers.Dense(hidden_size, activation='relu')(inputs)
        x = layers.Dense(1)(x)
        return x
    
    def _build_model(self, input_shape, hidden_size, learning_rate):
        inputs = keras.Input(shape=input_shape)
        outputs = self._build_layers(inputs, hidden_size)
        model = keras.Model(inputs=inputs, outputs=outputs, name="ValueEstimator")
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.summary()
        return model
    
    
    
class PolicyEstimator:
    def __init__(self, 
                 input_shape, 
                 num_picker_actions, 
                 num_agv_actions,
                 num_pickers=1, 
                 num_agvs=2, 
                 hidden_size=30, 
                 learning_rate=0.01,
                 print_intermediate_values=False):
        '''
        input_shape: input shape of the first layer. this should be the shape of the observation space.
        num_picker_actions: the size of the pickers discrete action space
        num_agv_actions: the size of the AGVs discrete action space
        num_pickers: number of pickers in the system.
        num_agvs: number of agvs in the system.
        hidden_size: dictates the size of the densely connected hidden layers in the common network
        learning_rate: learning rate used in the optimizer
        '''
        
        self.selected_action_inputs = None
        self.num_pickers = num_pickers
        self.num_agvs = num_agvs
        self.print_intermediate_values = print_intermediate_values
        
        self.model = self._build_model(
            input_shape = input_shape, 
            num_picker_actions = num_picker_actions,
            num_agv_actions = num_agv_actions,
            num_pickers = num_pickers,
            num_agvs = num_agvs,
            hidden_size = hidden_size,
            learning_rate = learning_rate)
    
    
    def predict(self, state):
        '''
        state: an observation of the state space
        '''
        return self.model.predict(state)
                                  
    def update(self, state, target, actions):
        '''
        state: an observation of the state space
        target: a single value target for loss
        actions: selected discrete actions for each head, given as a list. eg. [picker0, agv1, agv2] = [0, 5, 6]
        '''
        # need a target for each agv and picker
        targets = [np.array(target) for _ in range(self.num_agvs + self.num_pickers)]
        self._set_actions_for_loss(actions)
        # history = self.model.fit(state, targets, batch_size=1, verbose=0)
        history = self.model.train_on_batch(state, targets)
        return history
    
    def _set_actions_for_loss(self, values):
        '''
        Sets the tensorflow Variables to the values provided
        '''
        assert(self.selected_action_inputs is not None) # has been initialized
        assert(len(self.selected_action_inputs) == len(values))
        
        for i in range(len(self.selected_action_inputs)):
            tf.keras.backend.set_value(self.selected_action_inputs[i], values[i])    
    
    
    
    def _build_agv_layers(self, inputs, num_actions):
        '''
        inputs: The input layers
        num_actions: number of possible actions in a discrete action space. This dictates the output size.
        '''
        _inputs = inputs
        if self.print_intermediate_values: _inputs = tf.keras.backend.print_tensor(_inputs, 'agv input')
        x = layers.Dense(num_actions, activation='softmax')(_inputs)
        # note: no softmax shaping of output, as we want to just preserve the probability distribution
        # perhaps if our action spaces get too close to zero we could try softmax this?
        return x
    
    
    def _build_picker_layers(self, inputs, num_actions):
        '''
        inputs: The input layers
        num_actions: number of possible actions in a discrete action space. This dictates the output size.
        '''
        _inputs = inputs
        if self.print_intermediate_values: _inputs = tf.keras.backend.print_tensor(_inputs, 'picker input')
        x = layers.Dense(num_actions, activation='softmax')(_inputs) 
        return x
       

    # # https://stackoverflow.com/questions/43818584/custom-loss-function-in-keras
    # # https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
    # # https://stackoverflow.com/questions/53440551/producing-a-softmax-on-two-channels-in-tensorflow-and-keras
    def _custom_loss_wrapper(self, selected_action):
        '''
        selected_action: the action selected from a discrete action space.
        
        The policy estimator outputs a probability distribution of actions.
        An action is selected OUTSIDE the policy estimator.
        We need a way to calculate the loss based on actions selected outside the policy estimator.
        '''
        def _custom_loss(y_true, y_pred):
            '''
            y_true: the action that is selected by the RL algo.
            y_pred: the NN's output probability distribution of actions. Shape: (13)
            '''
            action_probs_squeezed = tf.keras.backend.squeeze(y_pred, axis=0) # go from shape (None, 13) to (13)
            if self.print_intermediate_values: action_probs_squeezed = tf.keras.backend.print_tensor(action_probs_squeezed, 'action_probs_squeezed')
            
            _selected_action = selected_action
            if self.print_intermediate_values: _selected_action = tf.keras.backend.print_tensor(selected_action, 'selected_action')
            
            picked_action_probability = tf.keras.backend.gather(action_probs_squeezed, _selected_action) # softmax action probability actually selected
            if self.print_intermediate_values: picked_action_probability = tf.keras.backend.print_tensor(picked_action_probability, 'picked_action_probability')
            
            log_component = -tf.keras.backend.log(picked_action_probability)
            if self.print_intermediate_values: log_component = tf.keras.backend.print_tensor(log_component, 'log_component')
            
            _y_true = y_true
            if self.print_intermediate_values: _y_true = tf.keras.backend.print_tensor(_y_true, 'y_true')
                
            loss = log_component * _y_true

            return loss
        return _custom_loss

   
    def _build_model(self, input_shape, num_picker_actions, num_agv_actions, num_pickers, num_agvs, hidden_size, learning_rate):
        '''
        Creates a model.
        A neural network head will be created for each picker and AGV.
        Hidden size dictates the size of the densely connected hidden layers in the common network.
        '''
        
        inputs = keras.Input(shape=input_shape)
        
        # BUILD COMMON LAYERS
        common_layer_output = layers.Dense(hidden_size, activation='relu')(inputs)
        
        
        output_heads = []
        loss_functions = []
        self.selected_action_inputs = []
        
        # BUILD PICKER HEAD AND ACTION INPUTS FOR LOSS
        for picker_id in range(num_pickers):
            head = self._build_picker_layers(common_layer_output, num_picker_actions)
            output_heads.append(head) 
            # action_probs = layers.Activation('softmax')(head) # softmax over the outputs of NN to get probability distribution
            
            # provide a way to input an action before calculating loss
            action_input = tf.keras.backend.variable(0, dtype=tf.int32, name="action_picker{}".format(picker_id))
            self.selected_action_inputs.append(action_input)
            
            loss_functions.append(self._custom_loss_wrapper(action_input))
            
            
        # BUILD AGV HEAD AND ACTION INPUTS FOR LOSS  
        for agv_id in range(num_agvs):
            head = self._build_agv_layers(common_layer_output, num_agv_actions)
            output_heads.append(head) 
            # action_probs = layers.Activation('softmax')(head) # this is now done inside each head.
            
            # provide a way to input an action before calculating loss
            action_input = tf.keras.backend.variable(0, dtype=tf.int32, name="action_agv{}".format(agv_id))
            self.selected_action_inputs.append(action_input)   
            
            loss_functions.append(self._custom_loss_wrapper(action_input))
            

            
        model = keras.Model(inputs=inputs, outputs=output_heads, name="PolicyEstimator")
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=loss_functions)
        model.summary()
        return model
            