import tensorflow as tf
from gym_helpers import flatten_space_sample


class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, env, learning_rate=0.01, hidden_size=30, scope="policy_estimator"):
        
        with tf.variable_scope(scope):

            input_size = len(flatten_space_sample(env.observation_space.sample()))
            # 1 output for each picker, AGV
            # eg. [picker, agv1, agv2] = [13, 12, 12]
            output_sizes = list(env.action_space.nvec)
            num_outputs = len(output_sizes)
            full_output_size = sum(output_sizes)
            
            # this is a terrible hack. this needs to change if we change action spaces.
            picker_output_size = int(env.picker_action_space.nvec)
            agv_output_size = int(env.agv_action_space.nvec)
            num_pickers = env.num_pickers
            num_agvs = env.num_agvs
            
            self.state = tf.placeholder(tf.float32, [1, input_size], name="state") 
            # state = tf.placeholder(tf.float32, [None, input_size])
            
            self.actions = []
            
            for i in range(num_pickers):
                # action = tf.placeholder(tf.int32, [picker_output_size], name="picker{}_action".format(i))
                action = tf.placeholder(tf.int32, name="picker{}_action".format(i))
                self.actions.append(action)
            
            for i in range(num_agvs):
                # action = tf.placeholder(tf.int32, [agv_output_size], name="agv{}_action".format(i))
                action = tf.placeholder(tf.int32, name="agv{}_action".format(i))
                self.actions.append(action)
                
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            
            self.fully_connected1 = tf.contrib.layers.fully_connected(
                inputs=self.state, 
                num_outputs=hidden_size, 
                activation_fn=tf.nn.relu#, 
                #weights_initializer=tf.zeros_initializer,
                #biases_initializer=tf.zeros_initializer
            )
            
            
            self.output_layers = []
            self.losses = []
            self.action_probs = []
            self.picked_action_probs = []
            self.optimizers = []
            self.train_ops = []
            
            for i in range(len(output_sizes)):
                
                output_size = int(output_sizes[i])
                
                output_layer = tf.contrib.layers.fully_connected(
                    inputs = self.fully_connected1, 
                    num_outputs=output_size,
                    activation_fn=None#,# tf.nn.softmax
                    #weights_initializer=tf.zeros_initializer#,
                    #biases_initializer=tf.zeros_initializer
                )
                self.output_layers.append(output_layer)
                
                action_probs = tf.squeeze(tf.nn.softmax(output_layer))
                self.action_probs.append(action_probs)
                
                # Based on action selected (outside), get the action probability from softmax output
                picked_action_prob = tf.gather(action_probs, self.actions[i]) 
                self.picked_action_probs.append(picked_action_prob)
                
                # Loss and train op
                loss = -tf.log(picked_action_prob) * self.target
                self.losses.append(loss)

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.optimizers.append(optimizer)
                
                train_op = optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())
                self.train_ops.append(train_op)
        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, actions, sess=None):
        sess = sess or tf.get_default_session()
        
        # feed_dict = { self.state: state, self.target: target, self.actions: actions  
        feed_dict = { self.state: state, self.target: target }
        for i in range(len(actions)):
            feed_dict[self.actions[i]] = actions[i]
    
        train_ops_and_losses = [ [trainop, loss] for trainop, loss in zip(self.train_ops, self.losses)] 
        # print(train_ops_and_losses)
        # _, losses = sess.run(train_ops_and_losses, feed_dict)
        sess_run_output = sess.run(train_ops_and_losses, feed_dict)
        losses = [x[1] for x in sess_run_output]
        return losses

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, env, learning_rate=0.1, hidden_size=30, scope="value_estimator"):
        
        with tf.variable_scope(scope):
            
            input_size = len(flatten_space_sample(env.observation_space.sample()))
            
            self.state = tf.placeholder(tf.float32, [1, input_size], name="state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            
            self.fully_connected1 = tf.contrib.layers.fully_connected(
                inputs=self.state, 
                num_outputs=hidden_size, 
                activation_fn=tf.nn.relu#, 
                #weights_initializer=tf.zeros_initializer#,
                #biases_initializer=tf.zeros_initializer
            )
            
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs = self.fully_connected1, 
                num_outputs=1,
                activation_fn=None#,
                #weights_initializer=tf.zeros_initializer#,
                #biases_initializer=tf.zeros_initializer
            )

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss