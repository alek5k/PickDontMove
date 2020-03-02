
# https://github.com/tensorflow/tensorflow/issues/33487


import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def sample(logits):
    noise = tf.random.uniform(tf.shape(logits))
    return tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

class ActorCritic(Model):
    def __init__(self, numActions):
        super(ActorCritic, self).__init__()

        # self.conv1 = Conv2D(32, 3, 1, padding='same', activation='relu')
        # self.conv2 = Conv2D(64, 3, 1, padding='same', activation='relu')
        # self.conv3 = Conv2D(64, 3, 1, padding='same', activation='relu')
        # self.flatten = Flatten()
        # self.d1 = Dense(512, activation='relu')

        self.conv1 = Conv2D(64, 3, 1, padding='same', activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')

        self.pi = Dense(numActions, activation=None)
        self.vf = Dense(1, activation=None)

    def call(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return x

    def step(self, x):
        x = self(x)

        value = self.vf(x)
        v = value[:, 0]
        actions = self.pi(x)
        a = sample(actions)

        return a, v

    def lastValues(self, x):
        x = self(x)
        value = self.vf(x)
        v = value[:, 0]
        return v

    def value(self, x):
        x = self(x)
        return self.vf(x)

    def actionValues(self, x):
        x = self(x)
        return self.pi(x)

    def both(self, x):
        x = self(x)
        actionValues = self.pi(x)
        values = self.vf(x)
        return actionValues, values


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

class Agent:
    def __init__(self, numActions,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4, alpha=0.99, epsilon=1e-5,
        gamma=0.95):

        self.model = ActorCritic(numActions)
        self.entropyFactor = ent_coef
        self.valueLossFactor = vf_coef
        self.maxGradNorm = max_grad_norm
        self.learningRate = lr
        self.decay = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=alpha, epsilon=epsilon)

    def train(self, states, rewards, actions, values):
        with tf.GradientTape() as tape:
            pisPred, valuesPred = self.model.both(states)

            advs = rewards - values
            # Get the probabilities of the chosen actions.
            negLogPolicy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=pisPred)
            pgLoss = tf.reduce_mean(advs * negLogPolicy)
            vfLoss = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(valuesPred), rewards) / 2.0)
            entropy = tf.reduce_mean(cat_entropy(pisPred))
            loss = pgLoss - entropy * self.entropyFactor + self.valueLossFactor * vfLoss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.maxGradNorm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradNorm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return pgLoss, vfLoss, entropy

    def save(self, savePath):
        tf.keras.models.save_model(self.model, savePath)
        print ('saved model', savePath)

    def load(self, loadPath):
        self.model = tf.keras.models.load_model(loadPath)
        print ('loaded model', loadPath)

    def getValues(self, state):
        # Neural network wants batches so wrap single state in an array to add dimension.
        return self.model.actionValues(np.asarray([state], dtype='float32'))

    def selectAction(self, state, validActions, randomRatio=-1):
        # Neural network wants batches so wrap single state in an array to add dimension.
        action, value = self.model.step(np.asarray([state], dtype='float32'))
        action = action[0] if len(action) else action
        value = value[0] if len(value) else value

        if randomRatio >= 0:
            randNum = random.uniform(0, 1)
            if randNum < randomRatio:
                action = np.random.choice(validActions)

        if validActions is not None and action not in validActions:
            action = np.random.choice(validActions)

        return action, value

    def learn(self, states, actions, rewards, dones, values):

        # Update / discount rewards. TODO - understand this..
        lastValues = self.model.lastValues(states).numpy()
        # discount/bootstrap off value fn
        for n, (rewards1, dones1, value1) in enumerate(zip(rewards, dones, lastValues)):
            rewards1 = rewards1.tolist()
            dones1 = dones1.tolist()
            if dones1[-1] == 0:
                rewards1 = discount_with_dones(rewards1 + [value1], dones1 + [0], self.gamma)[:-1]
            else:
                rewards1 = discount_with_dones(rewards1, dones1, self.gamma)
            rewards[n] = rewards1

        # Combine envs and steps with flatten.
        # State is already flattened / joined.
        rewards = rewards.flatten()
        actions = actions.flatten()
        values = values.flatten()

        return self.train(states, rewards, actions, values)