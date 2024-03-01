import os
import numpy as np
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.device("/cpu:0")

import keras
from tensorflow.keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.processors import WhiteningNormalizerProcessor

import myenv

np.random.seed(0)
tf.random.set_seed(0)

class MyProcessor(Processor):
    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, batch):
        return np.squeeze(batch, axis=1)

# Set environment
env = myenv.Env()
n_actions = env.action_space.shape[0]
x0_mean, x0_std = np.load('x0_mean.npy'), np.load('x0_std.npy')
x1_mean, x1_std = np.load('x1_mean.npy'), np.load('x1_std.npy')

# Actor
actor = keras.models.Sequential()
actor.add(keras.layers.Conv1D(16, 3, activation = 'tanh', input_shape = env.observation_space.shape))
actor.add(keras.layers.MaxPool1D(2))
actor.add(keras.layers.Conv1D(32, 3, activation = 'tanh'))
actor.add(keras.layers.MaxPool1D(2))
actor.add(keras.layers.Flatten())
actor.add(keras.layers.Dense(32, activation = 'tanh'))
actor.add(keras.layers.Dropout(0.2))
actor.add(keras.layers.Dense(n_actions, activation = 'tanh'))
actor.summary()

# Critic
action_input = keras.layers.Input(shape=(n_actions,), name='action_input')
observation_input = keras.layers.Input(shape = env.observation_space.shape, name='observation_input')
x = keras.layers.BatchNormalization()(observation_input)
x = keras.layers.Conv1D(16, 3, activation = 'tanh')(x)
x = keras.layers.MaxPool1D(2)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv1D(32, 3, activation = 'tanh')(x)
x = keras.layers.MaxPool1D(2)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation = 'tanh')(x)
x = keras.layers.Concatenate()([x, action_input])
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(32, activation = 'tanh')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(16, activation = 'tanh')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(1, activation = 'linear')(x)
critic = keras.Model(inputs = [action_input, observation_input], outputs = x)
critic.summary()

# Compile model
memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=.15, mu=0., sigma=.1)
myprocessor = MyProcessor(nb_inputs=n_actions)
agent = DDPGAgent(processor=myprocessor,
                  nb_actions=n_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=500,
                  random_process=random_process, gamma=.99, batch_size=64, target_model_update=1e-3)
agent.compile(optimizer=Adam(lr=1e-4), metrics=['mae'])

# Train and save model
agent.fit(env, nb_steps=100000, visualize=False, verbose=0)
#agent.save_weights('ddpg_weights.h5f', overwrite=True)
agent.actor.save('actor.h5', save_format='h5')
agent.test(env, nb_episodes=100, visualize=False)
