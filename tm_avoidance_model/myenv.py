import math, gym
import numpy as np
from keras import models

def f1_m(y_true, y_pred):
    return 1.0

def sigmoid(z):
    return 1/(1+math.e**(-z))

# Predictor setting
data_dir = '../data/'
predictor_dir = '../tm_prediction_model/'
n_predictor = 5

# Action setting
low_action, high_action = [-1, -1, -1], [1, 1, 1]
clip_action1 = [0, 0, 0]
clip_action2 = [10000, 1, 1]
threshold = 0.5

# Normalization
s_mean = np.array([3.977, 1.587, 0.5103, 23303., 42.93]) # ne, te, 1/q, pres, rot
s_std = np.array([2.764, 1.560, 0.4220, 35931., 58.47])
a_mean = np.array([4072.8761393229165, 0.6, 0.41])
a_std = np.array([3145.5935872395835, 0.31, 0.31])

class Env(gym.GoalEnv):
    def __init__(self):
        super(Env, self).__init__()
        # Load data and models
        self.x0, self.x1 = np.load(data_dir + 'x0.npy'), np.load(data_dir + 'x1.npy')
        self.x0_mean, self.x0_std = self.x0.mean(axis=0).astype(np.float32), self.x0.std(axis=0).astype(np.float32)
        self.x1_mean, self.x1_std = self.x1.mean(axis=0).astype(np.float32), self.x1.std(axis=0).astype(np.float32)
        self.predictors = [models.load_model(predictor_dir + f'best_model_{i}', custom_objects={'f1_m':f1_m}) for i in range(n_predictor)]

        # Save normalizing factors
        np.save('x0_mean.npy', self.x0_mean)
        np.save('x0_std.npy', self.x0_std)
        np.save('x1_mean.npy', self.x1_mean)
        np.save('x1_std.npy', self.x1_std)

        # Balance
        yy = np.mean([p.predict([self.x0, self.x1]) for p in self.predictors], axis=0)
        idx_pos = (sigmoid(yy[1]) > threshold)
        x0_pos, x1_pos = self.x0[idx_pos].copy(), self.x1[idx_pos].copy()
        for _ in range((len(idx_pos) - sum(idx_pos)) // sum(idx_pos) - 1):
            self.x0 = np.append(self.x0, x0_pos, axis=0)
            self.x1 = np.append(self.x1, x1_pos, axis=0)

        # Setting for RL
        self.action_space = gym.spaces.Box(
            low = np.array(low_action),
            high = np.array(high_action),
            dtype = np.float32
        )
        self.observation_space = gym.spaces.Box(
            low = -2 * np.ones_like(self.x1_mean), #self.x1_mean - 2 * self.x1_std,
            high = 2 * np.ones_like(self.x1_mean), #self.x1_mean + 2 * self.x1_std,
            dtype = np.float32
        )
        
        # Initialize
        self.episodes = 0
        self.reset()

    def reset(self):
        self.episodes += 1
        self.i_model = np.random.randint(n_predictor)
        self.idx = np.random.randint(len(self.x0))
        return (self.x1[self.idx] - s_mean) / s_std

    def step(self, action):
        # Take action
        action1 = np.clip(action * a_std + a_mean, clip_action1, clip_action2)
        x0_tmp, x1_tmp = self.x0[[self.idx]].copy(), self.x1[[self.idx]].copy()
        x0_tmp[0, 2] = action1[0]
        x0_tmp[0, 3] = min(1.0, action1[0] * self.x0_mean[3] / self.x0_mean[2])
        x0_tmp[0, 6] = action1[1]
        x0_tmp[0, 7] = action1[2]

        # Predict next step
        y = self.predictors[self.i_model].predict([x0_tmp, x1_tmp])
        betan, tearability = y[0][0], sigmoid(y[1][0])

        # Estimate reward
        if tearability < threshold:
            reward = betan
        else:
            reward = threshold - tearability
        print(self.episodes, action[0], betan, tearability, reward)
        return (self.x1[self.idx] - s_mean) / s_std, reward, True, {}

    def render(self, mode = 'human'):
        pass

    def close(self):
        pass
