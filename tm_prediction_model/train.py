import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

data_dir = '../data/' # data directory

# labels for inputs and outputs
inputs_0d = ['bt', 'ip', 'pinj', 'tinj', 'R0_EFITRT1', 'kappa_EFITRT1', 'tritop_EFIT01', 'tribot_EFIT01', 'gapin_EFIT01', 'ech_pwr_total', 'EC.RHO_ECH'] # at t+dt
inputs_1d = ['thomson_density_mtanh_1d', 'thomson_temp_mtanh_1d', '1/qpsi_EFITRT1', 'pres_EFIT01', 'cer_rot_csaps_1d'] # at t
outputs = ['betan_EFITRT1', 'tm_label'] # at t+dt

n_trial = 5 # number of ensemble models
jump = 1 # down-sampling (1: no downsample)
val_fraction = 0.2 # validation set fraction
test_fraction = 0.1 # test set fraction

def preprocess(data_dir):
    x0 = np.load(data_dir + 'x0.npy')
    x1 = np.load(data_dir + 'x1.npy')
    y = np.load(data_dir + 'y.npy')
    z = np.load(data_dir + 'z.npy')
    return x0, x1, y, z

def idx_split(n, test_size, random=True, random_state=0):
    idx = np.arange(n)
    if random:
        np.random.seed(random_state)
        np.random.shuffle(idx)
    n_train = int(n * (1 - test_size))
    return idx[:n_train], idx[n_train:]

if __name__ == '__main__':
    # Preprocess
    x0, x1, y, z = preprocess(data_dir)
    idx_a, idx_b = idx_split(len(y), test_size = test_fraction, random = False)
    x1, x1_test, x0, x0_test, y, y_test = x1[idx_a], x1[idx_b], x0[idx_a], x0[idx_b], y[idx_a], y[idx_b]
    #testshots = np.array(list(set(z[idx_b])), dtype=np.int64)
    #np.save('testshots.npy', testshots)

    # Set weighted loss
    neg, pos = np.bincount(y[:, 1].astype(int))
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    for seed in range(n_trial):
        # Set random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        idx_a, idx_b = idx_split(len(y), test_size = val_fraction, random_state = seed)
        x1_train, x1_val, x0_train, x0_val, y_train, y_val = x1[idx_a], x1[idx_b], x0[idx_a], x0[idx_b], y[idx_a], y[idx_b]

        # Balance
        idx_pos = (y_train[:, 1] == 1)
        x0_pos, x1_pos, y_pos = x0_train[idx_pos], x1_train[idx_pos], y_train[idx_pos]
        for _ in range(neg // pos - 1):
            x0_train = np.append(x0_train, x0_pos, axis = 0)
            x1_train = np.append(x1_train, x1_pos, axis = 0)
            y_train = np.append(y_train, y_pos, axis = 0)
        
        # Shuffle data
        idx = np.arange(len(y_train))
        np.random.shuffle(idx)
        x0_train, x1_train, y_train = x0_train[idx], x1_train[idx], y_train[idx]

        # Build model
        input_shape1, input_shape0, output_shape = x1.shape[1:], x0.shape[1:], y.shape[1]
        input0 = keras.layers.Input(shape = input_shape0)
        input1 = keras.layers.Input(shape = input_shape1)
        
        y1 = keras.layers.BatchNormalization()(input1)
        y1 = keras.layers.Conv1D(16, 3, activation = 'sigmoid')(y1)
        y1 = keras.layers.MaxPool1D(2)(y1)
        y1 = keras.layers.BatchNormalization()(y1)
        y1 = keras.layers.Conv1D(32, 3, activation = 'sigmoid')(y1)
        y1 = keras.layers.MaxPool1D(2)(y1)
        y1 = keras.layers.BatchNormalization()(y1)
        y1 = keras.layers.Flatten()(y1)
        y1 = keras.layers.Dense(32, activation = 'sigmoid')(y1)
        y1 = keras.layers.BatchNormalization()(y1)
        y1 = keras.layers.Dense(4, activation = 'sigmoid')(y1)

        y2 = keras.layers.Concatenate()([y1, input0])
        y2 = keras.layers.BatchNormalization()(y2)
        y2 = keras.layers.Dense(64, activation = 'sigmoid')(y2)
        y2 = keras.layers.BatchNormalization()(y2)
        y2 = keras.layers.Dense(32, activation = 'sigmoid')(y2)
        y2 = keras.layers.BatchNormalization()(y2)
        y2 = keras.layers.Dropout(0.2)(y2)
        y2 = keras.layers.Dense(output_shape, activation = 'linear')(y2)

        model = keras.Model(inputs = [input0, input1], outputs = [y2[:,0], y2[:,1]])
        model.summary()

        # Compile model
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss = [
                tf.keras.losses.MeanSquaredError(), # for betan
                tf.keras.losses.BinaryCrossentropy(from_logits=True) # for tm
            ],
            metrics = [f1_m]
        )
        callbacks = [EarlyStopping(monitor=f'val_tf.__operators__.getitem_{2*seed+1}_f1_m', mode='max', patience=10, restore_best_weights=True)]

        # Train and save model
        hist = model.fit([x0_train, x1_train], [y_train[:,0], y_train[:,1]], batch_size=512, epochs=200, callbacks=callbacks, validation_data=([x0_val, x1_val], [y_val[:,0], y_val[:,1]]), verbose=2) #class_weight=class_weight)
        model.save(f'best_model_{seed}')
        
        # Predict result
        _, yy_train = model.predict([x0_train, x1_train])
        _, yy_val = model.predict([x0_val, x1_val])
        _, yy_test = model.predict([x0_test, x1_test])

        # ROC, AUC
        for y_true, y_pred, label in [[y_train[:,1], yy_train, 'train'], [y_val[:,1], yy_val, 'val'], [y_test[:,1], yy_test, 'test']]:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc_value = auc(fpr, tpr)

            fig = plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label=f'{label}_{seed} (area = {auc_value:.3f})')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title(f'ROC curve for {label}')
            plt.legend(loc='best')
            plt.savefig(f'ROC_{label}_{seed}.png')

