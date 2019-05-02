import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras import backend as K
K.set_image_data_format('channels_first')
K.set_learning_phase(1)

from utils.influence_helpers import influence_binary_top_model_explicit, data_poisoning_attack, compute_bottleneck_features
from utils.influence_helpers import grad_influence_wrt_input, construct_top_model, train_top_model, sync_top_model_to_full_model
from poison_attack import InfluencePoisoner

from simple_kerasinstance import SimpleCNN

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#Config
input_shape = (3,32,32)
#SET TRAINED TO TRUE IF MODEL IS TRAINED AND SAVED
trained = True
features_computed = True

#Dataset preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
train_bird_frog = np.where(np.logical_or(y_train == 2, y_train == 6))
test_bird_frog = np.where(np.logical_or(y_test == 2, y_test == 6))
x_train = x_train[train_bird_frog] / 255.
y_train = y_train[train_bird_frog]
y_train[y_train == 6] = 1
y_train[y_train == 2] = 0
x_test = x_test[test_bird_frog] / 255.
y_test = y_test[test_bird_frog]
y_test[y_test == 6] = 1
y_test[y_test == 2] = 0

#SET RNG SEED
np.random.seed(32)

sess = K.get_session()
model = SimpleCNN(input_shape=input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dataset = (x_train, y_train, x_test, y_test)
model_weights_path = "models/cifar10_bird_frog_simple_cnn.h5"
bottleneck_features_path = "precomputed_features"
reduced_train_size = 1000

poison_instance = InfluencePoisoner(dataset = dataset, model = model, model_weights_path = model_weights_path, bottleneck_features_path = bottleneck_features_path, reduced_train_size = reduced_train_size)

confidence_before, conf_after = poison_instance.poison_dataset()
