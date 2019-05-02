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

from simple_kerasinstance import SimpleCNN

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

class InfluencePoisoner:

    def __init__(self,
                 dataset,
                 model,
                 model_weights_path = None,
                 bottleneck_features_path = None,
                 reduced_train_size = None,
                 num_test_to_poison = 1,
                 num_train_to_use = 10,
                 step_size = 0.01,
                 num_iters = 100,
                 bounding_box_radius = 0.05,
                 bottleneck_layer = -1
                ):

        #Dataset, model etc.
        self.dataset = dataset
        self.model = model
        self.model_weights_path = model_weights_path
        self.bottleneck_features_path = bottleneck_features_path
        if reduced_train_size == None:
            self.reduced_size = len(dataset[0])
        else:
            self.reduced_size = reduced_train_size

        #Attack parameters
        self.num_test_to_poison = num_test_to_poison
        self.num_train_to_use = num_train_to_use
        self.step_size = step_size
        self.num_iters = num_iters
        self.bounding_box_radius = bounding_box_radius
        self.bottleneck_layer = bottleneck_layer

        #Tensorflow session
        self.sess = K.get_session()


    def poison_dataset(self):
        """
        Completes the influence poisoning attack on self.model
        trained on self.dataset for the first self.num_test_to_poison
        correctly classified points in the test set. Each targeted poisoning attack
        is allowed to change self.num_train_to_use examples in the train set.

        Returns:
            Poisoned X_points
            Predictions after poisoning
            Confidences before poisoning
            Confidences after poisoning
        """
        x_train, y_train, x_test, y_test = self.dataset

        #Train the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if self.model_weights_path == None:
            print('Training the model')
            self.model.fit(x_train, y_train, epochs=5)
            v = self.model.evaluate(x_test, y_test)
            print(v)
        else:
            self.model.load_weights(self.model_weights_path)

        #Compute Bottleneck features: Note that we use the full train set to compute the bottleneck features
        train_bottleneck_features = None
        test_bottleneck_features = None

        #Reduce train set size before training top model
        x_train = x_train[:self.reduced_size]
        y_train = y_train[:self.reduced_size]

        #TODO: Will have to make the indexing here more generic for
        #other models
        if self.bottleneck_features_path == None:
            print("COMPUTING TRAIN BOTTLENECK FEATURES")
            train_bottleneck_features = []
            for k in range(10):
                print("ITER ", k)
                bottleneck_features = compute_bottleneck_features(self.model, sess, x_train[1000*k:1000*(k+1)], -1)
                train_bottleneck_features.append(bottleneck_features)
            train_bottleneck_features = np.vstack(train_bottleneck_features)

            print("COMPUTING TEST BOTTLENECK FEATURES")
            test_bottleneck_features = compute_bottleneck_features(self.model, sess, x_test, -1)

        else:
            train_path = os.path.join(self.bottleneck_features_path, 'train_bottleneck_features.npy')
            test_path = os.path.join(self.bottleneck_features_path, 'test_bottleneck_features.npy')
            train_bottleneck_features = np.load(train_path)[:self.reduced_size]
            test_bottleneck_features = np.load(test_path)

        #Top model stuff: 
        lamb = 1
        top_model = construct_top_model(512, 1, "binary_crossentropy", True, lamb)
        train_top_model(top_model, train_bottleneck_features, y_train, lamb)
        sync_top_model_to_full_model(top_model, self.model)

        #Correct indices
        preds = top_model.predict(test_bottleneck_features)
        rounded_preds = np.round(preds)
        correct_indices = np.where(np.logical_and((rounded_preds.flatten() == y_test), (y_test == 0), (preds.flatten() > 0.25)))[0]

        confidences_before = []
        confidences_after = []

        #Finally actually perform the poisoning for each model
        for idx, test_index_to_flip in enumerate(correct_indices[0:self.num_test_to_poison]):
            z_bottleneck_test = [(test_bottleneck_features[test_index_to_flip], y_test[test_index_to_flip])]

            confidence_before = top_model.predict(np.array([z[0] for z in z_bottleneck_test]))
            confidences_before.append(confidence_before)

            #Sort the train indices according to influence scores
            grad_norms = grad_influence_wrt_input(self.model, self.sess, z_bottleneck_test, x_train, train_bottleneck_features, y_train, lamb, print_every=500)
            sorted_indices = list(reversed(np.argsort(grad_norms)))

            _ , confidence_after = data_poisoning_attack(self.model, self.sess, z_bottleneck_test, x_train, train_bottleneck_features, y_train, sorted_indices[:self.num_train_to_use], lamb, self.step_size, self.num_iters, self.bounding_box_radius, self.bottleneck_layer)
            confidences_after.append(confidences_after)

        return confidences_before, confidences_after
