import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras import backend as K
# K.set_image_data_format('channels_first')
# K.set_learning_phase(1)

from utils.influence_helpers import influence_binary_top_model_explicit, data_poisoning_attack, compute_bottleneck_features
from utils.influence_helpers import grad_influence_wrt_input, construct_top_model, train_top_model, sync_top_model_to_full_model
from squeeze_net_keras_wdropout import SqueezeNetBinaryKeras

# from simple_kerasinstance import SimpleCNN

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import os

class InfluencePoisoner:

    def __init__(self,
                 dataset,
                 model,
                 model_weights_path = None,
                 train_model = True,
                 bottleneck_features_path = None,
                 reduced_train_size = None,
                 test_indices_to_flip = [0],
                 batch_poison = True,
                 num_train_to_use_list = [10],
                 step_size = 0.01,
                 num_iters = 100,
                 bounding_box_radius = 0.05,
                 bottleneck_layer = -1,
                 verbose = 1
                ):

        #Dataset, model etc.
        self.dataset = dataset
        self.model = model
        self.model_weights_path = model_weights_path
        self.train_model = train_model
        self.bottleneck_features_path = bottleneck_features_path
        if reduced_train_size == None:
            self.reduced_size = len(dataset[0])
        else:
            self.reduced_size = reduced_train_size

        #Attack parameters
        self.test_indices_to_flip = test_indices_to_flip
        self.batch_poison = batch_poison
        self.num_train_to_use_list = num_train_to_use_list
        self.step_size = step_size
        self.num_iters = num_iters
        self.bounding_box_radius = bounding_box_radius
        self.bottleneck_layer = bottleneck_layer

        self.verbose = verbose

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
        K.set_learning_phase(0)

        #Train the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if self.model_weights_path == None and self.train_model == True:
            print('Training the model')
            self.model.fit(x_train, y_train, epochs=5)
            v = self.model.evaluate(x_test, y_test)
            print(v)
        elif self.train_model == True:
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
                bottleneck_features = compute_bottleneck_features(self.model, self.sess, x_train[1000*k:1000*(k+1)], -1)
                train_bottleneck_features.append(bottleneck_features)
            train_bottleneck_features = np.vstack(train_bottleneck_features)

            print("COMPUTING TEST BOTTLENECK FEATURES")
            test_bottleneck_features = compute_bottleneck_features(self.model, self.sess, x_test, -1)

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

        confidences = {}

        z_bottleneck_test = [(test_bottleneck_features[t], y_test[t]) for t in self.test_indices_to_flip]

        confidence_before = top_model.predict(np.array([z[0] for z in z_bottleneck_test]))
        confidences[0] = confidence_before.flatten()

        #Delete top model to save space
        del top_model

        if self.batch_poison:
            #Poison entire batch at once

            #Sort the train indices according to influence scores
            grad_norms = grad_influence_wrt_input(self.model, self.sess, z_bottleneck_test, x_train, train_bottleneck_features, y_train, lamb, print_every=500)
            sorted_indices = list(reversed(np.argsort(grad_norms)))

            for num_train_to_use in self.num_train_to_use_list:
                print("POISONING", num_train_to_use, "POINTS")
                poisoned_points, output_history = data_poisoning_attack(self.model, self.sess, z_bottleneck_test, x_train, train_bottleneck_features, y_train, sorted_indices[:num_train_to_use], lamb, self.step_size, self.num_iters, self.bounding_box_radius, self.bottleneck_layer, verbose=self.verbose)
                confidences[num_train_to_use] = output_history[-1].flatten()
        else:
            z_by_n = []
            for z in z_bottleneck_test:
                #Poison each z individually
                z_confs = []

                #Sort the train indices according to influence scores
                grad_norms = grad_influence_wrt_input(self.model, self.sess, [z], x_train, train_bottleneck_features, y_train, lamb, print_every=500)
                sorted_indices = list(reversed(np.argsort(grad_norms)))

                for num_train_to_use in self.num_train_to_use_list:
                    print("POISONING", num_train_to_use, "POINTS")
                    poisoned_points, output_history = data_poisoning_attack(self.model, self.sess, [z], x_train, train_bottleneck_features, y_train, sorted_indices[:num_train_to_use], lamb, self.step_size, self.num_iters, self.bounding_box_radius, self.bottleneck_layer, verbose=self.verbose)
                    z_confs.append(output_history[-1][0][0])
                z_by_n.append(z_confs)

            for i, num_train_to_use in enumerate(self.num_train_to_use_list):
                confidences[num_train_to_use] = np.array([c[i] for c in z_by_n])

        return confidences


def __main__():

    x_train = np.load("data/ship_bike_x_train.npy").astype('float32')/255
    y_train = np.load("data/ship_bike_y_train.npy").astype('float32')

    x_test = np.load("data/ship_bike_x_test.npy").astype('float32')/255
    y_test = np.load("data/ship_bike_y_test.npy").astype('float32')

    randomize = np.arange(len(x_train))
    np.random.shuffle(randomize)
    x_train = x_train[randomize]
    y_train = y_train[randomize]

    ship_bike_dataset = (x_train, y_train, x_test, y_test)

    model = SqueezeNetBinaryKeras(input_shape=(256, 256, 3), drop_p = 0.4)
    model_weights_path ="models/imagenet_ship_bike_squeezenet_do04.h5"

    model1 = (0, "models/imagenet_ship_bike_squeezenet_do00.h5")
    model2 = (0.2, "models/imagenet_ship_bike_squeezenet_do02.h5")
    model3 = (0.4, "models/imagenet_ship_bike_squeezenet_do04.h5")
    model4 = (0.6, "models/imagenet_ship_bike_squeezenet_do06.h5")

    model_list = [model1, model2, model3, model4]
    results = []

    for mo in model_list:
        p , path = mo
        K.set_image_data_format('channels_last')

        model = SqueezeNetBinaryKeras(input_shape=(256, 256, 3), drop_p = p)
        sess = K.get_session()


        with sess.as_default():
            K.set_learning_phase(0)

            do_model = InfluencePoisoner(ship_bike_dataset, model,
                                model_weights_path =path,
                                num_train_to_use_list = [40],
                                bounding_box_radius = 1,
                                step_size = 0.1)

            results.append(do_model.poison_dataset())
        K.clear_session()
    return results


__main__()
