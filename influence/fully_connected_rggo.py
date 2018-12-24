import math

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet


class Fully_connected_rggo(GenericNeuralNet):

    def __init__(self, input_dim, weight_decay, **kwargs):

        self.input_dim = input_dim
        self.weight_decay = weight_decay

        # Initialize session
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

        # Make Keras use the same session as Tensorflow to avoid duplicities.
        K.set_session(self.sess)

        # We need to tell the parent class we already have a tf.Session
        kwargs['exist_tf_session'] = True

        super(Fully_connected_rggo, self).__init__(**kwargs)


    def get_all_params(self):
        layer_names = ['hidden1', 'hidden2', 'softmax_linear']

        all_params = []
        for layer in layer_names:
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                all_params.append(temp_tensor)

        return all_params


    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def inference(self, input_x, hidden1_units=8, hidden2_units=8, output_units=2):
        """Build the model up to where it may be used for inference.
        Args:
            hidden1_units: Size of the first hidden layer.
            hidden2_units: Size of the second hidden layer.
            output_units: Size of the output layer. Must be the same size of num_classes. Beware of binary classification,
            even in that case we need to have 2 neurons, because we are going to do a softmax classification.
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """

        # Hidden 1
        with tf.variable_scope('hidden1'):

            weights = variable_with_weight_decay(
                'weights',
                [self.input_dim * hidden1_units],
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)
            biases = variable(
                'biases',
                [hidden1_units],
                tf.constant_initializer(0.0))
            weights_reshaped = tf.reshape(weights, [self.input_dim, hidden1_units])
            hidden1 = tf.nn.relu(tf.matmul(input_x, weights_reshaped) + biases)

            """
            # Variable declaration
            weights = tf.Variable(
                tf.truncated_normal([self.input_dim, hidden1_units],
                                    stddev=1.0 / math.sqrt(float(self.input_dim))),
                                    name='weights',
                                    dtype=tf.float32)
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                    name='biases',
                                    dtype=tf.float32)

            # Operation
            hidden1 = tf.nn.relu(tf.matmul(input_x, weights) + biases)

            # We need to extract the tf.Variables associated with the graph and flatten them (because this repository needs it)
            weights_flattened = tf.reshape(weights, [-1], name='flattened_weights')
            weight_decay = tf.multiply(tf.nn.l2_loss(weights_flattened), self.weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            """

        # Hidden 2
        with tf.variable_scope('hidden2'):

            weights = variable_with_weight_decay(
                'weights',
                [hidden1_units * hidden2_units],
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)
            biases = variable(
                'biases',
                [hidden2_units],
                tf.constant_initializer(0.0))
            weights_reshaped = tf.reshape(weights, [hidden1_units, hidden2_units])
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_reshaped) + biases)

            """
            # Variable declaration
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                                    name='weights',
                                    dtype=tf.float32)
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                    name='biases',
                                    dtype=tf.float32)

            # Operation
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

            # We need to extract the tf.Variables associated with the graph and flatten them (because this repository needs it)
            weights_flattened = tf.reshape(weights, [-1], name='flattened_weights')
            weight_decay = tf.multiply(tf.nn.l2_loss(weights_flattened), self.weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            """

        # Linear
        with tf.variable_scope('softmax_linear'):

            weights = variable_with_weight_decay(
                'weights',
                [hidden2_units * output_units],
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)
            biases = variable(
                'biases',
                [output_units],
                tf.constant_initializer(0.0))
            weights_reshaped = tf.reshape(weights, [hidden2_units, output_units])
            logits = tf.matmul(hidden2, weights_reshaped) + biases


            """
            # Variable declaration
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, output_units],
                                    stddev=1.0 / math.sqrt(float(hidden2_units))),
                                    name='weights',
                                    dtype=tf.float32)
            biases = tf.Variable(tf.zeros([output_units]),
                                    name='biases',
                                    dtype=tf.float32)

            # Operation
            logits = tf.matmul(hidden2, weights) + biases

            # We need to extract the tf.Variables associated with the graph and flatten them (because this repository needs it)
            weights_flattened = tf.reshape(weights, [-1], name='flattened_weights')
            weight_decay = tf.multiply(tf.nn.l2_loss(weights_flattened), self.weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            """

        return logits


    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds


# TODO 
# Reescribir fijandome en BinaryLogisticRegressionWithLBFGS y la diferencia con la clase GenericNeuralNet
"""
def get_influence_on_test_loss(self, test_indices, train_idx, 
        approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
        loss_type='normal_loss',
        ignore_training_error=False,
        ignore_hessian=False
        ):
"""