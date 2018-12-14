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

        #self.model = self._build_model(input_dim)
        #self.logits_tensor = None

        super(Fully_connected_rggo, self).__init__(**kwargs)

    """
    def _build_model(self, input_dim):
        
        It's mandatory to name every layer
        Last layer must have a linear activation so we can obtain "logits" in the 'inference' method
        and the desired activation must be added in the 'predictions' method
        

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(input_dim,), name="dense1"))
        model.add(Dropout(0.5, name="dropout1"))
        model.add(Dense(512, activation='relu', name="dense2"))
        model.add(Dropout(0.5, name="dropout2"))
        model.add(Dense(1, activation='linear', name="dense3"))
        #model.add(Activation('sigmoid', name="activation1")) # This layer is moved to 'predictions' method

        self.layer_names = [layer.name for layer in model.layers if layer.name.startswith("dense")]
        # We need to reject dropout layers because doesn't contain weights, hence we don't need them here.

        return model
    """

    def get_all_params(self):
        layer_names = ['hidden1', 'hidden2', 'softmax_linear']

        all_params = []
        for layer in layer_names:
            for var_name in ['flattened_weights', 'biases']:
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


    def inference(self, input_x, hidden1_units=8, hidden2_units=8, output_units=1):
        """Build the MNIST model up to where it may be used for inference.
        Args:
            images: Images placeholder, from inputs().
            hidden1_units: Size of the first hidden layer.
            hidden2_units: Size of the second hidden layer.
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """

        # Hidden 1
        with tf.name_scope('hidden1'):

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

        # Hidden 2
        with tf.name_scope('hidden2'):

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

        # Linear
        with tf.name_scope('softmax_linear'):

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