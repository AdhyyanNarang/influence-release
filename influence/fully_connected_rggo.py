
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
        self.model = self._build_model(input_dim)
        self.logits_tensor = None

        # Initialize session
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

        # Make Keras use the same session as Tensorflow to avoid duplicities.
        K.set_session(self.sess)

        # We need to tell the parent class we already have a tf.Session
        kwargs['exist_tf_session'] = True

        super(Fully_connected_rggo, self).__init__(**kwargs)


    def _build_model(self, input_dim):
        """
        It's mandatory to name every layer
        Last layer must have a linear activation so we can obtain "logits" in the 'inference' method
        and the desired activation must be added in the 'predictions' method
        """

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


    def get_all_params(self):
        """
        Required method. It's called in the base class' constructor
        """

        # We need to extract the tf.Variables associated with the graph and flatten them (because this repository needs it)
        with tf.variable_scope('flattened_weights'):
            flatten_weights = [tf.reshape(self.model.get_layer(elem).weights[0], [-1], name=elem) for elem in self.layer_names]


        all_params = []

        for layer_n in self.layer_names:        

            # First block to try
            for var_name in ['flattened_weights', 'bias']:
                temp_tensor = self.sess.get_tensor_by_name("%s/%s:0" % (layer_n, var_name))
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


    def inference(self, input_x):
        logits_tensor = self.model(input_x)
        return logits_tensor


    def predictions(self, logits):
        preds = Activation('sigmoid', name="activation1")(logits)
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