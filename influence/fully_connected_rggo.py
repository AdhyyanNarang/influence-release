
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

        # Make Keras use the same session as Tensorflow to avoid duplicities.
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.logits_tensor = None
        self.model = self._build_model(input_dim)

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

        self.layer_names = [layer.name for layer in model._layers if layer.name.startswith("dense")][1:]
        # First layer is automatically created and contains no weights, hence we don't need it here.
        # We need to reject dropout layers for the same reason

        return model


    def get_all_params(self):
        """
        Required method. It's called in the base class' constructor
        """

        all_params = []

        for layer_n in self.layer_names:        

            # First block to try
            #for var_name in ['weights', 'biases']:
            #    temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer_n, var_name))
            #    all_params.append(temp_tensor)

            # This block should work, but I want to try first block
            temp_layer = self.model.get_layer(layer_n)
            all_params.append(temp_layer.weights)
            all_params.append(temp_layer.bias)

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