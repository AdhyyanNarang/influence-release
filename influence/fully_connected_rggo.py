
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet


class fully_connected_rggo(GenericNeuralNet):

    def __init__(self, input_dim, weight_decay, **kwargs):

        self.input_dim = input_dim
        self.weight_decay = weight_decay

        # Make Keras use the same session as Tensorflow to avoid duplicities.
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.model = self._build_model(input_shape)

        self.layer_names = [layer.name for layer in model._layers][1:]
        # First layer is automatically created and contains no weights

        super(fully_connected_rggo, self).__init__(**kwargs)


    def _build_model(self):
        """
        It's mandatory to name every layer
        """

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(self.input_dim,), name="dense1"))
        model.add(Dropout(0.5, name="dropout1"))
        model.add(Dense(512, activation='relu', name="dense2"))
        model.add(Dropout(0.5, name="dropout2"))
        model.add(Dense(1, activation='sigmoid', name="dense3"))



        """ TODO check if needed
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        """


    def get_all_params(self):
        """
        Required method. It's called in the base class' constructor
        """

        all_params = []

        for layer_n in self.layer_names:        

            # First block to try
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)   

            # This block should work, but I want to try first block
            #temp_layer = self.model.get_layer(layer_n)
            #all_params.append(temp_layer.weights)
            #all_params.append(temp_layer.bias)

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
        raise NotImplementedError


    def predictions(self, input_x):
        """
        Ensure input_x it's normalized before calling this method
        """

        return model.predict(input_x)


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