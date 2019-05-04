# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras import regularizers

# Helper libraries
import numpy as np
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

"""
Define util functions
"""

def adv_evaluate(dataset, model_input, epsilon_input):
    x_train, y_train, x_test, y_test = dataset
    wrap = KerasModelWrapper(model_input)
    fgsm = FastGradientMethod(wrap)
    adv_x = fgsm.generate_np(x_test, eps = epsilon_input, clip_min = -2, clip_max = 2)
    preds_adv = model_input.predict(adv_x)
    eval_par = {'batch_size': 60000}
    test_loss, acc = model_input.evaluate(adv_x, y_test)
    return acc, adv_x, preds_adv

#Uses adversarial examples from cleverhans
def train_and_test(model_input, X_input, y_input, epochs = 2, eps_test = 0.2):
    model_input.fit(X_input, y_input, epochs = epochs)
    print('finished training')
    test_loss, test_acc = model_input.evaluate(test_images, test_labels)
    #Evaluation of normal model on adversarial test examples
    print('Test accuracy:', test_acc)
    #Evaluation of adversarially trained model
    acc, adv_x_np, preds = adv_evaluate(model_input, 0.2)
    print('Test accuracy on adversarial examples:', acc)


