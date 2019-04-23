#first train and save model (use LBFGS? sklearn solver?)
#implement the model loader
#choose a (correctly classified) test point, compute (and save) influence (as well as s_test values)
#visualize top helpful and harmful influencers from each class, as well as least helpful and harmful influencers from each class
#implement ncgs solver, plot against the exact influence
#try swapping out Hessian with Gauss-Newton approximation, plot against exact and ncgs
#implement poisoning

#note to self: poisoning is implemented by taking the derivative wrt x of the influence function (note how the differentials commute,
#and also it's defined as the change in influence as x changes, and hence is how the retraining loss changes)
#oh so it doesn't "exactly" commute, but note that we can still precompute the influence
#and then just take the gradient wrt the leftmost loss gradient (in the transposed equation). That's what they do.
#they actually are only taking one layer gradient approximation of influence (only parameter grads on top model, though propagated through to input)
#oh...so since they are just computing influence at the last layer, they only need the inverse hessian vector product at the last layer
#if the last layer is a modestly-sized softmax layer, the hessian can be computed exactly. Is solving the linear system explicitly faster than fmin_ncg?



import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K

from utils.influence_helpers import influence_binary_top_model_explicit, data_poisoning_attack, compute_bottleneck_features, grad_influence_wrt_input

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#SET RNG SEED
np.random.seed(32)

#SET TRAINED TO TRUE IF MODEL IS TRAINED AND SAVED
trained = True

#EXTRACT JUST 0/8 POINTS
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_08 = np.where(np.logical_or(y_train == 0, y_train == 8))
test_08 = np.where(np.logical_or(y_test == 0, y_test == 8))
x_train = x_train[train_08].reshape((-1, 28 * 28)) / 255.
y_train = y_train[train_08]
y_train[y_train == 8] = 1
x_test = x_test[test_08].reshape((-1, 28 * 28)) / 255.
y_test = y_test[test_08]
y_test[y_test == 8] = 1

#CONSTRUCT KERAS MODEL
lamb = 1
model = Sequential()
model.add(Dense(1, input_shape=(28 * 28,), activation="sigmoid", kernel_regularizer=regularizers.l2(lamb)))

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

if not trained:
    #TRAIN A LOGISTIC REGRESSOR USING SKLEARN/LBFGS
    from sklearn import linear_model
    C = 1.0 / (len(x_train) * lamb)        
    sklearn_model = linear_model.LogisticRegression(
        C=C,
        tol=1e-8,
        fit_intercept=True,
        solver='lbfgs',
        multi_class='multinomial',
        warm_start=True,
        max_iter=1000)
    sklearn_model.fit(x_train, y_train)

    #PARAMETER SURGERY
    model.layers[0].set_weights([sklearn_model.coef_.T, sklearn_model.intercept_])

    #VERIFY ACCURACY
    print(model.evaluate(x_test, y_test))

    #SAVE MODEL
    model.save_weights("models/mnist_08.h5")

#LOAD MODEL
model.load_weights("models/mnist_08.h5")



sess = K.get_session()

x_train = x_train[:500]
y_train = y_train[:500]

x_train_top = compute_bottleneck_features(model, sess, x_train, -1)
test_pts_top = compute_bottleneck_features(model, sess, x_train[:1], -1)
z_test_bottleneck_list = [(test_pts_top[0], y_train[0])]

grad_norms = grad_influence_wrt_input(model, sess, z_test_bottleneck_list, x_train, x_train_top, y_train, lamb, print_every=30)
sorted_indices = list(reversed(np.argsort(grad_norms)))

poisoned_points = data_poisoning_attack(model, sess, z_test_bottleneck_list, x_train, x_train_top, y_train, sorted_indices[:1], lamb, 0.2, 50, 0.2, -1)

#Visualize a poisoned point and the target point


poison_test = poisoned_points[0].reshape(28,28)
orig = x_train[0].reshape(28,28)
unpoisoned_test = x_train[sorted_indices[0]].reshape(28,28)

plt.imshow(unpoisoned_test, cmap="gray")
plt.show()
plt.imshow(orig, cmap="gray")
plt.show()
plt.imshow(poison_test, cmap="gray")
plt.show()

#Compute influence wrt first two train points by next three train points

influence_vals = influence_binary_top_model_explicit(model, sess,
                                                    [(x_train[0], y_train[0]), (x_train[1], y_train[1])], [(x_train[2], y_train[2]), (x_train[3], y_train[3]), (x_train[4], y_train[4])],
                                                    x_train, y_train, lamb)

print(influence_vals)




