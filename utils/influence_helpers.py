import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy as sp
from scipy.linalg import solve_triangular
from sklearn import linear_model


class DataLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.curr_index = 0
        self.shuffle_data()

    def shuffle_data(self):
        np.random.seed(0)
        shuffle_indices = list(range(len(self.X)))
        np.random.shuffle(shuffle_indices)
        self.X = self.X[shuffle_indices]
        self.y = self.y[shuffle_indices]

    def next_batch(self, batch_size):
        if self.curr_index + batch_size < len(self.X):
            next_input = self.X[self.curr_index : self.curr_index + batch_size]
            next_output = self.y[self.curr_index : self.curr_index + batch_size]
            self.curr_index += batch_size
            return next_input, next_output
        else:
            self.curr_index = 0
            self.shuffle_data()
            return self.next_batch(batch_size)


def get_tensors_from_keras_model(model):
    input_ph = K.placeholder(shape=model.layers[0].input_shape, dtype="float32")
    output_ph = K.placeholder(shape=model.layers[-1].output_shape, dtype="float32")
    model_output = model(input_ph)
    loss_fxn = getattr(K, model.loss)
    #Loss tensor will not include regularization, which is good (we only wanted regularization on the Hessian)
    loss_tensor = loss_fxn(output_ph, model_output)
    param_tensors = []
    for w in model.trainable_weights:
        param_tensors.append(w)
    return input_ph, output_ph, loss_tensor, param_tensors


def hessian_vector_product(ys, xs, v):
    grads = gradients(ys, xs)
    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    grads_with_none = gradients(elemwise_products, xs)
    return_grads = [
        grad_elem if grad_elem is not None \
        else tf.zeros_like(x) \
        for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads


def populate_feed_dict(input_ph, input_vals, output_ph, output_vals, v_ph, v_vals):
    feed_dict = {input_ph: input_vals, output_ph: output_vals}
    for i in range(len(v_ph)):
        feed_dict[v_ph[i]] = v_vals[i]
    return feed_dict


def hessian_binary_crossentropy(weights, bias, X, y, reg):
    if bias is not None:
        param_vec = np.vstack([weights, bias])
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    else:
        param_vec = weights
    s = sp.special.expit(X.dot(param_vec))
    omega = s * (1 - s)
    unscaled_hess = (X * omega).T.dot(X)
    scaled_hess = unscaled_hess / X.shape[0]
    #REGULARIZE HESS
    scaled_hess[np.arange(scaled_hess.shape[0]), np.arange(scaled_hess.shape[1])] += 2 * reg
    return scaled_hess
    

def cho_solve(A, b):
    #Want to solve Ax = b, equivalent to LL*x = b
    cho_lower = np.linalg.cholesky(A)
    #First solve Ly = b
    y = solve_triangular(cho_lower, b, lower=True)
    #Then solve L*x = y
    x = solve_triangular(cho_lower.T, y, lower=False)
    return x


def inverse_hvp_binary_top_model_explicit(model, sess, vecs, X, y, reg):
    weights = model.layers[-1].get_weights()
    if len(weights) > 1:
        weights, bias = weights[0], weights[1]
    else:
        weights = weights[0]
        bias = None
    hess = hessian_binary_crossentropy(weights, bias, X, y, reg)
    return cho_solve(hess, vecs)


def top_model_binary_gradients(model, sess, z_list, batch_z=False):
    input_ph, output_ph, loss_tensor, param_tensors = get_tensors_from_keras_model(model)
    if model.layers[-1].use_bias:
        bias = param_tensors[-1]
        weights = param_tensors[-2]
        grad_op = tf.gradients(loss_tensor, [weights, bias])
    else:
        weights = param_tensors[-1]
        grad_op = tf.gradients(loss_tensor, [weights])
    gradients = []
    if batch_z:
        g = sess.run(grad_op, feed_dict={input_ph: z_list[0], output_ph: z_list[1].reshape(-1,1)})
        gradients.append(g)
    else:
        for z in z_list:
            g = sess.run(grad_op, feed_dict={input_ph: [z[0]], output_ph: [[z[1]]]})
            gradients.append(g)
    return gradients


def s_test_binary_top_model_explicit(model, sess, z_test_list, X, y, reg, batch_z=False):
    if batch_z:
        batch_xs, batch_ys = list(zip(*z_test_list))
        batch_xs = np.stack(batch_xs)
        batch_ys = np.stack(batch_ys)
        z_test_list = [batch_xs, batch_ys]
    grads = top_model_binary_gradients(model, sess, z_test_list, batch_z)
    vecs = [np.vstack(g) for g in grads]
    vecs = np.hstack(vecs)
    return inverse_hvp_binary_top_model_explicit(model, sess, vecs, X, y, reg)


def s_test_binary_top_model_approx(model, sess, z_test_list, X, y, reg):
    pass


def influence_binary_top_model_explicit(model, sess, z_test_list, z_list, X, y, reg, batch_z=False):
    s_test_vals = s_test_binary_top_model_explicit(model, sess, z_test_list, X, y, reg, batch_z=batch_z)
    z_grads = top_model_binary_gradients(model, sess, z_list, batch_z=batch_z)
    z_grads = [np.vstack(g) for g in z_grads]
    z_grads = np.hstack(z_grads)
    #(i,j) element is influence of z_j on z_test_i
    influence_matrix = np.zeros((s_test_vals.shape[1], z_grads.shape[1]))
    for i in range(s_test_vals.shape[1]):
        for j in range(z_grads.shape[1]):
            influence = -s_test_vals[:,i].dot(z_grads[:,j])
            influence_matrix[i,j] = influence
    return influence_matrix


def influence_binary_top_model_approx(model, sess, z, z_test):
    pass


def construct_grad_influence_wrt_input_op_binary_top_model_explicit(model, sess):
    input_ph, output_ph, loss_tensor, param_tensors = get_tensors_from_keras_model(model)
    if model.layers[-1].use_bias:
        bias = param_tensors[-1]
        weights = param_tensors[-2]
        weight_grad_op = tf.gradients(loss_tensor, [weights, bias])
        s_test_placeholder = [K.placeholder(shape=weights.shape, dtype="float32"), K.placeholder(shape=bias.shape, dtype="float32")]
    else:
        weights = param_tensors[-1]
        weight_grad_op = tf.gradients(loss_tensor, [weights])
        s_test_placeholder = [K.placeholder(shape=weights.shape, dtype="float32")]
    influence_op = -tf.add_n(
                             [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b)))
                             for a, b in zip(weight_grad_op, s_test_placeholder)])
    grad_influence_wrt_input_op = tf.gradients(influence_op, input_ph)
    return grad_influence_wrt_input_op, s_test_placeholder, input_ph, output_ph, loss_tensor, param_tensors


def apply_poison_perturbation(X, poison_indices, grads, step_size, projection_fn):
    for i in range(len(poison_indices)):
        X[poison_indices[i]] += step_size * np.sign(grads[i])
    X[poison_indices] = projection_fn(X[poison_indices])


def create_projection_fn(X_orig, bounding_box_radius):
    lower_bound = np.maximum(
            -np.zeros_like(X_orig),
            X_orig - bounding_box_radius)
    upper_bound = np.minimum(
            np.ones_like(X_orig),
            X_orig + bounding_box_radius)

    def projection_fn(X):
        return np.clip(X, lower_bound, upper_bound)

    return projection_fn


#need to verify this on multilayer networks
#Such that input to given layer is the bottleneck feature
def compute_bottleneck_features(full_model, sess, X, bottleneck_layer):
    return sess.run(full_model.layers[bottleneck_layer].input, feed_dict={full_model.layers[0].input: X})


def update_bottleneck_features(X, X_top, poison_indices, full_model, sess, bottleneck_layer):
    X_top[poison_indices] = compute_bottleneck_features(full_model, sess, X[poison_indices], bottleneck_layer)


def construct_top_model(input_size, output_size, loss_type, use_bias, reg, init_model=None):
    model = Sequential()
    if loss_type == "binary_crossentropy":
        model.add(Dense(output_size, input_shape=(input_size,), activation="sigmoid", kernel_regularizer=regularizers.l2(reg)))
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    elif loss_type == "categorical_crossentropy":
        model.add(Dense(output_size, input_shape=(input_size,), activation="softmax", kernel_regularizer=regularizers.l2(reg)))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if init_model:
        model.layers[0].set_weights(init_model.layers[-1].get_weights())
    return model


def train_top_model(top_model, X, y, reg):
    C = 1.0 / (X.shape[0] * reg)
    sklearn_model = linear_model.LogisticRegression(
        C=C,
        tol=1e-8,
        fit_intercept=top_model.layers[0].use_bias,
        solver='lbfgs',
        multi_class='multinomial',
        warm_start=True,
        max_iter=1000)
    sklearn_model.fit(X, y)
    top_model.layers[0].set_weights([sklearn_model.coef_.T, sklearn_model.intercept_])


def sync_top_model_to_full_model(top_model, full_model):
    full_model.layers[-1].set_weights(top_model.layers[0].get_weights())


def data_poisoning_attack(model, sess, z_test_bottleneck_list, X, X_top, y, poison_indices, reg, step_size, num_iters, bounding_box_radius, bottleneck_layer, verbose=True):
    #Save original model weights so they can be restored after poisoning
    original_weights = model.get_weights()
    #Copy input data, because we will modify them in place to reduce copying overhead
    X = np.copy(X)
    X_top = np.copy(X_top)
    projection_fn = create_projection_fn(X[poison_indices], bounding_box_radius)
    #Get gradient of influence op and placeholder variables
    grad_influence_wrt_input_op, s_test_placeholder, input_ph, output_ph, _, _ = construct_grad_influence_wrt_input_op_binary_top_model_explicit(model, sess)
    #Construct the top model
    top_model = construct_top_model(model.layers[-1].input_shape[1], model.layers[-1].output_shape[1], model.loss, True, reg, init_model=model)
    for k in range(num_iters):
        if verbose:
            print("ATTACK ITER: ", k)
        #Compute s_test
        s_test_vals = s_test_binary_top_model_explicit(top_model, sess, z_test_bottleneck_list, X_top, y, reg, batch_z=True)[:,0]
        #Evaluate the gradient of influence op for each poison point
        grads = []
        if model.layers[-1].use_bias:
            weight_val, bias_val = s_test_vals[:-1].reshape(-1,1), s_test_vals[-1:]
            for p in poison_indices:
                grad = sess.run(grad_influence_wrt_input_op, feed_dict={s_test_placeholder[0]: weight_val, s_test_placeholder[1]: bias_val, input_ph: [X[p]], output_ph: [[y[p]]]})[0]
                grads.append(grad.reshape(X[0].shape))
        else:
            weight_val = s_test_vals.reshape(-1,1)
            for p in poison_indices:
                grad = sess.run(grad_influence_wrt_input_op, feed_dict={s_test_placeholder: weight_val, input_ph: [X[p]], output_ph: [[y[p]]]})[0]
                grads.append(grad.reshape(X[0].shape))
        #Poison points, modifies X and X_top
        apply_poison_perturbation(X, poison_indices, grads, step_size, projection_fn)
        update_bottleneck_features(X, X_top, poison_indices, model, sess, bottleneck_layer)
        #Retrain top model
        train_top_model(top_model, X_top, y, reg)
        #Load retrained weights into model
        sync_top_model_to_full_model(top_model, model)
        #Print the new confidence scores for z_test
        if verbose:
            print(top_model.predict(np.array([z[0] for z in z_test_bottleneck_list])))
    #Restore original weights
    model.set_weights(original_weights)
    #Return the poisoned points
    return X[poison_indices]


def grad_influence_wrt_input(model, sess, z_test_bottleneck_list, X, X_top, y, reg, get_norms=True, print_every=0):
    #Get gradient of influence op and placeholder variables
    grad_influence_wrt_input_op, s_test_placeholder, input_ph, output_ph, _, _ = construct_grad_influence_wrt_input_op_binary_top_model_explicit(model, sess)
    #Construct the top model
    top_model = construct_top_model(model.layers[-1].input_shape[1], model.layers[-1].output_shape[1], model.loss, True, reg, init_model=model)
    #Compute s_test
    s_test_vals = s_test_binary_top_model_explicit(top_model, sess, z_test_bottleneck_list, X_top, y, reg, batch_z=True)[:,0]
    #Evaluate the gradient of influence op for each poison point
    grads = []
    if model.layers[-1].use_bias:
        weight_val, bias_val = s_test_vals[:-1].reshape(-1,1), s_test_vals[-1:]
        for k in range(X.shape[0]):
            if print_every > 0 and k % print_every == 0:
                print("COMPUTING GRAD INFLUENCE FOR ", k)
            grad = sess.run(grad_influence_wrt_input_op, feed_dict={s_test_placeholder[0]: weight_val, s_test_placeholder[1]: bias_val, input_ph: [X[k]], output_ph: [[y[k]]]})[0]
            if get_norms:
                grads.append(np.sum(np.abs(grad)))
            else:
                grads.append(grad.reshape(X[0].shape))
    else:
        weight_val = s_test_vals.reshape(-1,1)
        for k in range(X.shape[0]):
            if print_every > 0 and k % print_every == 0:
                print("COMPUTING GRAD INFLUENCE FOR ", k)
            grad = sess.run(grad_influence_wrt_input_op, feed_dict={s_test_placeholder: weight_val, input_ph: [X[k]], output_ph: [[y[k]]]})[0]
            if get_norms:
                grads.append(np.sum(np.abs(grad)))
            else:
                grads.append(grad.reshape(X[0].shape))
    return grads




#TODO: test on multilayer net (probably just dump a pre-trained deep net into it, see how it fares)
#TODO: write full-model influence calculator
#TODO: extend to multiclass? maybe only use fmin_ncg solver (and not exact Hessian method)
#TODO: implement Max-Mahalnobis network (it actually shouldn't be too hard I think, and is independent of all this)
#NOTE: NumPy -> CuPy?







def hessian_categorical_crossentropy(weights, bias, X, y, reg):
    pass

# def hessian(input_ph, output_ph, loss_tensor, param_tensors, sess, X, y):
#     hess = tf.hessians(loss_tensor, param_tensors)
#     hess_val = sess.run(hess, feed_dict={input_ph: X, output_ph: y})
#     return hess_val

#ASSUME THAT WE'LL ALWAYS STACK WEIGHTS ABOVE BIASES
#This should use the top model and the bottleneck features
#Test with multiclass
#Oh wait can't the hessians for softmax logistic regression be computed exactly?
def one_layer_exact_hessian(input_ph, output_ph, loss_tensor, weight_tensor, bias_tensor, sess, X, y):
    if bias_tensor is not None:
        block_hess = tf.hessians(loss_tensor, [weight_tensor, bias_tensor])
        cross_hess = tf.gradients(tf.gradients(loss_tensor, weight_tensor)[0], bias_tensor)
        block_vals, dwb = sess.run([block_hess, cross_hess], feed_dict={input_ph: X, output_ph: y})
        dww, dbb = block_vals
        print(dww.shape, dbb.shape)
        print(dwb)


def hessian_keras(model, sess, X, y):
    input_ph, output_ph, loss_tensor, param_tensor = get_tensors_from_keras_model(model)
    hess_val = hessian(input_ph, output_ph, loss_tensor, param_tensor, sess, X, y)
    return hess_val


def s_test_with_exact_hessian(model, sess, z_test):
    pass

def influence_with_exact_hessian(model, sess, z, z_test):
    pass

#Construct the top model approximation within the function, for ease of use
def poison_with_exact_hessian(model, sess, z_test, X, y, poison_index):
    pass



###FUNCTIONS I WANT###
#return exact hessian using only last layer (can use either weights or both weights + bias, include reg as parameter? need to test to see reg equivalency)
#return exact inverse hvp using only last layer
#return exact s_test
#compute grad of influence wrt input using only last layer
#retrain last layer
#poison attack
#return exact hvp over full model
#minibatch version of exact hvp over full model???
#compute influence over full model (fmin_ncg method)
#compute influence using only last layer (can save hessian for this)
#Gauss-Newton approximations? Might not need these
#Saving/precomputing functionality? Probably don't need it

#THOUGHTS:
#Do I want a function which will construct a top model approximation
#Do I want an object to store bottleneck features? Or is that something that should just be done in advance
#Yeah so: Have model weights and bottleneck features computed in advance

#What happens if we do adversarial training, but instead of perturbing random points, we perturb the highest influence points?
#What if we perturb the lowest influence points? Hypothesis: former leads to better models, more efficient adversarial training

#What if we did a full model poisoning + retraining (for a relatively small but still deep model)? Actually nah.



# def get_inverse_hvp(input_ph, output_ph, loss_tensor, weight_ph, sess, v, X, y,
#                           batch_size=None, scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
#     inverse_hvp = None
#     v_placeholders = [K.placeholder(t.shape) for t in v]
#     data_loader = DataLoader(X, y)
#     for i in range(num_samples):
#         curr_estimate = v
#         for j in range(recursion_depth):
#             print(j)
#             next_input, next_output = data_loader.next_batch(batch_size)
#             feed_dict = populate_feed_dict(input_ph, next_input, output_ph, next_output, v_placeholders, curr_estimate)
#             hvp = hessian_vector_product(loss_tensor, weight_ph, v_placeholders)
#             hessian_vector_val = sess.run(hvp, feed_dict=feed_dict)
#             curr_estimate = [a + (1 - damping) * b - c/scale for (a,b,c) in zip(v, curr_estimate, hessian_vector_val)]
#         if inverse_hvp is None:
#             inverse_hvp = [b/scale for b in curr_estimate]
#         else:
#             inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, curr_estimate)]
#     inverse_hvp = [a/num_samples for a in inverse_hvp]
#     return inverse_hvp


# def get_inverse_hvp_keras(model, sess, v, X, y,
#                                 batch_size=None, scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
#     input_ph, output_ph, loss_tensor, model_params = get_tensors_from_keras_model(model)
#     return get_inverse_hvp_lissa(input_ph, output_ph, loss_tensor, model_params, sess, v, X, y,
#                                  batch_size, scale, damping, num_samples, recursion_depth)


# def s_test(model, sess, z_test_in, z_test_out, X, y,
#            batch_size=None, scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
#     input_ph, output_ph, loss_tensor, model_params = get_tensors_from_keras_model(model)
#     grads = gradients(loss_tensor, model_params)
#     feed_dict = {input_ph: [z_test_in], output_ph: [z_test_out]}
#     grad_wrt_input = sess.run(grads, feed_dict=feed_dict)
#     return get_inverse_hvp_keras(model, sess, grad_wrt_input, X, y,
#                                        batch_size=batch_size, scale=scale, damping=damping, num_samples=num_samples, recursion_depth=recursion_depth)






