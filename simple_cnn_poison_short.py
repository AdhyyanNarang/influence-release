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

#Config
input_shape = (3,32,32)
#SET TRAINED TO TRUE IF MODEL IS TRAINED AND SAVED
trained = True
features_computed = True

#Dataset preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
train_bird_frog = np.where(np.logical_or(y_train == 2, y_train == 6))
test_bird_frog = np.where(np.logical_or(y_test == 2, y_test == 6))
x_train = x_train[train_bird_frog] / 255.
y_train = y_train[train_bird_frog]
y_train[y_train == 6] = 1
y_train[y_train == 2] = 0
x_test = x_test[test_bird_frog] / 255.
y_test = y_test[test_bird_frog]
y_test[y_test == 6] = 1
y_test[y_test == 2] = 0

#SET RNG SEED
np.random.seed(32)

sess = K.get_session()
model = SimpleCNN(input_shape=input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if not trained:
    model.fit(x_train, y_train, epochs=5)
    v = model.evaluate(x_test, y_test)
    print(v)
    model.save_weights("models/cifar10_bird_frog_simple_cnn.h5")

model.load_weights("models/cifar10_bird_frog_simple_cnn.h5")

if not features_computed:
    print("COMPUTING TRAIN BOTTLENECK FEATURES")
    train_bottleneck_features = []
    for k in range(10):
        print("ITER ", k)
        bottleneck_features = compute_bottleneck_features(model, sess, x_train[1000*k:1000*(k+1)], -1)
        train_bottleneck_features.append(bottleneck_features)
    train_bottleneck_features = np.vstack(train_bottleneck_features)

    print("COMPUTING TEST BOTTLENECK FEATURES")
    test_bottleneck_features = compute_bottleneck_features(model, sess, x_test, -1)

    np.save("precomputed_features/train_bottleneck_features.npy", train_bottleneck_features)
    np.save("precomputed_features/test_bottleneck_features.npy", test_bottleneck_features)

train_bottleneck_features = np.load("precomputed_features/train_bottleneck_features.npy")
test_bottleneck_features = np.load("precomputed_features/test_bottleneck_features.npy")


#Reduce training set sizes
train_bottleneck_features = train_bottleneck_features[:1000]
x_train = x_train[:1000]
y_train = y_train[:1000]



lamb = 1
top_model = construct_top_model(512, 1, "binary_crossentropy", True, lamb)
train_top_model(top_model, train_bottleneck_features, y_train, lamb)
sync_top_model_to_full_model(top_model, model)

preds = top_model.predict(test_bottleneck_features)
rounded_preds = np.round(preds)

correct_indices = np.where(np.logical_and((rounded_preds.flatten() == y_test), (y_test == 0), (preds.flatten() > 0.25)))[0]
test_index_to_flip = correct_indices[0]
print(test_index_to_flip)

z_test_bottleneck_list = [(test_bottleneck_features[test_index_to_flip], y_test[test_index_to_flip])]

# grad_norms = grad_influence_wrt_input(model, sess, z_test_bottleneck_list, x_train, train_bottleneck_features, y_train, lamb, print_every=500)
# sorted_indices = list(reversed(np.argsort(grad_norms)))
# print(sorted_indices)
sorted_indices = [368, 579, 62, 822, 459, 369, 222, 752, 862, 948, 856, 719, 381, 754, 148, 151, 415, 219, 177, 664, 291, 427, 200, 216, 715, 191, 791, 696, 54, 358, 49, 969, 511, 595, 571, 109, 522, 657, 454, 410, 930, 808, 785, 623, 812, 622, 192, 707, 3, 937, 324, 451, 119, 15, 161, 468, 994, 413, 209, 179, 641, 848, 123, 241, 954, 973, 394, 624, 133, 366, 889, 473, 255, 355, 233, 385, 338, 672, 683, 775, 2, 789, 962, 885, 139, 268, 218, 725, 488, 73, 515, 97, 574, 55, 225, 476, 939, 408, 587, 534, 287, 442, 443, 58, 562, 252, 289, 653, 51, 94, 788, 726, 686, 729, 917, 866, 975, 372, 870, 794, 506, 996, 378, 869, 501, 607, 127, 351, 79, 610, 37, 879, 388, 897, 691, 204, 199, 162, 439, 158, 232, 344, 841, 137, 167, 302, 509, 67, 849, 904, 382, 353, 825, 18, 354, 878, 215, 313, 598, 258, 915, 632, 391, 913, 832, 85, 778, 875, 806, 497, 126, 642, 317, 175, 677, 63, 486, 837, 928, 830, 359, 436, 206, 958, 20, 739, 92, 580, 640, 529, 768, 734, 70, 309, 496, 482, 858, 773, 863, 783, 553, 297, 50, 306, 5, 864, 931, 997, 938, 149, 48, 150, 330, 824, 83, 272, 698, 854, 472, 514, 155, 33, 701, 300, 143, 842, 467, 893, 340, 475, 838, 166, 953, 251, 171, 339, 880, 403, 121, 536, 748, 611, 722, 572, 890, 145, 59, 512, 487, 334, 612, 552, 102, 373, 663, 456, 285, 75, 617, 844, 510, 577, 592, 176, 257, 557, 131, 998, 0, 706, 527, 603, 554, 568, 409, 803, 943, 712, 106, 530, 630, 910, 214, 326, 17, 895, 229, 103, 528, 989, 337, 643, 660, 26, 981, 16, 721, 556, 795, 987, 78, 853, 769, 122, 293, 269, 946, 851, 240, 448, 39, 620, 829, 923, 978, 396, 96, 877, 920, 633, 951, 320, 484, 873, 540, 763, 733, 668, 764, 628, 941, 295, 82, 349, 322, 152, 380, 786, 286, 425, 134, 727, 100, 737, 992, 602, 539, 833, 296, 236, 213, 916, 458, 230, 749, 991, 901, 333, 132, 431, 586, 244, 545, 434, 419, 347, 787, 235, 84, 801, 7, 750, 400, 852, 159, 892, 894, 124, 919, 117, 243, 399, 735, 887, 898, 6, 703, 732, 99, 865, 441, 81, 666, 525, 444, 433, 341, 87, 828, 517, 694, 69, 14, 38, 636, 207, 757, 19, 615, 342, 30, 709, 411, 667, 804, 762, 988, 600, 933, 420, 766, 576, 479, 184, 141, 781, 319, 52, 784, 186, 266, 270, 513, 547, 839, 563, 965, 716, 271, 283, 29, 535, 940, 559, 329, 759, 25, 194, 414, 899, 927, 406, 263, 711, 675, 426, 428, 868, 679, 531, 53, 673, 383, 494, 437, 308, 140, 650, 21, 220, 932, 644, 316, 418, 136, 771, 76, 108, 682, 670, 619, 780, 113, 724, 626, 881, 590, 35, 135, 697, 968, 273, 950, 470, 648, 335, 495, 203, 276, 34, 906, 767, 190, 575, 384, 290, 261, 834, 524, 4, 964, 855, 720, 455, 774, 60, 57, 234, 477, 492, 227, 125, 64, 471, 631, 815, 793, 947, 655, 146, 693, 544, 558, 740, 65, 182, 689, 174, 638, 980, 88, 457, 583, 31, 447, 900, 665, 493, 993, 599, 605, 896, 423, 299, 814, 1, 976, 584, 561, 765, 613, 264, 944, 582, 702, 886, 564, 499, 460, 743, 705, 412, 700, 404, 68, 142, 343, 185, 942, 761, 292, 678, 303, 627, 796, 744, 379, 652, 197, 464, 310, 831, 850, 661, 242, 195, 543, 387, 800, 154, 972, 659, 516, 876, 614, 609, 169, 188, 573, 807, 112, 685, 453, 971, 567, 481, 318, 503, 925, 979, 417, 699, 957, 430, 202, 254, 435, 533, 223, 46, 747, 634, 713, 47, 10, 523, 802, 101, 375, 228, 588, 277, 872, 282, 432, 424, 323, 560, 328, 649, 360, 952, 24, 742, 934, 591, 647, 756, 265, 676, 625, 637, 281, 107, 231, 267, 760, 959, 77, 581, 593, 371, 250, 918, 977, 782, 491, 8, 936, 336, 438, 208, 445, 728, 178, 352, 485, 331, 746, 542, 914, 836, 402, 89, 144, 128, 305, 480, 867, 249, 187, 40, 566, 658, 888, 845, 776, 799, 982, 259, 999, 639, 312, 217, 741, 621, 465, 90, 662, 398, 755, 731, 618, 507, 671, 224, 986, 45, 12, 827, 314, 857, 805, 210, 246, 758, 183, 239, 884, 790, 237, 818, 882, 247, 911, 226, 74, 489, 364, 565, 797, 446, 376, 288, 345, 908, 695, 555, 483, 311, 389, 463, 490, 278, 926, 966, 147, 505, 32, 462, 22, 548, 984, 922, 551, 138, 248, 294, 909, 198, 596, 723, 688, 635, 504, 817, 173, 570, 205, 390, 104, 520, 163, 594, 813, 327, 284, 656, 253, 674, 114, 95, 93, 589, 91, 718, 645, 43, 365, 840, 42, 260, 401, 983, 521, 963, 164, 386, 835, 332, 646, 891, 346, 356, 27, 474, 847, 518, 745, 393, 710, 130, 608, 684, 110, 990, 821, 974, 995, 397, 279, 985, 651, 526, 168, 115, 304, 361, 120, 905, 274, 211, 44, 508, 256, 66, 165, 478, 629, 201, 13, 792, 153, 407, 550, 961, 367, 156, 392, 221, 606, 654, 597, 541, 262, 61, 11, 585, 500, 36, 955, 466, 816, 912, 843, 960, 708, 298, 819, 193, 160, 429, 307, 929, 616, 692, 956, 23, 810, 717, 321, 170, 871, 280, 809, 71, 949, 538, 569, 363, 921, 469, 374, 779, 325, 80, 189, 129, 601, 118, 578, 546, 28, 903, 41, 116, 883, 549, 502, 98, 111, 970, 350, 449, 461, 826, 861, 770, 377, 777, 416, 238, 172, 680, 348, 301, 860, 967, 738, 820, 690, 772, 422, 370, 874, 157, 907, 681, 532, 105, 846, 181, 440, 798, 395, 823, 180, 704, 935, 902, 86, 736, 537, 357, 405, 196, 945, 72, 421, 753, 714, 450, 669, 859, 212, 751, 498, 56, 245, 519, 811, 315, 452, 275, 604, 924, 362, 730, 9, 687]

print('Entered Data Poisoning Attack')

poisoned_points = data_poisoning_attack(model, sess, z_test_bottleneck_list, x_train, train_bottleneck_features, y_train, sorted_indices[:10], lamb, 0.01, 100, 0.05, -1)

unpoisoned_test = np.transpose(x_train[sorted_indices[0]], (1,2,0))
poison_test = np.transpose(poisoned_points[0], (1,2,0))
orig = np.transpose(x_test[test_index_to_flip], (1,2,0))



plt.figure(1)
plt.imshow(unpoisoned_test, cmap="gray")
plt.figure(2)
plt.imshow(orig, cmap="gray")
plt.figure(3)
plt.imshow(poison_test, cmap="gray")
plt.show()










