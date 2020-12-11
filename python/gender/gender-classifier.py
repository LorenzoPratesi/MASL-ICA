from gender.main import *

import logging
from time import time

import matplotlib.pyplot as plt
from numpy.random import RandomState

from sklearn.svm import SVC
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Display progress logs on stdout
n_row, n_col = 5, 10
n_components = n_row * n_col
image_shape = (64, 64)

# #############################################################################
# Load faces data
path = "../resources/gender_dataset/Training"

[X, y] = read_images(path, limit=1000)

#plot_samples(X, dgrid=(n_row, n_col), plot_title="Original samples")

X_processed = process_input(X)

# split into a training and testing set
X_train, X_test, y_train, y_test = split_data(X_processed, y)

n_samples, n_features = X_train.shape

# global centering
faces_centered = X_train - X_train.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap, interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


# #############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    ('Eigenfaces - PCA',
     PCA(n_components=n_components, svd_solver='randomized', whiten=True),
     True),

    ('Independent components - FastICA',
     FastICA(n_components=n_components, whiten=True),
     True),

    ('Factor Analysis components - FA',
     FactorAnalysis(n_components=n_components, max_iter=20),
     True),
]

# #############################################################################
# Plot a sample of the input data
random_faces = get_random_faces(faces_centered, dim=n_row * n_col)

plot_gallery("Original faces", X_test[:n_components])

a = X_test - X_test.mean(axis=0)
a -= a.mean(axis=1).reshape(X_test.shape[0], -1)
plot_gallery("First centered faces", a[:n_components])

plt.show()

# #############################################################################
# Do the estimation and plot it

for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = X_processed
    if center:
        data = faces_centered

    X_extracted = estimator.fit(data)

    train_time = (time() - t0)

    X_train_model, X_test_model = train_text_transform_Model(X_extracted, X_train, X_test)

    clf = classification_svc(X_train_model, y_train)
    print("done in %0.3fs" % (time() - t0))

    y_pred = clf.predict(X_test_model)

    if hasattr(X_extracted, 'cluster_centers_'):
        components_ = X_extracted.cluster_centers_
    else:
        components_ = X_extracted.components_

    plot_title = '%s - Train time %.1fs' % (name, train_time)
    plot_gallery(plot_title, components_[:n_components])

    prediction_titles = [title(y_pred, y_test, ['female', 'male'], i) for i in range(y_pred.shape[0])]

    plot_images(X_test, plot_title, prediction_titles, image_shape[0], image_shape[1], n_row=n_row, n_col=n_col)

    print("Accuracy: %0.2f%%" % (accuracy_score(y_test, y_pred) * 100))
    print(classification_report(y_test, y_pred))


plt.show()
