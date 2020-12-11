import os
import PIL.Image as Image
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def read_images(path, limit=50):
    gender_dict = {'female': 0, 'male': 1}

    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            c = 0
            for filename in os.listdir(subject_path):
                if c == limit:
                    break
                c += 1
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    X.append(im)
                    y.append(gender_dict[subdirname])
                    pass
                except IOError:
                    continue
    return [X, y]


def process_input(X):
    def img_process(img):
        r_dims = (64, 64)
        img = img.resize(r_dims).convert("L")
        return np.asarray(img, dtype=np.uint8).reshape(-1)

    return np.asarray(list(map(img_process, X)))


def get_random_faces(faces, dim=25):
    random.seed(123)
    return random.choices(faces, k=dim)


def plot_samples(X, dgrid, plot_title):
    samples = get_random_faces(X, dim=dgrid[0] * dgrid[1])
    _, grid = plt.subplots(dgrid[0], dgrid[1], figsize=(dgrid[1] * 2, dgrid[0] * 2))
    for i in range(dgrid[0]):
        for j in range(dgrid[1]):
            grid[i][j].imshow(samples[dgrid[1] * i + j])

    plt.suptitle(plot_title, size=16)
    plt.show()


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


def train_text_transform_Model(model, X_train, X_test):
    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_model = model.transform(X_train)
    X_test_model = model.transform(X_test)

    return X_train_model, X_test_model


def classification_svc(X_train_model, y_train):
    print("Fitting the classifier to the training set")

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # clf = SVC(kernel='rbf', class_weight='balanced')
    clf = clf.fit(X_train_model, y_train)

    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)


def plot_images(images, plot_title, titles, height, width, n_row=5, n_col=10):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(plot_title, size=16)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((height, width)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    plt.show()

