import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA, PCA


def style_axis():
    fig = plt.figure()
    ax = fig.add_subplot()
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.margins(.3, .3)
    return fig, ax


# let's create our signals
np.random.seed(100)
U1 = np.random.uniform(-1, 1, 2000)
U2 = np.random.uniform(-1, 1, 2000)

fig, ax = style_axis()

ax.scatter(U1, U2, marker=".", s=5, c='b')
ax.set_title("Original composition", pad=20)

plt.show()

# now comes the mixing part. we can choose a random matrix for the mixing

A = np.array([[.84, .52], [.84, .27]])

U_source = np.array([U1, U2])
U_mix = U_source.T.dot(A)

# plot of our dataset

fig, ax = style_axis()

ax.set_title("Observed mixture", pad=20)
ax.scatter(U_mix[:, 0], U_mix[:, 1], marker=".", s=5, c='b')

plt.show()

# PCA and whitening the dataset
U_pca = PCA(whiten=True).fit_transform(U_mix)

# let's plot the datasets
fig, ax = style_axis()

ax.set_title("PCA", pad=20)
ax.scatter(U_pca[:, 0], U_pca[:, 1], marker=".", s=5, c='b')

plt.show()

# ICA
U_ica = FastICA(whiten=True).fit_transform(U_mix)

# let's plot the datasets
fig, ax1 = style_axis()

#ax1.set_title("ICA", pad=20)
ax1.scatter(U_ica[:, 0], U_ica[:, 1], marker=".", s=5, c='b')

plt.show()
