import numpy as np
from matplotlib import pyplot as plt


def plot_results(test_set, fitted_model, heat=False, hist=True, support_vectors=None):
    "Plotting SVM-Classification results"

    plt.figure(figsize=(15, 7))

    features = test_set[0:-1].reshape([2, -1])
    classes = test_set[-1]
    sample_size = test_set.shape[1]

    predicted = fitted_model.predict(features.T)
    Error = features[0:, classes != np.round(predicted)]

    Accuracy = 1 - Error.shape[1] / sample_size
    print("Wrong predicted samples = {}".format(Error.shape[1]))
    print("Accuracy = {:4.3f}".format(Accuracy))

    x_min, x_max = features[0].min() - 1, features[0].max() + 1
    y_min, y_max = features[1].min() - 1, features[1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting Contours
    if heat:
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.colorbar()
    plt.contour(xx, yy, Z, levels=[0], colors='k')

    # Slice first 2 dimensions for plotting
    K1 = features[0:2, classes == -1]
    K2 = features[0:2, classes == 1]

    plt.scatter(K1[0], K1[1], alpha=.4, label="Class 1")
    plt.scatter(K2[0], K2[1], alpha=.4, label="Class 2", c="brown")
    plt.scatter(Error[0], Error[1], marker="x", color="k", label="wrong predicted")

    if support_vectors is not None:
        plt.scatter(support_vectors[0], support_vectors[1], alpha=.5, label="support vectors")

    plt.legend(bbox_to_anchor=(1, 1))
    plt.axis("off")
    plt.show()

    # Histogram
    if hist:
        plt.figure(figsize=(15, 3))
        plt.hist(predicted[classes == -1], alpha=0.5, label="Class 1")
        plt.hist(predicted[classes == 1], alpha=0.5, color="brown", label="Class2")
        ax = plt.gca()
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, max(175, ax.get_ylim()[1])])

        n = 0
        x = [1, 3]
        for spine in ax.spines.values():
            if n in x: spine.set_visible(False)
            n += 1

        line = np.array([(0, 0), (0, 100)]).T
        plt.plot(line[0], line[1], "--k")

        plt.show()


def plot_data(set1, set2):

    fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 3]})
    fig.set_size_inches(15, 8)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    plt.sca(ax1)
    ax1.hist(set1[0], alpha=0.5)
    ax1.hist(set2[0], color="brown", alpha=0.5)
    plt.xticks([0], [""])

    ax2.axis('off')

    ax3.scatter(set1[0], set1[1], alpha=0.4)
    ax3.scatter(set2[0], set2[1], color="brown", alpha=0.4)

    plt.sca(ax4)
    ax4.hist(set1[1], alpha=0.5, orientation=u'horizontal')
    ax4.hist(set2[1], color="brown", alpha=0.5, orientation=u'horizontal')
    plt.yticks([0], [""])

    n = 0
    x = [1, 3]
    for spine in ax1.spines.values():
        if n in x: spine.set_visible(False)
        n += 1
    n = 0
    for spine in ax3.spines.values():
        if n in x: spine.set_visible(False)
        n += 1
    n = 0
    for spine in ax4.spines.values():
        if n in x: spine.set_visible(False)
        n += 1

    plt.show()
