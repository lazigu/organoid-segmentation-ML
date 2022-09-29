from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle
import seaborn as sns


def visualize_feature_stack(feature_stack, z_slice):
    """
    Visualization of the ith z-slice
    """

    if len(feature_stack) == 0:
        return

    width = 3
    height = int(len(feature_stack) / 3)
    if height * width < len(feature_stack):
        height = height + 1

    fig, axes = plt.subplots(height, width, figsize=(10, 10))

    # reshape(image.shape) is the opposite of ravel() here. We just need it for visualization.
    for i, f in enumerate(feature_stack):
        if height > 1:
            cle.imshow(f[z_slice], plot=axes[int(i / 3)][i % 3], colormap=plt.cm.gray)
        else:
            cle.imshow(f[z_slice], plot=axes[i], colormap=plt.cm.gray)

    w = len(feature_stack) % width
    if w > 0:
        w = width - w
        while (w > 0):
            if height > 1:
                axes[-1][width - w].set_visible(False)
            else:
                axes[width - w].set_visible(False)
            w = w - 1

    plt.show()


def plot_matching_dataset_results(metrics, taus, save_dir=None):
    """
    Utilities function to plot various metric values against a list of thresholds based on Stardist 3D training notebook.
    """
    import matplotlib.pyplot as plt

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    labels = ('Precision', 'Recall', 'Accuracy', 'F1', 'Mean true score', 'Panoptic quality')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    plt.subplots_adjust(wspace=1)
    for m, lbl in zip(('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'panoptic_quality'), labels):
        ax1.plot(taus, [s._asdict()[m] for s in metrics], '.-', lw=2, label=lbl)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.6, 1))

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in metrics], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.legend(frameon=False)

    if save_dir is not None:
        plt.savefig(save_dir,bbox_inches='tight')


def plot_confusion_matrix(metrics, taus=None, thres=None, save_dir=None):
    """
    Confusion matrix is plotted in the form:
    TP|FN
    FP|TN
    """
    cm_data_1 = []
    cm_data_2 = []

    for m in ('tp', 'fn', 'fp'):
        if m == "tp" or m == "fn":
            if thres and taus is not None:
                cm_data_1.append(metrics[taus.index(thres)]._asdict()[m])
            else:
                cm_data_1.append(metrics._asdict()[m])

        elif m == "fp":
            if thres and taus is not None:
                cm_data_2.append(metrics[taus.index(thres)]._asdict()[m])
            else:
                cm_data_2.append(metrics._asdict()[m])

    cm_data_2.append(0)  # tn are not known
    cm_data = [cm_data_1, cm_data_2]
    fig, ax = plt.subplots(figsize=(4, 5))
    sns.heatmap(cm_data, annot=True, cmap='Blues', fmt='d', annot_kws={'fontsize': 30}, ax=ax, yticklabels=False,
                xticklabels=False)

    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
