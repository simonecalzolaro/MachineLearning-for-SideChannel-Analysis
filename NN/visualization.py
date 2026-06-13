# Basics
import matplotlib
matplotlib.use('svg') # Avoid interactive mode (and save files as .SVG as default)
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# Custom
import results


def plot_nicv(nicvs, configs, output_path):

    """
    Plots NICV values and saves the result in a SVG file.

    Parameters:
        - nicvs (np.ndarray):
            NICV values to plot.
        - configs (str list):
            Device-key configurations that generate the traces used to compute
            NICV values.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    cmap = plt.cm.Set1
    colors = cmap(range(len(configs)))

    f, ax = plt.subplots(4, 4, figsize=(25,25))

    for i, c in enumerate(configs):
        row = 0
        for b in range(16):
            col = b % 4

            ax[row, col].plot(nicvs[i][b], label=c, color=colors[i])
            ax[row, col].legend()
            ax[row, col].set_title(f'Byte {b}')
            ax[row, col].set_xlabel('Samples')
            ax[row, col].set_ylabel('NICV')

            if col == 3:
                row += 1

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_history(history, output_path):

    """
    Plots a train history and saves the result in a SVG file.

    Parameters:
        - history (dict):
            Train history to plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    f, ax = plt.subplots(1, 2, figsize=(18,8))

    ax[0].plot(history['loss'], label='train_loss')
    ax[0].plot(history['val_loss'], label='val_loss')
    ax[0].set_title('Train and Val Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(history['accuracy'], label='train_acc')
    ax[1].plot(history['val_accuracy'], label='val_acc')
    ax[1].set_title('Train and Val Acc')
    ax[1].set_ylabel('Acc')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    ax[1].grid()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_conf_matrix(conf_matrix, output_path):

    """
    Plots a confusion matrix and saves the result in a SVG file.

    Parameters:
        - conf_matrix (np.ndarray):
            Confusion matrix to plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    cmap = plt.cm.Blues

    f = plt.figure(figsize=(10,8))
    plt.imshow(conf_matrix, cmap=cmap)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.colorbar()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_ges(ges, labels, title, ylim_max, output_path):

    """
    Plots GEs using Google Turbo color-palette and saves the result in a .SVG file.

    Parameters:
        - ges (np.ndarray):
            GEs to plot.
        - labels (str list):
            Labels to use as plot-legend.
        - title (str):
            Title of the plot.
        - ylim_max (int):
            Upper limit for y-axis.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    # Set the color palette
    cmap = plt.cm.jet # Google Turbo
    colors = cmap(range(0, cmap.N, int(cmap.N/len(ges))))

    # Plot the GEs
    f, ax = plt.subplots(figsize=(10,5))
    for ge, l, c in zip(ges, labels, colors):
        ax.plot(ge, label=l, marker='o', color=c)

    ax.set_title(title)
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1))
    ax.set_ylim([-3, ylim_max])
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('GE')
    ax.legend()
    ax.grid()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )
    f.savefig(
        output_path.replace('svg', 'png'),
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_overlap(all_ges, to_compare, title, ylim_max, output_path):

    """
    Plots GEs resulting from 2 different DKTA experiments in a single plane and
    saves the result in a SVG file.

    Parameters:
        - all_ges (np.ndarray):
            GEs to plot.
        - to_compare (int list):
            Bytes whose results are compared.
        - ylim_max (int):
            Upper limit for y-axis.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    colors = list(clrs.TABLEAU_COLORS)

    f, ax = plt.subplots(figsize=(10,5))

    for i, ges in enumerate(all_ges): # i used for indexing the compared bytes

        for j, ge in enumerate(ges): # j used for labeling

            ge = ge[:10]

            if j == len(ges) - 1: # Label only the last element of each group
                label = f'Byte {to_compare[i]}'
                ax.plot(ge, color=colors[i], marker='o', label=label)
            else:
                ax.plot(ge, color=colors[i], marker='o')

    ax.set_title(title)
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1)) # Consider the last ge, but all have same length
    ax.set_ylim([-3, ylim_max])
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('GE')
    ax.legend()
    ax.grid()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )
    f.savefig(
        output_path.replace('svg', 'png'),
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_soa_vs_custom(soa_ge, custom_ge, threshold, title, ylim_max, output_path):

    """
    Plots GEs derived with both the State-of-the-Art approach and the Genetic
    Algorithm approach, highlighting the minimum number of traces that ensures GE
    values less than a given threshold.

    Parameters:
        - soa_ge (np.ndarray):
            GE derived with a State-of-the-Art approach.
        - custom_ge (np.ndarray):
            GE derived with a custom approach.
        - threshold (float):
            Threshold for GE values.
        - title (str):
            Title of the plot.
        - ylim_max (int):
            Upper limit for y-axis.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    f = plt.figure(figsize=(10,5))

    plt.plot(soa_ge, label='SoA', marker='o', color='b')
    v_value = results.min_att_tr(soa_ge, threshold)
    plt.axvline(v_value, color='b', linestyle='--', label=f'SoA GE<={threshold}')

    plt.plot(custom_ge, label='GenAlg', marker='o', color='r')
    v_value = results.min_att_tr(custom_ge, threshold)
    plt.axvline(v_value, color='r', linestyle='--', label=f'GenAlg GE<={threshold}')

    plt.xticks(range(len(soa_ge)), labels=range(1, len(soa_ge)+1)) # Consider the soa_ge, but both have same length
    plt.ylim([-3, ylim_max])
    plt.xlabel('Number of Attack Traces')
    plt.ylabel('GE')
    plt.title(title)
    plt.legend()
    plt.grid()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_min_att_tr(min_att_tr, threshold, xlabels, ylim_max, title, output_path):

    """
    Plots the minimum number of attack traces that allows to have GE values less
    than a given threshold, for different experiments.

    Parameters:
        - min_att_tr (int list):
            Minimum number of attack traces to have GE values less than the
            threshold.
        - threshold (float):
            Threshold for GE values.
        - xlabels (int list):
            Markers for x-axis values.
        - title (str):
            Title of the plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    f = plt.figure(figsize=(10,5))

    plt.plot(min_att_tr, marker='o', color='b')

    plt.xticks(range(len(min_att_tr)), labels=xlabels)
    plt.ylim([1, ylim_max])
    plt.xlabel('Number of Total Train-Traces')
    plt.ylabel(f'Number of Attack Traces for GE<={threshold}')
    plt.title(title)
    plt.grid()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)


def plot_overlap_min_att_tr(soa_data, custom_data, threshold, xlabels, ylim_max, title, output_path):

    """
    Plots the minimum number of attack traces that allows to have GE values less
    than a given threshold using data derived from both the State-of-the-Art
    approach and a custom approach.

    Parameters:
        - soa_data (np.array):
            Data relative to State-of-the-Art approach.
        - custom_data (np.array):
            Data relative to custom approach.
        - threshold (float):
            Threshold for GE values.
        - xlabels (int list):
            Markers for x-axis values.
        - ylim_max (int):
            Upper limit for y-axis.
        - title (str):
            Title of the plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    f = plt.figure(figsize=(10,5))

    plt.plot(soa_data, marker='o', color='b', label='SoA')
    plt.plot(custom_data, marker='o', color='r', label='GenAlg')

    plt.xticks(range(len(soa_data)), labels=xlabels)
    plt.ylim([1, ylim_max])
    plt.xlabel('Number of Total Train-Traces')
    plt.ylabel(f'Number of Attack Traces for GE<={threshold}')
    plt.title(title)
    plt.legend()
    plt.grid()

    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    plt.close(f)