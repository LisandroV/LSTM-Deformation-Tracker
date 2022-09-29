import matplotlib

# matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf
import io

from memory_profiler import profile

# @profile # was used to investigate memory leak
def create_weight_matrix_image(weights_matrix, image_name):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        weights_matrix (array, shape = [m, n])
        image_name (str)
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(weights_matrix, interpolation="nearest", cmap=plt.cm.afmhot)
    plt.title(image_name)
    plt.colorbar()

    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(weights_matrix.astype('float') / weights_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    # threshold = weights_matrix.max() / 2.
    # for i, j in itertools.product(range(weights_matrix.shape[0]), range(weights_matrix.shape[1])):
    # color = "white" if weights_matrix[i, j] > threshold else "black"
    # plt.text(j, i, labels[i, j], horizontalalignment="center", color=color) # to plot weight on every cell

    plt.tight_layout()
    weights_image = plot_to_image(figure)
    return weights_image


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    plt.clf()
    plt.pause(0.01)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    buf.close()
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class PlotWeightsCallback(tf.keras.callbacks.Callback):
    """
    callback to plot the weights of the network
    note: define the log_dir property on the model, or no image will be saved
    """

    def __init__(self, plot_freq=100):
        """
        plot_step: how often the weight plot will be generated
        """
        super(PlotWeightsCallback, self).__init__()
        self.plot_step = plot_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.plot_step) != 0:
            return

        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if len(layer_weights) < 3:
                image_titles = ["input_weights", "bias_weights"]  # for dense layer
            else:
                image_titles = [
                    "input_weights",
                    "recurrent_weights",
                    "bias_weights",
                ]  # for recurrent layer
            for index, weight_matrix in enumerate(layer_weights):
                image_name = f"{layer.name}_{image_titles[index]}"
                file_writer = tf.summary.create_file_writer(
                    self.model.log_dir + "/weights/" + image_name
                )
                with file_writer.as_default():
                    if weight_matrix.ndim > 1:
                        weights_image = create_weight_matrix_image(
                            weight_matrix, image_name
                        )
                    else:
                        weights_image = create_weight_matrix_image(
                            weight_matrix.reshape(weight_matrix.size, 1), image_name
                        )  # edge case to handle matrix shape (n,)
                    tf.summary.image(image_name, weights_image, step=epoch)
