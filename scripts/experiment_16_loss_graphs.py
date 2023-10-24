"""
Plot the min, mix and average of all the random search experiments
"""
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea
from tensorflow import make_ndarray
import matplotlib
import math
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")

tf_size_guidance = {
    'tensors': 12000 # Loading too much data is slow...
}

def get_tensorboard_y_data(path):
    event_acc = ea.EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()
    #print(event_acc.Tags()) # Show all tags in the log file
    tensor_events = event_acc.Reload().Tensors('epoch_loss')
    y = np.stack([make_ndarray(te.tensor_proto) for te in tensor_events])
    return y

def tensorboard_smooth(scalars: list[float], weight: float) -> list[float]:
    """
    Tensorboard implementation to smooth a function
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

def get_smooth_y(path):
    y = get_tensorboard_y_data(path)
    y_smooth = tensorboard_smooth(y, 0.999)

    return y_smooth

def plot_xy(x,y):
    fig = plt.figure("Entrenamiento búsqueda aleatoria")
    fig.suptitle("Entrenamiento búsqueda aleatoria")
    ax = fig.add_subplot(111)

    ax.plot(x, y)
    plt.yscale("log")
    plt.show()

def format_int(num):
    """
        turns to format 7 -> 007
    """
    digits =len(str(num))
    return '0'*(3 - digits) + str(num)

if __name__ == "__main__":
    path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/logs_final/final_with_2023_10_20-00_53_37/{}/execution0/validation"#/events.out.tfevents.1697834025.hybris.local.81912.268.v2"

    all_ys = []
    for i in range(120):
        y = get_smooth_y(path.format(format_int(i)))
        all_ys.append(y)
        print(f"Processed experiment #{i}")

    y_mean = np.stack(all_ys).mean(axis=0)
    y_std = np.stack(all_ys).std(axis=0)
    y_min = np.stack(all_ys).min(axis=0)

    x = np.arange(12000)

    fig, ax =plt.subplots(1)
    ax.plot(x, y_mean, lw=2, label='mean population 1', color='blue')

    best_train_y = get_tensorboard_y_data(path.format{'089'}) # all_ys[89]
    #ax.plot(x, all_ys[89], lw=2, label='mean population 1', color='red')
    ax.plot(x, all_ys[89], lw=2, label='mean population 1', color='red')
    ax.fill_between(x, y_mean+y_std, y_min, facecolor='blue', alpha=0.4)


    #ax.plot(x, y)
    plt.yscale("log")
    plt.show()

