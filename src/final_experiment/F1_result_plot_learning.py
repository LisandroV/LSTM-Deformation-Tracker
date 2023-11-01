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
    'tensors': 12000
}

def get_tensorboard_y_data(path, data_size: int):
    event_acc = ea.EventAccumulator(path, {'tensors': data_size})
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

def get_smooth_y(path, data_size=12000, smooth_ratio=0.999):
    y = get_tensorboard_y_data(path, data_size)
    y_smooth = tensorboard_smooth(y, smooth_ratio)

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


def get_log_summary(path):
    """
        Returns all the Y values from a given tensorboard log path
        * y_mean
        * y_std
        * y_min
    """
    summary = {}
    all_ys = []
    for i in range(120):
        y = get_smooth_y(path.format(format_int(i)))
        all_ys.append(y)
        print(f"Processed experiment #{i}")

    summary['all_ys'] = all_ys
    summary['y_mean'] = np.stack(all_ys).mean(axis=0)
    summary['y_std'] = np.stack(all_ys).std(axis=0)
    summary['y_min'] = np.stack(all_ys).min(axis=0)

    return summary

def save_summary(file_name, summary):
    # must create this dir before using
    with open(f"./src/final_experiment/tmp/1_random_search/{file_name}", 'wb') as f:
        np.save(f, summary['all_ys'])
        np.save(f, summary['y_mean'])
        np.save(f, summary['y_std'])
        np.save(f, summary['y_min'])

def load_summary(file_name):
    with open(f"./src/final_experiment/tmp/1_random_search/{file_name}", 'rb') as f:
        summary = {}
        summary['all_ys'] = np.load(f)
        summary['y_mean'] = np.load(f)
        summary['y_std'] = np.load(f)
        summary['y_min'] = np.load(f)

        return summary

if __name__ == "__main__":
    train_path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/src/final_experiment/logs/random_search_with_teacher/experiment_2023_10_29-23_54_12/{}/execution0/train"
    validation_path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/src/final_experiment/logs/random_search_with_teacher/experiment_2023_10_29-23_54_12/{}/execution0/validation"

    try:
        train_summary = load_summary("train_graph_summary.npy")
        validation_summary = load_summary("validation_graph_summary.npy")
        print("Loaded summary files.")
    except:
        print("Error loading summary files, calculating summary.")
        train_summary = get_log_summary(train_path)
        save_summary("train_graph_summary.npy", train_summary)
        validation_summary = get_log_summary(validation_path)
        save_summary("validation_graph_summary.npy", validation_summary)

    x = np.arange(12000)

    fig, ax =plt.subplots(1)

    # Train plot
    train_best_y =  train_summary['all_ys'][104] # get_tensorboard_y_data(path.format('089'))
    ax.plot(x, train_summary['y_mean'], lw=2, label='Promedio en conjunto de entrenamiento', color='dodgerblue')
    #ax.plot(x, train_best_y, lw=2, label='mean population 1', color='red')
    ax.fill_between(x, train_summary['y_mean']+train_summary['y_std'], train_summary['y_min'], facecolor='dodgerblue', alpha=0.4)

    # Validation plot
    validation_best_y =  validation_summary['all_ys'][104] # get_tensorboard_y_data(path.format('089'))
    ax.plot(x, validation_summary['y_mean'], lw=2, label='Promedio en conjunto de validacion', color='darkorange')
    ax.plot(x, validation_best_y, lw=2, label='Mejor resultado en conjunto de validación', color='magenta')
    ax.fill_between(x, validation_summary['y_mean']+validation_summary['y_std'], validation_summary['y_min'], facecolor='orange', alpha=0.4)

    plt.xlabel("Épocas")
    plt.ylabel("Error (MSE)")
    ax.legend()
    plt.yscale("log")
    plt.show()
