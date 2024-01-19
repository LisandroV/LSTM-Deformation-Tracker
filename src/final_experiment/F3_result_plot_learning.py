"""
Plot the min, mix and average of all the random search experiments without teacher forcing
"""
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea
from F1_result_plot_learning import load_summary, get_distributed_log_summary, save_summary
import matplotlib
import math
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")

if __name__ == "__main__":
    train_path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/src/final_experiment/logs/random_search_without_teacher/e{}_rs/experiment_2023_11_11-02_27_35/{}/execution0/train"
    validation_path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/src/final_experiment/logs/random_search_without_teacher/e{}_rs/experiment_2023_11_11-02_27_35/{}/execution0/validation"
    epochs = 18000
    smooth_ratio = 0.3
    plot_limit = 9000

    try:
        train_summary = load_summary("without_teacher_train_summary.npy")
        validation_summary = load_summary("without_teacher_validation_summary.npy")
        print("Loaded summary files.")
    except:
        print("Error loading summary files, calculating summary.")
        train_summary = get_distributed_log_summary(train_path, epochs, smooth_ratio)
        save_summary("without_teacher_train_summary.npy", train_summary)
        validation_summary = get_distributed_log_summary(validation_path, epochs, smooth_ratio)
        save_summary("without_teacher_validation_summary.npy", validation_summary)

    x = np.arange(plot_limit)

    fig, ax =plt.subplots(1)

    # Train plot
    train_best_y =  train_summary['all_ys'][42] # get_tensorboard_y_data(path.format('089'))
    ax.plot(x, train_summary['y_mean'][:plot_limit], lw=2, label='Promedio en conjunto de entrenamiento', color='dodgerblue')
    #ax.plot(x, train_best_y, lw=2, label='mean population 1', color='red')
    ax.fill_between(x, train_summary['y_mean'][:plot_limit]+train_summary['y_std'][:plot_limit], train_summary['y_min'][:plot_limit], facecolor='dodgerblue', alpha=0.4)

    # Validation plot
    validation_best_y =  validation_summary['all_ys'][42] # get_tensorboard_y_data(path.format('089'))
    ax.plot(x, validation_summary['y_mean'][:plot_limit], lw=2, label='Promedio en conjunto de validacion', color='darkorange')
    ax.plot(x, validation_best_y[:plot_limit], lw=2, label='Mejor resultado en conjunto de validación', color='magenta')
    ax.fill_between(x, validation_summary['y_mean'][:plot_limit]+validation_summary['y_std'][:plot_limit], validation_summary['y_min'][:plot_limit], facecolor='orange', alpha=0.4)

    # min validation arrow
    ax.annotate('mínimo en\nvalidación', xy=(30, 0.004957165), xytext=(800, 0.003), va='top', ha='left', arrowprops=dict(facecolor='black', shrink=0.05))


    plt.xlabel("Épocas")
    plt.ylabel("Error (MSE)")
    ax.legend()
    plt.yscale("log")
    plt.show()
