"""
Plot the min, mix and average of all the random search experiments
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from F1_result_plot_learning import get_smooth_y

matplotlib.use("QtAgg")

if __name__ == "__main__":
    train_path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/src/final_experiment/logs/best_params_with_teacher/experiment_2023_10_30-21_26_05/train"
    validation_path = "/Users/ndroid/Documents/tesis/repos/Deformation-Tracker/src/final_experiment/logs/best_params_with_teacher/experiment_2023_10_30-21_26_05/validation"

    train_history = get_smooth_y(train_path, data_size=36000, smooth_ratio=0.01)
    validation_history = get_smooth_y(validation_path, data_size=36000, smooth_ratio=0.01)

    x = np.arange(36000)

    fig, ax =plt.subplots(1)

    # Train plot
    ax.plot(x, train_history, lw=2, label='Conjunto de entrenamiento', color='dodgerblue', alpha=0.5)

    # Validation plot
    ax.plot(x, validation_history, lw=2, label='Conjunto de validación', color='darkorange', alpha=0.5)

    plt.xlabel("Épocas")
    plt.ylabel("Error (MSE)")
    ax.legend()
    plt.yscale("log")
    plt.show()
