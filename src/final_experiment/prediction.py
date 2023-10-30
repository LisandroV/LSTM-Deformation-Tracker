"""
    The only purpose of this file is to see the performance of the trained models.
"""

import os
import numpy as np
import sys
sys.path.append('./src')
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import tensorflow as tf
from tensorflow import keras
from concave_hull import concave_hull_indexes


from utils.script_arguments import get_script_args
from dataset import create_datasets
import plots.dataset_plotter as plotter
from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel

np.random.seed(42)
tf.random.set_seed(42)

script_args = get_script_args()

# STORED_MODEL_DIR: str = "saved_models/best_11_no_teacher_subclassing_100n"
STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_best_params_with_teacher/"
# STORED_MODEL_DIR: str = "saved_models/best_12_random_search_50n_11"
CHECKPOINT_MODEL_DIR: str = f"{STORED_MODEL_DIR}/checkpoint/"
# CHECKPOINT_MODEL_DIR: str = f"{STORED_MODEL_DIR}/checkpoint/train/"
SHOULD_TRAIN_MODEL: bool = script_args.train

train_dataset, validation_dataset = create_datasets()

model = DeformationTrackerModel()

# print(model.summary())

model.compile(loss="mse", optimizer="adam")
model.setTeacherForcing(False)

# try:
prev_model = keras.models.load_model(
    STORED_MODEL_DIR,
    custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
)

model.setTeacherForcing(True)
model.build(input_shape=[(None, 100, 2), (None, 100, 4)])  # init model weights
#model.set_weights(prev_model.get_weights())  # to use last model
model.load_weights(CHECKPOINT_MODEL_DIR)  # to use checkpoint
print("Using stored model.")

# EVALUACION --------------------------------------------------------------------
# model.setTeacherForcing(False)
# train_loss = model.evaluate(
#     [train_dataset['X_control_points'][:,:1,:], train_dataset['X_finger']],
#     train_dataset['Y']
# )
# print(f"Stored model loss on training set: {train_loss}")
# validation_loss = model.evaluate(
#     [validation_dataset['X_control_points'][:,:1,:], validation_dataset['X_finger']],
#     validation_dataset['Y']
# )
# print(f"Stored model loss on validation set: {validation_loss}")



# DATASET PLOT -----------------------------------------------------------------
for i in range(100):
    plotter.scatter_plot(
        train_dataset['X_control_points'][:47,i,:],
        finger_position = train_dataset['finger_position'][100-i,:]
    )

import ipdb;ipdb.set_trace();

# PREDICTION -------------------------------------------------------------------

# ONE-STEP PREDICTION
model.setTeacherForcing(True)

finger_position_plot = lambda positions: lambda ax: ax.scatter(
    range(100), positions[:, 0], positions[:, 1]*-1, s=10
)

y_pred = model.predict([validation_dataset['X_control_points'], validation_dataset['X_finger']])
predicted_polygons = y_pred.swapaxes(0, 1)



POLIGON_NUMBER = 10

fig = plt.figure(f"Predicción #{POLIGON_NUMBER}")
fig.suptitle(f"predicción #{POLIGON_NUMBER}")
ax = fig.add_subplot(111)

ax.scatter(train_dataset['X_control_points'][:,POLIGON_NUMBER,0], train_dataset['X_control_points'][:,POLIGON_NUMBER,1]*-1, color='red', s=30)
ax.scatter(validation_dataset['finger_position'][POLIGON_NUMBER,0], validation_dataset['finger_position'][POLIGON_NUMBER,1]*-1, color='dodgerblue', s=130)
poligon_indexes = list(concave_hull_indexes(predicted_polygons[POLIGON_NUMBER,:,:], length_threshold=0.05,))
poligon_indexes.append(poligon_indexes[0])
#poligon_indexes = [ 3, 38, 0, 25, 16, 39 ,6, 33, 24, 43, 7, 9, 34, 20, 15, 41, 26, 1, 36, 12, 28, 19, 44, 4, 27, 14, 37, 11, 23, 42, 13, 30, 18, 35, 2, 46, 22, 8, 32, 10, 45, 21, 31, 5, 40, 17, 29, 3] # 50
print(poligon_indexes)
ax.plot(
    predicted_polygons[POLIGON_NUMBER,:,0].take(poligon_indexes,axis=0), # X
    predicted_polygons[POLIGON_NUMBER,:,1].take(poligon_indexes,axis=0)*-1, # Y
    color='black',
    alpha=0.5
)
# for index, p in enumerate(predicted_polygons[POLIGON_NUMBER,:,:]):
#     ax.annotate(str(index),(p[0],p[1]*-1))

ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()


# PREDICTION ---------------------------------------------------------------------------------

# plotter.plot_npz_control_points(predicted_polygons, plot_cb=finger_position_plot(validation_dataset['finger_position']))


# MULTIPLE-STEP PREDICTION
# model.setTeacherForcing(False)

# finger_position_plot = lambda positions: lambda ax: ax.scatter(
#     range(100), positions[:, 0], positions[:, 1]*-1, s=10
# )

# y_pred = model.predict([validation_dataset['X_control_points'][:,:1,:], validation_dataset['X_finger']])
# predicted_polygons = y_pred.swapaxes(0, 1)
# plotter.plot_npz_control_points(predicted_polygons, plot_cb=finger_position_plot(validation_dataset['finger_position']))
