"""

"""

import os
import numpy as np
import sys
sys.path.append('./src')
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to supress tf warnings
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from concave_hull import concave_hull_indexes


from utils.script_arguments import get_script_args
from final_experiment.dataset import create_datasets, create_test_dataset
import plots.dataset_plotter as plotter
from subclassing_models import DeformationTrackerBiFlowModel as DeformationTrackerModel


STORED_MODEL_DIR: str = "src/final_experiment/saved_models/best_e2_rs"
CHECKPOINT_MODEL_DIR: str = f"{STORED_MODEL_DIR}/checkpoint/"


if __name__ == "__main__":
    train_dataset, validation_dataset = create_datasets()

    model = DeformationTrackerModel()
    model.compile(loss="mse", optimizer="adam")
    model.setTeacherForcing(False)

    # LOAD MODEL -------------------------------------------------------------------
    prev_model = keras.models.load_model(
        STORED_MODEL_DIR,
        custom_objects={"DeformationTrackerModel": DeformationTrackerModel},
    )

    model.setTeacherForcing(True)
    model.build(input_shape=[(None, 100, 2), (None, 100, 4)])  # init model weights
    model.set_weights(prev_model.get_weights())  # to use last model
    #model.load_weights(CHECKPOINT_MODEL_DIR)  # to use checkpoint
    print("Using stored model.")

    model.setTeacherForcing(False)

    finger_position_plot = lambda positions: lambda ax: ax.scatter(
        range(100), positions[:, 0], positions[:, 1]*-1, s=10
    )

    # # PREDICTION TRINING SET -------------------------------------------------------------------
    # y_pred = model.predict([train_dataset['X_control_points'][:47,:1,:], train_dataset['X_finger'][:47,:,:]])
    # for frame_number in range(100):
    #     scale = 200
    #     polygon_center = [623.30482589, 497.98404272]
    #     scale = 196.7508968341383

    #     finger_data = train_dataset['X_finger'][1,:,:2]

    #     concave_hull = list(concave_hull_indexes(y_pred[:,0,:], length_threshold=0.05,))
    #     concave_hull.append(concave_hull[0])

    #     frame_points = y_pred[:,frame_number,:].take(concave_hull,axis=0)

    #     img = mpimg.imread(f'data/sponge_centre/images/frame{frame_number}.jpg')
    #     plt.suptitle(f"Predicción {frame_number + 1}")
    #     plt.imshow(img)
    #     plt.scatter( # finger position
    #         finger_data[frame_number,0]*scale + polygon_center[0],
    #         finger_data[frame_number,1]*scale + polygon_center[1],
    #         color='lime', s=100
    #     )
    #     plt.scatter( # control points
    #         frame_points[:,0]*scale + polygon_center[0],
    #         frame_points[:,1]*scale + polygon_center[1],
    #         color='cyan', s=30
    #     )
    #     # link points with line
    #     plt.plot(
    #         frame_points[:,0]*scale + polygon_center[0],
    #         frame_points[:,1]*scale + polygon_center[1],
    #         color='cyan'
    #     )

    #     plt.xlim([390, 850])
    #     plt.ylim([250, 650])
    #     plt.gca().invert_yaxis()
    #     #plt.show()
    #     plt.savefig(f"./src/final_experiment/tmp/prediction_gif/best_on_validation_set/sponge_centre/images/{frame_number}.png")
    #     plt.clf()


    # PREDICTION VALIDATION SET -------------------------------------------------------------------
    # y_pred = model.predict([validation_dataset['X_control_points'][:47,:1,:], validation_dataset['X_finger'][:47,:,:]])
    # for frame_number in range(100):
    #     scale = 200
    #     polygon_center = [508.94235277, 506.56720458] # normalization: means[0]
    #     scale = 202.29362620517776 # normalization: scale

    #     finger_data = validation_dataset['X_finger'][1,:,:2]

    #     concave_hull = list(concave_hull_indexes(y_pred[:,0,:], length_threshold=0.05,))
    #     concave_hull.append(concave_hull[0])

    #     frame_points = y_pred[:,frame_number,:].take(concave_hull,axis=0)

    #     img = mpimg.imread(f'data/sponge_longside/images/frame{frame_number}.jpg')
    #     plt.suptitle(f"Predicción {frame_number + 1}")
    #     plt.imshow(img)
    #     plt.scatter( # finger position
    #         finger_data[frame_number,0]*scale + polygon_center[0],
    #         finger_data[frame_number,1]*scale + polygon_center[1],
    #         color='lime', s=100
    #     )
    #     plt.scatter( # control points
    #         frame_points[:,0]*scale + polygon_center[0],
    #         frame_points[:,1]*scale + polygon_center[1],
    #         color='cyan', s=30
    #     )
    #     # link points with line
    #     plt.plot(
    #         frame_points[:,0]*scale + polygon_center[0],
    #         frame_points[:,1]*scale + polygon_center[1],
    #         color='cyan'
    #     )

    #     plt.xlim([250, 750])
    #     plt.ylim([300, 650])
    #     plt.gca().invert_yaxis()
    #     #plt.show()
    #     plt.savefig(f"./src/final_experiment/tmp/prediction_gif/best_on_validation_set/sponge_longside/images/{frame_number}.png")
    #     plt.clf()


    # PREDICTION TEST SET -------------------------------------------------------------------
    test_dataset = create_test_dataset()
    y_pred = model.predict([test_dataset['X_control_points'][:47,:1,:], test_dataset['X_finger'][:47,:,:]])
    for frame_number in range(100):
        scale = 200
        polygon_center = [609.09286834, 427.11354844] # normalization: means[0]
        scale = 199.36526475896835 # normalization: scale

        finger_data = test_dataset['X_finger'][1,:,:2]

        concave_hull = list(concave_hull_indexes(y_pred[:,0,:], length_threshold=0.05,))
        concave_hull.append(concave_hull[0])

        frame_points = y_pred[:,frame_number,:].take(concave_hull,axis=0)

        img = mpimg.imread(f'data/sponge_shortside/images/frame{frame_number}.jpg')
        plt.suptitle(f"Predicción {frame_number + 1}")
        plt.imshow(img)
        plt.scatter( # finger position
            finger_data[frame_number,0]*scale + polygon_center[0],
            finger_data[frame_number,1]*scale + polygon_center[1],
            color='lime', s=100
        )
        plt.scatter( # control points
            frame_points[:,0]*scale + polygon_center[0],
            frame_points[:,1]*scale + polygon_center[1],
            color='cyan', s=30
        )
        # link points with line
        plt.plot(
            frame_points[:,0]*scale + polygon_center[0],
            frame_points[:,1]*scale + polygon_center[1],
            color='cyan'
        )

        plt.xlim([370, 870])
        plt.ylim([120, 670])
        plt.gca().invert_yaxis()
        #plt.show()
        plt.savefig(f"./src/final_experiment/tmp/prediction_gif/best_on_validation_set/sponge_shortside/images/{frame_number}.png")
        plt.clf()
