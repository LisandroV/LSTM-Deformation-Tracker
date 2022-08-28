from tensorflow import keras, saved_model
import numpy as np


def save_best_model(
    new_model: keras.models,
    stored_model_name: str,
    X_data: np.ndarray,
    y_data: np.ndarray,
):
    """
    Loads the stored model, compares it with the new one, and replaces it if its better
    """
    new_model_error = new_model.evaluate(X_data, y_data)
    try:
        stored_model = keras.models.load_model(stored_model_name)
        stored_model_error = stored_model.evaluate(X_data, y_data)
    except:  # if there was no model saved
        stored_model_error = new_model_error + 1

    print(f"New model error: {new_model_error}")
    print(f"Stored model error:{stored_model_error}")
    if new_model_error < stored_model_error:
        print(
            f"Model was better than the previous. It was saved in: {stored_model_name}"
        )
        new_model.save(stored_model_name)
    else:
        print("New model was not better than the stored one")

def save_best_subclassing_model(
    new_model: keras.models,
    stored_model_name: str,
    X_data: np.ndarray,
    y_data: np.ndarray,
):
    """
    Loads the stored model, compares it with the new one, and replaces it if its better
    """
    new_model_error = new_model.evaluate(X_data, y_data)
    try:
        stored_model = saved_model.load(stored_model_name)
        stored_model_error = stored_model.evaluate(X_data, y_data)
    except:  # if there was no model saved
        stored_model_error = new_model_error + 1

    print(f"New model error: {new_model_error}")
    print(f"Stored model error:{stored_model_error}")
    if new_model_error < stored_model_error:
        print(
            f"Model was better than the previous. It was saved in: {stored_model_name}"
        )
        saved_model.save(new_model, stored_model_name)
    else:
        print("New model was not better than the stored one")
