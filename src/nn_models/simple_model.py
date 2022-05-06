# bibliography:
# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)
# Call the layer on the first three rows of the DataFrame to visualize an example of the output from this layer:

normalizer(numeric_features.iloc[:3])


# tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)


# https://medium.com/when-i-work-data/converting-a-pandas-dataframe-into-a-tensorflow-dataset-752f3783c168