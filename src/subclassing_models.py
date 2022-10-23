from unicodedata import name
import tensorflow as tf
from functools import reduce

# CREATE RECURRENT MODEL -------------------------------------------------------
class DeformationTrackerModel(tf.keras.Model):
    def __init__(self, log_dir="./logs", **kwargs):
        super().__init__(kwargs)
        self.hidden1 = tf.keras.layers.SimpleRNN(
            50,
            return_sequences=True,
            input_shape=[None, 5],
            name="first_hidden",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
        )
        self.hidden2 = tf.keras.layers.SimpleRNN(
            50,
            return_sequences=True,
            name="second_hidden",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
        )
        self.output_layer = tf.keras.layers.Dense(
            2,
            name="dense_out",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
        )
        self.__use_teacher_forcing__ = True
        self.log_dir = log_dir

    def setTeacherForcing(self, useTeacherForcing: bool):
        self.__use_teacher_forcing__ = useTeacherForcing

    # inputs = (cp_input, finger_input)
    def call(self, model_input):
        control_point_input, finger_input = model_input

        if self.__use_teacher_forcing__:  # With teacher forcing
            print("Using teacher forcing")
            layer_input = tf.keras.layers.Concatenate()(
                [control_point_input, finger_input]
            )
            hidden1 = self.hidden1(layer_input)
            hidden2 = self.hidden2(hidden1)
            model_output = self.output_layer(hidden2)
            return model_output

        else:  # No teacher forcing
            print("Not using teacher forcing")
            layer_output = control_point_input[
                :, :1, :
            ]  # first control point of the seq
            layer_outputs = []

            # TODO fix: WARNING:tensorflow:5 out of the last 9 calls to <function Model.make_predict_function.<locals>.predict_function at 0x7f5a62e04e50> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
            execute_model = lambda input_data: self.output_layer(
                self.hidden2(self.hidden1(tf.keras.layers.Concatenate()(input_data)))
            )
            for i in range(100):
                # tf.keras.backend.clear_session() # to solve this: https://stackoverflow.com/questions/66712301/creating-models-in-a-loop-makes-keras-increasingly-slower
                # next_layer_input = tf.keras.layers.Concatenate()([layer_output, finger_input[:,i:i+1,:]])
                # hidden1 = self.hidden1(next_layer_input)
                # hidden2 = self.hidden2(hidden1)
                # layer_output = self.output_layer(hidden2)
                layer_output = execute_model(
                    [layer_output, finger_input[:, i : i + 1, :]]
                )
                layer_outputs.append(layer_output)
            concat_func = lambda x, y: tf.keras.layers.Concatenate(axis=1)([x, y])
            model_output = reduce(concat_func, layer_outputs)
            return model_output


class DeformationTrackerBiFlowModel(tf.keras.Model):
    def __init__(self, log_dir="./logs", **kwargs):
        super().__init__(kwargs)
        self.hidden1 = tf.keras.layers.SimpleRNN(
            12,
            return_sequences=True,
            input_shape=[None, 5],
            name="first_hidden",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
        )
        self.output_layer = tf.keras.layers.Dense(
            2,
            name="dense_out",
            kernel_initializer="random_normal",
            bias_initializer="zeros",
        )
        self.__use_teacher_forcing__ = True
        self.log_dir = log_dir

    def setTeacherForcing(self, useTeacherForcing: bool):
        self.__use_teacher_forcing__ = useTeacherForcing

    # inputs = (cp_input, finger_input)
    def call(self, model_input):
        control_point_input, finger_input = model_input

        if self.__use_teacher_forcing__:  # With teacher forcing
            print("Using teacher forcing")
            layer_input = tf.keras.layers.Concatenate()(
                [control_point_input, finger_input]
            )
            hidden1 = self.hidden1(layer_input)
            last_layer_input = tf.keras.layers.Concatenate()(
                [control_point_input, hidden1]
            )
            model_output = self.output_layer(last_layer_input)
            return model_output

        else:  # No teacher forcing
            print("Not using teacher forcing")
            layer_output = control_point_input[
                :, :1, :
            ]  # first control point of the seq
            layer_outputs = []

            # TODO fix: WARNING:tensorflow:5 out of the last 9 calls to <function Model.make_predict_function.<locals>.predict_function at 0x7f5a62e04e50> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
            # execute_model = lambda input_data: self.output_layer(
            #     self.hidden2(self.hidden1(tf.keras.layers.Concatenate()(input_data)))
            # )
            for i in range(100):
                # tf.keras.backend.clear_session() # to solve this: https://stackoverflow.com/questions/66712301/creating-models-in-a-loop-makes-keras-increasingly-slower
                next_layer_input = tf.keras.layers.Concatenate()([layer_output, finger_input[:,i:i+1,:]])
                hidden1 = self.hidden1(next_layer_input)
                last_layer_input = tf.keras.layers.Concatenate()(
                    [
                    control_point_input[
                        :, :1, :
                    ],
                    # layer_output,
                    hidden1
                    ]
                )
                layer_output = self.output_layer(last_layer_input)
                # layer_output = execute_model(
                #     [layer_output, finger_input[:, i : i + 1, :]]
                # )
                layer_outputs.append(layer_output)
            concat_func = lambda x, y: tf.keras.layers.Concatenate(axis=1)([x, y])
            model_output = reduce(concat_func, layer_outputs)
            return model_output
