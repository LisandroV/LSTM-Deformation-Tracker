from unicodedata import name
import tensorflow as tf
from functools import reduce

# CREATE RECURRENT MODEL -------------------------------------------------------
class DeformationTrackerModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.hidden1 = tf.keras.layers.SimpleRNN(15, return_sequences=True, input_shape=[None, 5], name="first_hidden", kernel_initializer='random_normal',bias_initializer='zeros')
        self.hidden2 = tf.keras.layers.SimpleRNN(15, return_sequences=True, name="second_hidden", kernel_initializer='random_normal',bias_initializer='zeros')
        self.output_layer = tf.keras.layers.Dense(2, name="dense_out", kernel_initializer='random_normal',bias_initializer='zeros')
        self.__use_teacher_forcing__ = True

    def setTeacherForcing(self, useTeacherForcing: bool):
        self.__use_teacher_forcing__ = useTeacherForcing

    # inputs = (cp_input, finger_input)
    def call(self, model_input):
        control_point_input, finger_input = model_input

        if self.__use_teacher_forcing__: # With teacher forcing
            print("Using teacher forcing")
            layer_input = tf.keras.layers.Concatenate()([control_point_input, finger_input])
            hidden1 = self.hidden1(layer_input)
            hidden2 = self.hidden2(hidden1)
            model_output = self.output_layer(hidden2)
            return model_output

        else: # No teacher forcing
            print("Not using teacher forcing")
            layer_output = control_point_input[:,:1,:] # first control point of the seq
            layer_outputs=[]
            for i in range(100):
                # tf.keras.backend.clear_session() # to solve this: https://stackoverflow.com/questions/66712301/creating-models-in-a-loop-makes-keras-increasingly-slower
                next_layer_input = tf.keras.layers.Concatenate()([layer_output, finger_input[:,i:i+1,:]])# init layer input
                hidden1 = self.hidden1(next_layer_input)
                hidden2 = self.hidden2(hidden1)
                layer_output = self.output_layer(hidden2)
                layer_outputs.append(layer_output)
            concat_func = lambda x, y: tf.keras.layers.Concatenate(axis=1)([x, y])
            model_output = reduce(concat_func, layer_outputs)
            return model_output