from tensorflow.keras.layers import Input, Conv2D, Flatten, Concatenate, ReLU
from tensorflow.keras.models import Model
import tensorflow as tf
from DQNNFD import *
# assume NoisyFactorisedDense is defined/imported
board_input = Input(shape=(5,5,2), name='board_input')
swap_input = Input(shape=(1,), name='swap_input')

x = Conv2D(128, 3, padding='same', activation='relu', name='local_patterns')(board_input)
x = Conv2D(128, 3, padding='same', activation='relu', name='global_patterns')(x)
x = Conv2D(128, 3, padding='same', activation='relu', name='pattern_combinations')(x)
x_flat = Flatten()(x)
concatenated = Concatenate()([x_flat, swap_input])
d = NoisyFactorisedDense(256, name='decision_layer_1')(concatenated)
d = ReLU()(d)
d = NoisyFactorisedDense(128, name='decision_layer_2')(d)
d = ReLU()(d)

model = Model(inputs=[board_input, swap_input], outputs=d)
# Then:
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=False, dpi=96, to_file='model.png')