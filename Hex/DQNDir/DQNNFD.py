import tensorflow as tf
from tensorflow.keras.layers import Layer


class NoisyFactorisedDense(Layer):
    def __init__(self, units, sigma_init=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
    
    def build(self, input_shape):
        # nun of neurons from like the layer before
        self.in_features = input_shape[-1]
        limit = 1.0 / tf.sqrt(float(self.in_features))
        mu_initializer_w = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        mu_initializer_b = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        # The mean weights
        self.mu_w = self.add_weight(shape=(self.in_features, self.units), initializer=mu_initializer_w, name='mu_w')
        self.mu_b = self.add_weight(shape=(self.units, ), initializer=mu_initializer_b, name='mu_b')

        # Stand deviation guys, Initialization scaled by number of input neurons
        self.sigma_w = self.add_weight(shape=(self.in_features, self.units), initializer=tf.keras.initializers.Constant(self.sigma_init / tf.sqrt(float(self.in_features))), name='sigma_w')
        self.sigma_b = self.add_weight(shape=(self.units, ), initializer=tf.keras.initializers.Constant(self.sigma_init / tf.sqrt(float(self.in_features))), name='sigma_b')
        
        super().build(input_shape)
    
    def call(self, inputs):
        inputs = tf.cast(inputs, self.compute_dtype)
        noise_dtype = self.compute_dtype

        # Noise for in and out
        epsilon_in = tf.random.normal(shape=(self.in_features,), dtype=noise_dtype)
        epsilon_out = tf.random.normal(shape=(self.units,), dtype=noise_dtype)

        # Factorised noise
        f = lambda x: tf.sign(x) * tf.sqrt(tf.abs(x))
        epsilon_w = tf.tensordot(f(epsilon_in), f(epsilon_out), axes=0)
        epsilon_b = f(epsilon_out)

        # weights and biases but noisy
        w = self.mu_w + self.sigma_w* epsilon_w   
        b = self.mu_b + self.sigma_b* epsilon_b

        return tf.matmul(inputs, w) + b
