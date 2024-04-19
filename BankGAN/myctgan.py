# Based on https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py 
import tensorflow as tf
import math
import random
import os
import numpy as np
# Set seeds
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'



class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        bound = 1/math.sqrt(input_dim)
        w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
        b_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        #x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
        x = self.batch_norm(x)
        x = tf.keras.layers.ReLU()(x)
        out = tf.concat([x, inputs],1)
        return out

class Dense(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
     super(Dense, self).__init__()
     bound = 1/math.sqrt(input_dim)
     w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
     self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
     b_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
     self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )
  def call(self,inputs):
      return tf.matmul(inputs, self.w) + self.b

class DiscriminatorLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim, output_dim):
        super(DiscriminatorLayer, self).__init__()
        bound = 1/math.sqrt(input_dim)
        w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
        b_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )
    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = tf.keras.layers.LeakyReLU(alpha = 0.2)(x)
        out = tf.keras.layers.Dropout(rate = 0.5)(x)
        return out




class CTGAN(object):
    """Conditional Table GAN Synthesizer.
    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        
    """

    def __init__(self, embedding_dim=100, generator_dim=(256,128), discriminator_dim=(128, 256), 
                  pac=10):
        

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self.pac = pac
    
        
    def make_generator(self, sampler, transformer):
        inp_gen = tf.keras.layers.Input(shape=(self._embedding_dim + sampler.dim_cond_vec(),))
        input_dim = self._embedding_dim + sampler.dim_cond_vec()
        x = inp_gen
        for output_dim in self._generator_dim:
            x = ResidualLayer(input_dim, output_dim)(x)
            input_dim += output_dim
        x = Dense(input_dim, transformer.output_dimensions)(x)

        generator = tf.keras.models.Model(inp_gen, x)

        return generator

    def make_discriminator(self, sampler, transformer ):
        data_dim = transformer.output_dimensions
        dim_cond_vec = sampler.dim_cond_vec()
        input_dim = data_dim + dim_cond_vec
        pacdim = input_dim * self.pac
        inp_disc = tf.keras.layers.Input(shape=(pacdim,))
        input_dim = pacdim
        x = inp_disc
        for output_dim in self._discriminator_dim:
            x = DiscriminatorLayer(input_dim, output_dim)(x)
            input_dim = output_dim
        x = Dense(output_dim, 1)(x)

        discriminator = tf.keras.models.Model(inp_disc, x)
        return discriminator
