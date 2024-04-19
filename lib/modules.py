
import math
import tensorflow as tf 
import numpy as np
import random
import os

# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

def RandomNoise_Simulator_Wiener(samples, timesteps, features):
    '''
    generates random sequences from a Wiener process.
    '''
    z = tf.random.normal(mean=0, stddev=1, shape=(samples * timesteps, features), dtype=tf.float32)
    z = tf.cumsum(z, axis=0) / tf.sqrt(tf.cast(samples * timesteps, dtype=tf.float32))
    z = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    z = tf.reshape(z, (samples, timesteps, features))
    return z


def RandomNoise_Simulator_Normal(samples, timesteps, features):
    '''
    generates random sequences from a Gaussian Distribution.
    '''
    z = tf.random.normal(mean=0, stddev=1, shape=(samples * timesteps, features), dtype=tf.float32)
    z = tf.reshape(z, (samples, timesteps, features))
    return z

def get_angles(pos, i, d_model):
    """pos is a tuple (256=maximum_position_encoding,1), 
       i is a tuple (1, 128=d_model(feature embedding dimension)) """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))    #angle_rates(1,d_model=128)
    return pos * angle_rates     #(maximum_position_encoding,feature embedding dimension)  ex.(256, 128)
    
def positional_encoding(position, d_model):
    """position : an Integer number shows the maximum positions(timesteps) for encoding
       d_model:  an Integer shows the number of embedding features"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],    #(position=256,1)
                          np.arange(d_model)[np.newaxis, :],       #(1, d_model=128)
                          d_model)
    #angle_rads (256, 128)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]    #(1,256,128)

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len), a triangle matrix, upper are ones and, lower and main diagonal are zeroes

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(tf.reduce_sum(seq, axis=2), 0), tf.float32)
  
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(tar):

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask, dec_target_padding_mask    # combined_mask : (batch_size, 1, seq_len, seq_len)




class Dense(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, activation):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        bound = 1 / math.sqrt(self.input_dim)
        self.w = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer=tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound),
            trainable=True,
        )

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)

    def build(self, input_shape):
        bound = 1 / math.sqrt(self.input_dim)
        self.w = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer=tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound),
            trainable=True,
        )
        
    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = self.batch_norm(x)
        x = tf.keras.layers.ReLU()(x)
        out = tf.concat([x, inputs], axis= -1)
        return out




class InputEmbedLayer_Res(tf.keras.layers.Layer):
    def __init__(self, features, res_dims, d_embedding):
        super(InputEmbedLayer_Res, self).__init__()

        self.res_layers = []
        input_dim = features
        for res_dim in res_dims:
            self.res_layers.append(ResidualLayer(input_dim, res_dim))
            input_dim += res_dim
        
        self.dense = Dense(input_dim, d_embedding, tf.keras.activations.linear)

    def call(self, inputs):
        x = inputs
        for res_layer in self.res_layers:
            x = res_layer(x)
        x = self.dense(x)
        return x

class InputEmbedLayer(tf.keras.layers.Layer):
    def __init__(self, features, dff, d_embedding):
        super(InputEmbedLayer, self).__init__()
      
        self.dense1 = Dense(features, dff, tf.keras.activations.relu)
        self.dense2 = Dense(dff, d_embedding, tf.keras.activations.linear)

    def call(self, inputs):
        x = self.dense1(inputs)   
        x = self.dense2(x)       
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_embedding, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.Linear_q = Dense(d_embedding, d_model, tf.keras.activations.linear)
        self.Linear_k = Dense(d_embedding, d_model, tf.keras.activations.linear)
        self.Linear_v = Dense(d_embedding, d_model, tf.keras.activations.linear)

        self.dense = Dense(d_model, d_model, tf.keras.activations.linear)

        

    def call(self, x1, x2, x3, combined_mask):
        batch_size = tf.shape(x1)[0]
        q = self.Linear_q(x1)     # (batch_size, seq_len, d_model)
        k = self.Linear_k(x2)     # (batch_size, seq_len, d_model)
        v = self.Linear_v(x3)     # (batch_size, seq_len, d_model)
        
        #Split Heads 
        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.depth))
        q = tf.transpose(q, perm=[0,2,1,3])                    #(batch_size, num_heads, seq_len, depth)

        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.depth))
        k = tf.transpose(k, perm=[0,2,1,3])                     #(batch_size, num_heads, seq_len, depth)

        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
        v = tf.transpose(v, perm=[0,2,1,3])                      #(batch_size, num_heads, seq_len, depth)

        matmul_qk = tf.matmul(q, k, transpose_b=True)            #scores  (batch_size, num_heads, seq_len_q, seq_len_k)

        # scale scores(matmul_qk)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)     #(batch_size, num_heads, seq_len_q, seq_len_k)

        #add the mask to the scaled tensor
        scaled_attention_logits += (combined_mask * -1e9)           #(batch_size, num_heads, seq_len_q, seq_len_k)

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)    #(batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.matmul(attention_weights, v)             # (batch_size,  num_heads, seq_len_q, depth)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_inp_decoder, d_model, num_heads, dff, rate=0.1):
        """ this layer includes a MultiheadAttentionLayer(MHA), the output of MHA is added to its input and a normalization operation is applied on it, 
        and the result is feeded to a dense layer with dff dimension """
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_inp_decoder, d_model, num_heads)

        self.dense1 = Dense(d_model, dff, tf.keras.activations.relu)
        self.dense2 = Dense(dff, d_model, tf.keras.activations.linear)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    
    def call(self, x, training, mask):

        attn, attn_weights = self.mha(x, x, x, mask)  # (batch_size, target_seq_len, d_model)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(attn + x)

        out = self.dense1(out1)
        FC_out = self.dense2(out)
        FC_out = self.dropout2(FC_out, training=training)
        out2 = self.layernorm3(FC_out + out1)

        return out2, attn_weights

class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_inp_decoder, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.dec_layers = [DecoderLayer(d_inp_decoder, d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.num_layers = num_layers
        
    def call(self, x, training, mask):
        attention_weights = {}
        for i in range(self.num_layers):
            x, attentionweights = self.dec_layers[i](x, training, mask)
            attention_weights['decoder_layer{}'.format(i+1)] = attentionweights
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, features,dff, d_embedding, d_model, maximum_position_encoding,num_heads, num_layers,config, rate=0.1):
       super(Transformer, self).__init__()
   
       self.ORDER = config["ORDER"]
       self.FIELD_STARTS_IN = config["FIELD_STARTS_IN"]
       self.FIELD_DIMS_IN = config["FIELD_DIMS_IN"] 
       self.FIELD_STARTS_NET = config["FIELD_STARTS_NET"]
       self.FIELD_DIMS_NET = config["FIELD_DIMS_NET"]
       self.ACTIVATIONS = config["ACTIVATIONS"]
       
       self.input_layer = tf.keras.Sequential([tf.keras.layers.Input(shape=(None, features)),  InputEmbedLayer(features, dff, d_embedding)])
    #    self.InputLayer = tf.keras.layers.Input(shape=(None, features))
    #    self. InputEmbeddingLayer = InputEmbedLayer(features, dff, d_embedding)
       self.pos_encoding = positional_encoding(maximum_position_encoding, d_embedding)
       self.dropout = tf.keras.layers.Dropout(rate)
       self.DecoderStack = Decoder(num_layers, d_embedding, d_model, num_heads, dff)
       self.final_layer = tf.keras.layers.Dense(d_model, activation=None)

       for name, dim in self.FIELD_DIMS_NET.items():
            acti = self.ACTIVATIONS.get(name, None)
            self.__setattr__(name, tf.keras.layers.Dense(dim, activation=acti))
     

    

    def call(self, inp, tar):
        inp_inp = inp[:, :-1] # predict next from this
        inp_out = inp[:, 1:]

        # input_ = self.InputLayer(inp_inp)
        # x =  self. InputEmbeddingLayer(input_)
        x = self.input_layer(inp_inp)
        
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]     #x is the output of Input layer

        x = self.dropout(x, training=True)

        mask, _ = create_masks(tar)

        out, attention_weights = self.DecoderStack(x, True, mask)

        final_output = self.final_layer(out)
        preds = {}
        
        #print("Final output shape start", final_output.shape)
        for net_name in self.ORDER:
            #print("Running net", net_name)
            pred = self.__getattribute__(net_name)(final_output)
            #print("pred shape", pred.shape)
            preds[net_name] = pred
            
            st = self.FIELD_STARTS_IN[net_name]
            end = st + self.FIELD_DIMS_IN[net_name]
            to_add = inp_out[:, :, st: end]
            #print("Start and end", st, end)
            
            final_output = tf.concat([final_output, to_add], axis=-1)
            #print("Final output shape after",net_name, "is", final_output.shape, "\n")
        
        return preds, attention_weights
    

class Encoder_Decoder_lstm(tf.keras.Model):
    """ conditional=True : conditional training and conditional generating of data 
        flag = True :output for date features is clock-dimension 
    """
    def __init__(self, config, inp_feat, conditional = True, unit=128):
        super(Encoder_Decoder_lstm, self).__init__()

        self.ORDER = config["ORDER"]
        self.FIELD_STARTS_IN = config["FIELD_STARTS_IN"]
        self.FIELD_DIMS_IN = config["FIELD_DIMS_IN"]
        self.FIELD_STARTS_NET = config["FIELD_STARTS_NET"]
        self.FIELD_DIMS_NET = config["FIELD_DIMS_NET"]
        self.ACTIVATIONS = config["ACTIVATIONS"]

        self.conditional = conditional
        self.config = config
        self.inp_feat = inp_feat

        # LSTM layers
        self.encoder_lstm1 = tf.keras.layers.LSTM(units=unit, return_sequences=True,stateful=False, return_state=True,input_shape=(None, inp_feat))
        self.encoder_lstm2 = tf.keras.layers.LSTM(units=unit, return_state=True, stateful=False)
        
        self.decoder_lstm1 = tf.keras.layers.LSTM(units=unit, return_sequences=True)
        self.decoder_lstm2 = tf.keras.layers.LSTM(units=unit, return_sequences=True)
        
        net_dim = sum(self.FIELD_DIMS_NET.values())
        if self.conditional:
            # Dense layers for different outputs
            self.dense_layers = {}
            for name, dim in self.FIELD_DIMS_NET.items():
                acti = self.ACTIVATIONS.get(name, None)
                self.dense_layers[name] = tf.keras.layers.Dense(dim, activation=acti)
        else:
            self.dense_layer = tf.keras.layers.Dense(net_dim)
 

    def call(self, inp):
        inp_inp = inp[:, :-1]  # predict next from this
        inp_out = inp[:, 1:]
    
        # Encoder
        output1, state_h1, state_c1 = self.encoder_lstm1(inp_inp)
        output2, state_h2, state_c2 = self.encoder_lstm2(output1)

        # Decoder
        seq_len = inp_inp.shape[1]
        decoder_input = tf.keras.layers.RepeatVector(seq_len)(output2)
        L1 = self.decoder_lstm1(decoder_input, initial_state=[state_h1, state_c1])
        final_output = self.decoder_lstm2(L1, initial_state=[state_h2, state_c2])

        
        # Predictions for each field
        preds = {}
        if self.conditional:
            for net_name in self.ORDER:
                pred = self.dense_layers[net_name](final_output)
                preds[net_name] = pred

                st = self.FIELD_STARTS_IN[net_name]
                end = st + self.FIELD_DIMS_IN[net_name]
                to_add = inp_out[:, :, st:end]

                final_output = tf.concat([final_output, to_add], axis=-1)
        else:
            final_output = self.dense_layer(final_output)
            st = 0
            for name, dim in self.FIELD_DIMS_NET.items():
                acti = self.ACTIVATIONS.get(name, None)
                ed = st + dim
                if acti is None:
                   preds[name] = final_output[:,:,st:ed]
                elif acti == 'relu':
                   preds[name] = tf.keras.activations.relu(final_output[:, :, st:ed])
                st = ed
        return preds
    


class Encoder_Decoder_lstm_Inference(tf.keras.Model):
    def __init__(self, config, inp_feat, conditional = True, unit=128):
        super(Encoder_Decoder_lstm_Inference, self).__init__()

        self.ORDER = config["ORDER"]
        self.FIELD_STARTS_IN = config["FIELD_STARTS_IN"]
        self.FIELD_DIMS_IN = config["FIELD_DIMS_IN"]
        self.FIELD_STARTS_NET = config["FIELD_STARTS_NET"]
        self.FIELD_DIMS_NET = config["FIELD_DIMS_NET"]
        self.ACTIVATIONS = config["ACTIVATIONS"]

        self.unit = unit
        self.conditional = conditional

        # LSTM layers
        self.encoder_lstm1 = tf.keras.layers.LSTM(units=unit, return_sequences=True, stateful=True, return_state=True,input_shape=(None, inp_feat))
        self.encoder_lstm2 = tf.keras.layers.LSTM(units=unit, return_state=True, stateful=True)
        #self.decoder_repeat_vector = tf.keras.layers.RepeatVector(seq_len)
        self.decoder_lstm1 = tf.keras.layers.LSTM(units=unit, return_sequences=True)
        self.decoder_lstm2 = tf.keras.layers.LSTM(units=unit, return_sequences=True)
        
        net_dim = sum(self.FIELD_DIMS_NET.values())

        if self.conditional:
            # Dense layers for different outputs
            self.dense_layers = {}
            for name, dim in self.FIELD_DIMS_NET.items():
                acti = self.ACTIVATIONS.get(name, None)
                self.dense_layers[name] = tf.keras.layers.Dense(dim, activation=acti)
        else:
            self.dense_layer = tf.keras.layers.Dense(net_dim)

    def call(self, inp, return_decoder_lstm2_output=False):
       
     
        inp_inp = inp
        inp_out = inp
        # Encoder
        output1, state_h1, state_c1 = self.encoder_lstm1(inp_inp)
        output2, state_h2, state_c2 = self.encoder_lstm2(output1)

        # Decoder
        seq_len = inp_inp.shape[1]
        decoder_input = tf.keras.layers.RepeatVector(seq_len)(output2)
        L1 = self.decoder_lstm1(decoder_input, initial_state=[state_h1, state_c1])
        final_output = self.decoder_lstm2(L1, initial_state=[state_h2, state_c2])
        
        if return_decoder_lstm2_output:
            return final_output

        
        # Predictions for each field
        preds = {}
        if self.conditional:
            for net_name in self.ORDER:
                pred = self.dense_layers[net_name](final_output)
                preds[net_name] = pred

                st = self.FIELD_STARTS_IN[net_name]
                end = st + self.FIELD_DIMS_IN[net_name]
                to_add = inp_out[:, :, st:end]

                final_output = tf.concat([final_output, to_add], axis=-1)
        else:
            final_output = self.dense_layer(final_output)
            st = 0
            for name, dim in self.FIELD_DIMS_NET.items():
                acti = self.ACTIVATIONS.get(name, None)
                ed = st + dim
                if acti is None:
                   preds[name] = final_output[:,:,st:ed]
                else:
                   preds[name] = tf.keras.activations.relu(final_output[:, :, st:ed])
                st = ed
        return preds
    

