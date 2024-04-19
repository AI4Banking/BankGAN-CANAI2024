import numpy as np
import datetime
import calendar
import time
import tensorflow as tf
import pandas as pd
from lib.field_info import FieldInfo, FIELD_INFO_TCODE, FieldInfo_type2
from lib.modules import create_masks, Encoder_Decoder_lstm_Inference
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import csv
import json
import random
import os

# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

confighyper = load_config('config_hyper.json')
STRATEGY = confighyper['strategy']
loss_data_filename = confighyper['loss_data_filename']

fieldInfo = FieldInfo(STRATEGY)
#fieldInfo = FIELD_INFO_TCODE()
#fieldInfo = FieldInfo_type2()

loss_scce_logit = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_mse = tf.keras.losses.MeanSquaredError(reduction='none')

LOSS_WEIGHTS = {
 'td_sc':1.,
 'month': 0.015,
 'day': 0.025,
 'dtme': 0.025,
 'dow': 0.01,
 'tcode_num': 1.,
 'k_symbol_num':1.,
 'operation_num':1.0,
 'type_num':1.0,
 'log_amount_sc': 2.}

# INP_ENCODING = fieldInfo.INP_ENCODINGS
# NET_ENCODING = fieldInfo.NET_ENCODINGS
FIELD_STARTS_TAR = fieldInfo.FIELD_STARTS_TAR
FIELD_DIMS_TAR = fieldInfo.FIELD_DIMS_TAR
LOSS_TYPES = fieldInfo.LOSS_TYPES
FIELD_STARTS_IN = fieldInfo.FIELD_STARTS_IN
FIELD_DIMS_IN = fieldInfo.FIELD_DIMS_IN
FIELD_DIMS_NET = fieldInfo.FIELD_DIMS_NET

def bulk_encode_time_value(val, max_val):
    """ encoding date features in the clockwise dimension """
    x = np.sin(2 * np.pi / max_val * val)
    y = np.cos(2 * np.pi / max_val * val)
    return np.stack([x, y], axis=1)

def clock_to_probs(pt, pts):
    EPS_CLOCKP = 0.01
    pt_expanded = tf.expand_dims(pt, 2)                     # Shape: (n_seq, seq_len, 1, 2)
    pts_expanded = tf.reshape(pts, [1, 1, pts.shape[0], 2]) # Shape: (1, 1, net_dim, 2)

    ds = tf.constant(pts_expanded) - pt_expanded            # Shape: (n_seq, seq_len, net_dim, 2)
    sq_ds = np.sum(tf.square(ds+EPS_CLOCKP), axis= -1)       # Shape: (n_seq, seq_len, net_dim)
    raw_ps = 1/ sq_ds   
    return raw_ps / np.sum(raw_ps)

date_info = {'month':12, 'day':31, 'dtme':31, 'dow':7}
CLOCKS = {}
for k, val in date_info.items():
    CLOCKS[k] = tf.constant(bulk_encode_time_value(np.arange(val), val), dtype=tf.float32)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return  -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)


def loss_function(real, preds):
    loss_parts = []
    loss_parts_weighted = []
    mask = tf.math.logical_not(tf.math.equal(tf.reduce_sum(real, axis=2), 0))
    for k, k_pred in preds.items():
        st = FIELD_STARTS_TAR[k]
        end = st + FIELD_DIMS_TAR[k]
        loss_type = LOSS_TYPES[k]
        if loss_type == "scce":
           loss_ = loss_scce_logit(real[:, :, st:end], k_pred)
        elif loss_type == "pdf":
           loss_ = -log_normal_pdf(real[:, :, st:end], k_pred[:,:,0:1], k_pred[:,:,1:2])[:,:,0]
        elif loss_type == 'mse':
           loss_ = loss_mse(real[:, :, st:end], k_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_sum(loss_)/tf.reduce_sum(mask) 

        loss_parts.append(loss_)
        loss_parts_weighted.append(loss_ * LOSS_WEIGHTS[k])
    return tf.reduce_sum(loss_parts_weighted)

# Function to substitute month 0 with 12 and adjust days based on the month
def adjust_month_and_day(month, day):
    # Substitute month 0 with 12
    month = 12 if month == 0 else month

    # Adjust the day based on the month
    # Months with 31 days
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return month, 31 if day == 0 else day
    # February (not considering leap years in this example)
    elif month == 2:
        return month, 28 if day == 0 else day
    # Months with 30 days
    else:
        return month, 30 if day == 0 else day


def log_normal_pdf_gen(sample, mean, logvar, raxis=1):
    log2pi = tf.cast(tf.math.log(2. * np.pi), tf.float64)
    return  -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

def encode_rbf(array, rbf, net_name):
    """
    Transform a NumPy array using a fitted RepeatingBasisFunction and convert to a NumPy array of shape (n, num of rbf functions = 2).

    Parameters:
    array (np.array): Input NumPy array of shape (n,).
    rbf (RepeatingBasisFunction): Fitted RepeatingBasisFunction object.
    net_name: name of the the date column('dow', 'month', 'day', 'dtme')

    Returns:
    np.array: Transformed NumPy array of shape (n, num of rbf functions = 2).
    """
    # Convert the NumPy array to DataFrame
    df = pd.DataFrame(array, columns=[net_name])

    # Transform using the fitted RBF
    transformed_array = rbf.transform(df)

    return transformed_array

def encode_onehot(array, net_dim):
    """
    Converts an array of numbers to a one-hot encoded numpy array.
    
    Args:
    array (array-like): An array of numbers.
    net_dim (int): The dimension for the one-hot encoding.

    Returns:
    numpy.ndarray: A numpy array of shape (n, net_dim) with one-hot encoded values.
    """
    # Convert the input array to a TensorFlow tensor
    tensor = tf.constant(array)

    # Apply one-hot encoding
    one_hot_tensor = tf.one_hot(tensor, depth=net_dim)

    # Convert the one-hot encoded tensor to a numpy array and return
    return one_hot_tensor.numpy()

def raw_dates_to_reencoded(raw_preds, start_inds, AD, TD_SCALE,RBF_dic, max_days = 100, greedy_decode=False):

    """ 
    raw_preds: raw predictions (info about predicted day, month, dow, and days passed)
    start_inds: the index of the previous transaction's date in AD or ALL_DATES
    max_days:  the next transaction date is sampled among the next 100('max_days') days, starting from start_inds
    
    
        Computes a number of days passed for each based on inputs (either greedily or with sampling)
         returns the new_dates (old_dates + days passed) and their indicies   """
    

    # raw_preds[k][:, -1]-- get the last element in each sequence  
    for k in ["month", "day", "dow", "dtme"]:
        if raw_preds[k].shape[-1] == 2:
           raw_preds[k] = clock_to_probs(raw_preds[k], CLOCKS[k]) 

    all_ps = [tf.nn.softmax(raw_preds[k][:,-1]).numpy() for k in ["month", "day", "dow", "dtme"]]  #length of list: 4
    timesteps = np.zeros(len(start_inds)).astype(int)
    for i, (month_ps, day_ps, dow_ps, dtme_ps, td_pred, si) in enumerate(zip(*all_ps, raw_preds["td_sc"][:,-1].numpy(), start_inds)):
            
        ps = month_ps[AD[si:si+max_days,0]]*day_ps[AD[si:si+max_days,1]]*dow_ps[AD[si:si+max_days,2]] *dtme_ps[AD[si:si+max_days,-1]] * \
                    np.exp(log_normal_pdf_gen(AD[si:si+max_days,3]-si, mean = td_pred[0]*TD_SCALE, logvar=td_pred[1]*TD_SCALE))  #shape(max_days,)

        #print('AD[si:si+max_days].shape: ', AD[si:si+max_days].shape)
        if greedy_decode:
            timesteps[i] = np.argmax(ps)
        else:
            timesteps[i] = np.random.choice(max_days, p=ps/sum(ps))
            
    inds = start_inds + timesteps
        
        
    return_ = {}
    return_["td_sc"] = tf.expand_dims(timesteps.astype(np.float32)/ TD_SCALE, axis=1)

    if STRATEGY == 'banksformer':
        return_["month"] = bulk_encode_time_value(AD[inds, 0], 12)
        return_["day"] = bulk_encode_time_value(AD[inds, 1], 31)
        return_["dow"] = bulk_encode_time_value(AD[inds, 2], 7)
        return_["dtme"] = bulk_encode_time_value(AD[inds, -1], 31)

    elif STRATEGY == 'daterbf':
        return_["month"] = encode_rbf(AD[inds, 0], RBF_dic['month'], 'month')
        return_["day"] = encode_rbf(AD[inds, 1], RBF_dic['day'], 'day')
        return_["dow"] = encode_rbf(AD[inds, 2], RBF_dic['dow'], 'dow')
        return_["dtme"] = encode_rbf(AD[inds, 0], RBF_dic['dtme'], 'dtme')

    elif STRATEGY == 'dateonehot':
        return_['month'] = encode_onehot(AD[inds, 0], 12)
        return_["day"] =  encode_onehot(AD[inds, 1], 31)
        return_["dow"] =  encode_onehot(AD[inds, 2], 7)
        return_["dtme"] =  encode_onehot(AD[inds, -1], 31)
        #print(return_)

    raw_date = {}
    raw_date['month'] = AD[inds, 0]
    raw_date['day'] = AD[inds, 1]
    raw_date['year'] = AD[inds, 4]

    return return_, inds, raw_date


    #return return_, return_oh, inds, raw_date




def reencode_net_prediction(net_name, predictions, RBF_dic):
    """net_name is in ['tcode_num', 'dow', 'month', 'day', 'dtme', 'td_sc', 'log_amount_sc']
       predictions for ['tcode_num', 'dow', 'month', 'day', 'dtme'] are probabilities of dimensions [16, 7, 12, 31, 31]

       function:  transform predictions to the correct form to be used for conditional generating: 
                  convert 'tcode' to one-hot encoded vector, 
                  convert date features to clock dimension
                  extract the predicted mean of 'td' and 'amount' as predicted values 
                
    """
    
    print("reencode_net_prediction:", net_name, predictions.shape)
    batch_size = predictions.shape[0]
    date_info = {'month':12, 'day':31, 'dtme':31, 'dow':7}

    if "_num" in net_name:
        dim = FIELD_DIMS_NET[net_name]
        choices = np.arange(dim)
        ps = tf.nn.softmax(predictions, axis=2).numpy().reshape(-1, dim)    #predictions: (n_seq_to_generate, seq_len, dim=16)
        choosen =  np.reshape([np.random.choice(choices, p=p) for p in ps], newshape=(batch_size, -1))

        return tf.one_hot(choosen, depth=dim)      #(n_seq_to_generate, seq_len, dim=16)
    
    elif net_name in date_info.keys() and RBF_dic is None:                  #date representation is Clock Encoding or one-hot encoding
        dim = FIELD_DIMS_NET[net_name]
        choices = np.arange(dim)
        ps = tf.nn.softmax(predictions, axis=2).numpy().reshape(-1, dim)
        choosen =  np.array([np.random.choice(choices, p=p) for p in ps])
        #print('choosen',choosen, " ", net_name)
        if STRATEGY == 'banksformer':
           #print('banksbanks') 
           x = bulk_encode_time_value(choosen, max_val=dim)
           #print('x', x.shape, type(x))
           return np.reshape(x, newshape=(batch_size, -1, 2))
        else:
            x = encode_onehot(choosen, dim)
            #print(x.shape)
            return np.reshape(x, newshape=(batch_size, -1, dim))
           
    
    elif net_name in date_info.keys() and RBF_dic is not None:               #date representation is RBF
        #print(net_name)
        dim = FIELD_DIMS_NET[net_name]
        choices = np.arange(dim)
        ps = tf.nn.softmax(predictions, axis=2).numpy().reshape(-1, dim)
        choosen =  np.array([np.random.choice(choices, p=p) for p in ps])
        #print('choosen',choosen, " ", net_name)
        x = encode_rbf(choosen, RBF_dic[net_name], net_name)
        #print('x', x.shape, type(x))
        return np.reshape(x, newshape=(batch_size, -1, 2))


    # elif net_name in ['td_sc', "log_amount_sc"]:
    #     return predictions[:, :, 0:1]
    elif net_name in ['td_sc', "log_amount_sc"]:
        mean, log_var = predictions[:, :, 0:1],  predictions[:, :, 1:2]
        # sd = np.sqrt(np.exp(log_var))
        log_sd = log_var/2.
        return mean +  log_sd * np.random.normal(size=(batch_size, 1, 1)) 

    # elif net_name in date_info.keys() and 'cl' in INP_ENCODING[net_name] and 'cl' in NET_ENCODING[net_name]:
    #     dim = date_info[net_name]
    #     assert predictions.shape[-1] == FIELD_DIMS_NET[net_name]
    #     choices = np.arange(dim)
    #     preds = clock_to_probs(predictions, CLOCKS[net_name])
    #     ps = tf.nn.softmax(preds, axis=2).numpy().reshape(-1, dim)    #predictions: (n_seq_to_generate, seq_len, dim=dim)
    #     choosen =  np.reshape([np.random.choice(choices, p=p) for p in ps], newshape=(batch_size, -1))
    #     x = bulk_encode_time_value(choosen, max_val=dim)
        
        # return np.reshape(x, newshape=(batch_size, -1, 2))
    # elif net_name in ['td_sc', "log_amount_sc"]:
    #     return predictions[:, :, 0:1]


def call_to_generate(lstm, inp, start_inds, AD, TD_SCALE,RBF_dic):
    """
    This function is called 'lenght_of_sequences' times
    lstm : trained stacked lstm used for generating synthetic data
    inp: in the first call, it is a vector of features, dimension: #(n_seqs_to_generate, 1, n_feat_inp)
         in the second call, dim is: #(n_seqs_to_generate, 2, n_feat_inp)
         and so on.....
         inp is forwarded pass through the trained model 
    start_inds: #array of shape (n_seqs_to_generate,) specifies the starting date indexes in ALL_DATES array(and also in AD array), 
                 for each sequence to be generated
    AD: an array of shape(5478,6), each element is an array contains the information of date(month, day, dow,idx, year, dtme), 
        spanning 15 years.
    output: 
   
    Returns: preds, raw_preds, inds, encoded_preds, raw_date_info
    raw_preds: for categorical features and date features is the raw values given to a softmax function
               for 'td_sc' and 'log_amount_sc' is the output of dense layer(units=2) with relu activation function
    preds:  'tcode': one-hot-encoded   'date features': clock dimension   'td' , 'amount': predicted mean
    inds  : index of predicted dates in AD
    encoded_preds: the last term in each sequence of preds
    the returned preds have multiple timesteps, but we only care about the last (it's the only new one)   """

 
    final_output = lstm(inp, return_decoder_lstm2_output=True)

    ### Predict each field  ###
    
    #raw_preds is the outputs of the last dense layer of deep model. 
    raw_preds = {}
    #preds is the reencoded raw_preds, 'tcode' converts to one-hot encoded, 'date-features' are converted to clock-wise
    #and for 'amount' and 'td' the predicted mean is extracted. it is used for conditional generating. 
    preds = {}
    #encoded_preds_d is similar to preds for 'tcode', 'td', and 'amount', but for date features , the predicted date is computed 
    #based on a formula 
    encoded_preds_d = {}
   
    if lstm.conditional:
        for net_name in lstm.ORDER:  

            raw_preds[net_name] = lstm.dense_layers[net_name](final_output)
            en_pred = reencode_net_prediction(net_name, raw_preds[net_name], RBF_dic) 
            preds[net_name] = en_pred
                
            encoded_preds_d[net_name] = en_pred[:,-1,:] 
            #encoded_preds.append(pred[:,-1,:])
            final_output = tf.concat([final_output, en_pred], axis=2)

        
    else:
        final_output = lstm.dense_layer(final_output)
        st = 0
        for net_name, dim in lstm.FIELD_DIMS_NET.items():
            acti = lstm.ACTIVATIONS.get(net_name, None)
            ed = st + dim
            if acti is None:
                raw_preds[net_name] = final_output[:,:,st:ed]
            elif acti == 'relu':
                raw_preds[net_name] = tf.keras.activations.relu(final_output[:, :, st:ed])
            st = ed

            en_pred = reencode_net_prediction(net_name, raw_preds[net_name], RBF_dic) 
            preds[net_name] = en_pred

            encoded_preds_d[net_name] = en_pred[:,-1,:] 

    date_info, inds, raw_date_info = raw_dates_to_reencoded(raw_preds, start_inds, AD, TD_SCALE,RBF_dic)
    
    encoded_preds_d.update(date_info)
    for k in lstm.ORDER:
        assert encoded_preds_d[k].shape[-1] == FIELD_DIMS_IN[k]
    l = [encoded_preds_d[k] for k in lstm.ORDER]
    encoded_preds =  tf.expand_dims(tf.concat(l, axis=1), axis=1)   #tensor of shape (n_seqs_to_generate, 1, 26(input features))

    assert encoded_preds.shape[-1] == sum(FIELD_DIMS_IN.values())

    return preds, raw_preds, inds, encoded_preds, raw_date_info


def call_to_generate_type2(lstm, inp):

    final_output = lstm(inp, return_decoder_lstm2_output=True)
    raw_preds = {}
    #preds is the reencoded raw_preds, 'tcode' converts to one-hot encoded, 'date-features' are converted to clock-wise
    #and for 'amount' and 'td' the predicted mean is extracted. it is used for conditional generating. 
    preds = {}
    #encoded_preds_d is similar to preds for 'tcode', 'td', and 'amount', but for date features , the predicted date is computed 
    #based on a formula 
    encoded_preds_d = {}
    #encoded_preds = []
    
    if lstm.conditional:
        for net_name in lstm.ORDER:
            raw_preds[net_name] = lstm.dense_layers[net_name](final_output)
            en_pred = reencode_net_prediction(net_name, raw_preds[net_name], None) 
            preds[net_name] = en_pred
                    
            encoded_preds_d[net_name] = en_pred[:,-1,:] 
            #encoded_preds.append(pred[:,-1,:])
            final_output = tf.concat([final_output, en_pred], axis=2)
    else:
        final_output = lstm.dense_layer(final_output)
        st = 0
        for net_name, dim in lstm.FIELD_DIMS_NET.items():
            acti = lstm.ACTIVATIONS.get(net_name, None)
            ed = st + dim
            if acti is None:
                raw_preds[net_name] = final_output[:,:,st:ed]
            elif acti == 'relu':
                raw_preds[net_name] = tf.keras.activations.relu(final_output[:, :, st:ed])
            st = ed

            en_pred = reencode_net_prediction(net_name, raw_preds[net_name], None) 
            preds[net_name] = en_pred

            encoded_preds_d[net_name] = en_pred[:,-1,:] 

    l = [encoded_preds_d[k] for k in lstm.ORDER]
    encoded_preds =  tf.expand_dims(tf.concat(l, axis=1), axis=1)
    return preds, encoded_preds

    


def save_csv(results, loss_data_filename):
    with open(loss_data_filename, 'w', newline='') as file:
       writer = csv.writer(file)

       # Write the header
       writer.writerow(results.keys())

       # Write the data
       for row in zip(*results.values()):
           writer.writerow(row)


class Train(object):
    def __init__(self, lstm):
        self.lstm = lstm
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.validation_loss = tf.keras.metrics.Mean(name='val_loss')
        self.results = dict([(x, []) for x in ["loss", "val_loss"]])

    def train(self, train_batches, val_batches, epochs, early_stop):
        #optimizer = tf.keras.optimizers.Adam() 
        l2_norm_clip = 1.0
        noise_multiplier = 2.2
        
        optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier)
    
        for epoch in range(epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.validation_loss.reset_states()
            for (batch_no, (inp, tar)) in enumerate(train_batches):
                with tf.GradientTape() as tape:
                    predictions = self.lstm(inp)
                    loss = loss_function(tar, predictions)
                gradients = tape.gradient(loss, self.lstm.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_variables))
            
                self.train_loss(loss)
                if batch_no % 50 == 0:
                    print(f'Epoch {epoch+1} Batch{batch_no} Loss{self.train_loss.result(): .4f}')
            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f}')
            for (_, (x_cv, targ_cv)) in enumerate(val_batches):
                predictions_val = self.lstm(x_cv)
                loss_v = loss_function(targ_cv, predictions_val)
                self.validation_loss(loss_v)
            print(f"** on validation data loss is {self.validation_loss.result():.4f}")
            self.results["loss"].append(self.train_loss.result().numpy())
            self.results["val_loss"].append(self.validation_loss.result().numpy())
            
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
            
            if min(self.results["val_loss"] ) < min(self.results["val_loss"][-early_stop:] ):
                
                print(f"Stopping early, last {early_stop} val losses are: {self.results['val_loss'][-early_stop:]} \
                    \nBest was {min(self.results['val_loss'] ):.3f}\n\n")
                save_csv(self.results, loss_data_filename)
                self.lstm.save_weights('lstm_model_weights.h5')
                break
            save_csv(self.results, loss_data_filename)
        self.lstm.save_weights('lstm_model_weights.h5')

    def generate_synthetic_data(self, max_length, n_seqs_to_generate, df, attributes, n_feat_inp, RBF_dic = None):
        """ 
        max_length : length of the generated sequences
        n_seqs_to_generate: number of unique customers in the generated data
        df: original preprocessed dataframe
        attributes: an array of dimension(number_of_seqs_in_training_data,) of scaled attributes(age)
        """
        MAX_YEARS_SPAN = 20
        get_dtme = lambda d: calendar.monthrange(d.year, d.month)[1] - d.day

        START_DATE = df["datetime"].min()
        ATTR_SCALE = df["age"].std()
        LOG_AMOUNT_SCALE = df["log_amount"].std()
        TD_SCALE = df["td"].std()

        NUM_TO_TCODE = dict([(i, tc) for i, tc in enumerate(df['tcode'].unique())])
        # NUM_TO_K_SYMBOL = dict([(i, tc) for i, tc in enumerate(df['k_symbol'].unique())])
        # NUM_TO_OPERATION = dict([(i, tc) for i, tc in enumerate(df['operation'].unique())])
        # NUM_TO_TYPE = dict([(i, tc) for i, tc in enumerate(df['type'].unique())])

        END_DATE = START_DATE.replace(year = START_DATE.year+ MAX_YEARS_SPAN)

        ALL_DATES = [START_DATE + datetime.timedelta(i) for i in range((END_DATE - START_DATE).days)]
        AD = np.array([(d.month % 12, d.day % 31, d.weekday() % 7, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
        start_date_opts = df.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
        start_dates = np.random.choice(start_date_opts, size=n_seqs_to_generate) # sample start dates from real data

        seq_ages = np.random.choice(attributes, size=n_seqs_to_generate) # sample ages from real data

        #generate sequences
        start_inds = np.array([(d - START_DATE.date()).days for d in start_dates])    #array of shape (n_seqs_to_generate,)
        #print(start_inds)
        inp = np.repeat(np.array(seq_ages)[:, None, None], repeats=n_feat_inp, axis=2) / ATTR_SCALE   #(n_seqs_to_generate, 1, n_feat_inp) 
        raw_date_info_list = []
        
        conditional = self.lstm.conditional
        inference_model = Encoder_Decoder_lstm_Inference(self.lstm.config, self.lstm.inp_feat, conditional)
        dummy_input = tf.random.normal([n_seqs_to_generate, max_length, n_feat_inp])
        # Build the model by running dummy data through it
        inference_model(dummy_input)
        inference_model.load_weights('lstm_model_weights.h5')
        
        for i in range(max_length):     
            predictions, raw_ps, date_inds, enc_preds, raw_date  = call_to_generate(inference_model, inp, start_inds, AD, TD_SCALE, RBF_dic)
            #print(date_inds)
            enc_preds = tf.reshape(tf.constant(enc_preds), shape=(-1,1, n_feat_inp))      #(n_seqs_to_generate, 1, n_feat_inp=26)
            assert inp.shape[-1] == enc_preds.shape[-1]
            inp = tf.concat([inp, enc_preds], axis=1)   
            raw_date_info_list.append(raw_date)  
            start_inds = date_inds

        # Transform the data generated by BF back to the original data space
        seqs = inp
        ages = seqs[:, 0, :] * ATTR_SCALE
        seqs = seqs[:, 1:, :]
        assert np.sum(np.diff(ages)) == 0, f"Bad formating, expected all entries same in each row, got {ages}"
     
        amts = seqs[:, :, FIELD_STARTS_IN["log_amount_sc"]].numpy() * LOG_AMOUNT_SCALE
        amts = 10 ** amts
        amts = np.round(amts - 1.0, 2)
        days_passed = np.round(seqs[:, :, FIELD_STARTS_IN["td_sc"]] * TD_SCALE ).astype(int)

        t_code = np.argmax(seqs[:, :, FIELD_STARTS_IN["tcode_num"]: FIELD_STARTS_IN["tcode_num"] + FIELD_DIMS_IN["tcode_num"]], axis=-1)
        # k_symbol = np.argmax(seqs[:, :, FIELD_STARTS_IN["k_symbol_num"]: FIELD_STARTS_IN["k_symbol_num"] + FIELD_DIMS_IN["k_symbol_num"]], axis=-1)
        # operation = np.argmax(seqs[:, :, FIELD_STARTS_IN["operation_num"]: FIELD_STARTS_IN["operation_num"] + FIELD_DIMS_IN["operation_num"]], axis=-1)
        # type_ = np.argmax(seqs[:, :, FIELD_STARTS_IN["type_num"]: FIELD_STARTS_IN["type_num"] + FIELD_DIMS_IN["type_num"]], axis=-1)


        # Flatten arrays and translate transaction codes
        flattened_amts = amts.flatten()

        flattened_tcodes = t_code.flatten()
        # flattened_ksymbol = k_symbol.flatten()
        # flattened_operation = operation.flatten()
        # flattened_type = type_.flatten()

        translated_tcodes = [NUM_TO_TCODE[code] for code in flattened_tcodes]
        # translated_ksymbol = [NUM_TO_K_SYMBOL[code] for code in flattened_ksymbol]
        # translated_operation = [NUM_TO_OPERATION[code] for code in flattened_operation]
        # translated_type = [NUM_TO_TYPE[code] for code in flattened_type]
        

        # Create DataFrame for amounts and transaction codes
        df_synth = pd.DataFrame({
            'amount': flattened_amts,
            'transaction_code': translated_tcodes
        })

        # Create DataFrame for amounts and transaction codes
        # df_synth = pd.DataFrame({
        #     'amount': flattened_amts,
        #     'k_symbol': translated_ksymbol,
        #     'operation': translated_operation,
        #     'type' : translated_type
        # })

        # Handling account IDs
        num_customers = amts.shape[0]
        num_transactions = amts.shape[1]
        account_ids = np.repeat(range(num_customers), num_transactions)
        df_synth['account_id'] = account_ids

        # Handling date information
        months = []
        days = []
        years = []

        for customer in range(num_customers):
            print(customer)
            for transaction in range(num_transactions):
                months.append(raw_date_info_list[transaction]['month'][customer])
                days.append(raw_date_info_list[transaction]['day'][customer])
                years.append(raw_date_info_list[transaction]['year'][customer])

        # Converting lists to numpy arrays
        months = np.array(months)
        days = np.array(days)
        years = np.array(years)

        # Applying the adjustments to months and days
        adjusted_months, adjusted_days = zip(*[adjust_month_and_day(m, d) for m, d in zip(months, days)])

        # Converting to numpy arrays
        adjusted_months = np.array(adjusted_months)
        adjusted_days = np.array(adjusted_days)


        df_synth['year'] = years
        df_synth['month'] = adjusted_months
        df_synth['day'] = adjusted_days

        df_synth['date'] = pd.to_datetime(df_synth[['year', 'month', 'day']])

        # Handling days passed
        flattened_days_passed = days_passed.flatten()
        flattened_days_passed[::num_transactions] = 0  # Setting the first transaction's days_passed to 0
        df_synth['days_passed'] = flattened_days_passed

        return df_synth

    def generate_synthetic_tcode(self, max_length, n_seqs_to_generate, df, attributes, n_feat_inp):
        #generate 'tcode' sequences
        NUM_TO_TCODE = dict([(i, tc) for i, tc in enumerate(df['tcode'].unique())])
        ATTR_SCALE = df["age"].std()
        seq_ages = np.random.choice(attributes, size=n_seqs_to_generate) # sample ages from real data
        inp = np.repeat(np.array(seq_ages)[:, None, None], repeats=n_feat_inp, axis=2) / ATTR_SCALE   #(n_seqs_to_generate, 1, n_feat_inp) 
        
        conditional = self.lstm.conditional
        inference_model = Encoder_Decoder_lstm_Inference(self.lstm.config, self.lstm.inp_feat, conditional)
        dummy_input = tf.random.normal([n_seqs_to_generate, max_length, n_feat_inp])
        # Build the model by running dummy data through it
        inference_model(dummy_input)
        inference_model.load_weights('lstm_model_weights.h5')

        for i in range(max_length):     
            predictions, enc_preds = call_to_generate_type2(inference_model, inp)
            #print(date_inds)
            enc_preds = tf.reshape(tf.constant(enc_preds), shape=(-1,1, n_feat_inp))      #(n_seqs_to_generate, 1, n_feat_inp)
            inp = tf.concat([inp, enc_preds], axis=1)   
            print(inp.shape)
        seqs = inp
        ages = seqs[:, 0, :] * ATTR_SCALE
        seqs = seqs[:, 1:, :]
        assert np.sum(np.diff(ages)) == 0, f"Bad formating, expected all entries same in each row, got {ages}"
        tcodeseq = seqs[:, :, FIELD_STARTS_IN["tcode_num"]: FIELD_STARTS_IN["tcode_num"] + FIELD_DIMS_IN["tcode_num"]]
        t_code = np.argmax(seqs[:, :, FIELD_STARTS_IN["tcode_num"]: FIELD_STARTS_IN["tcode_num"] + FIELD_DIMS_IN["tcode_num"]], axis=-1)
        flattened_tcodes = t_code.flatten()
        translated_tcodes = [NUM_TO_TCODE[code] for code in flattened_tcodes]
        df_synth = pd.DataFrame({ 'tcode': translated_tcodes })
        # Handling account IDs
        num_customers = tcodeseq.shape[0]
        num_transactions = tcodeseq.shape[1]
        account_ids = np.repeat(range(num_customers), num_transactions)
        df_synth['account_id'] = account_ids

        return df_synth



    def generate_synthetic_data_type2(self, max_length, n_seqs_to_generate, df, attributes, n_feat_inp):
        "for generating data when the inputs are [tcode, amount, td]"
        LOG_AMOUNT_SCALE = df["log_amount"].std()
        TD_SCALE = df["td"].std()
        NUM_TO_TCODE = dict([(i, tc) for i, tc in enumerate(df['tcode'].unique())])
        ATTR_SCALE = df["age"].std()
        seq_ages = np.random.choice(attributes, size=n_seqs_to_generate) # sample ages from real data
        inp = np.repeat(np.array(seq_ages)[:, None, None], repeats=n_feat_inp, axis=2) / ATTR_SCALE   #(n_seqs_to_generate, 1, n_feat_inp) 
       
        start_date_opts = df.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
        sampled_start_dates = np.random.choice(start_date_opts, size=n_seqs_to_generate) # sample start dates from real data

        conditional = self.lstm.conditional
        inference_model = Encoder_Decoder_lstm_Inference(self.lstm.config, self.lstm.inp_feat, conditional)
        dummy_input = tf.random.normal([n_seqs_to_generate, max_length, n_feat_inp])
        # Build the model by running dummy data through it
        inference_model(dummy_input)
        inference_model.load_weights('lstm_model_weights.h5')

        for i in range(max_length):     
            raw_ps, enc_preds = call_to_generate_type2(inference_model, inp)
            #print(date_inds)
            enc_preds = tf.reshape(tf.constant(enc_preds), shape=(-1,1, n_feat_inp))      #(n_seqs_to_generate, 1, n_feat_inp)
            inp = tf.concat([inp, enc_preds], axis=1)   
            print(inp.shape)

            FIELD_STARTS_IN = fieldInfo.FIELD_STARTS_IN
            FIELD_DIMS_IN = fieldInfo.FIELD_DIMS_IN

        seqs = inp
        seqs = inp
        ages = seqs[:, 0, :] * ATTR_SCALE
        seqs = seqs[:, 1:, :]
        assert np.sum(np.diff(ages)) == 0, f"Bad formating, expected all entries same in each row, got {ages}"

        amts = seqs[:, :, FIELD_STARTS_IN["log_amount_sc"]].numpy() * LOG_AMOUNT_SCALE
        amts = 10 ** amts
        amts = np.round(amts - 1.0, 2)
        days_passed = np.round(seqs[:, :, FIELD_STARTS_IN["td_sc"]] * TD_SCALE ).astype(int)

        t_code = np.argmax(seqs[:, :, FIELD_STARTS_IN["tcode_num"]: FIELD_STARTS_IN["tcode_num"] + FIELD_DIMS_IN["tcode_num"]], axis=-1)
        flattened_amts = amts.flatten()

        flattened_tcodes = t_code.flatten()

        flattened_td = days_passed.flatten()


        translated_tcodes = [NUM_TO_TCODE[code] for code in flattened_tcodes]

        df_synth = pd.DataFrame({
            'amount': flattened_amts,
            'tcode': translated_tcodes,
            'td': flattened_td

        })
        # Handling account IDs
        num_customers = amts.shape[0]
        num_transactions = amts.shape[1]
        account_ids = np.repeat(range(num_customers), num_transactions)
        df_synth['account_id'] = account_ids

        # Identify the first transaction for each account
        first_transactions = df_synth.groupby('account_id').head(1).index
        # Set 'td' to 0 only for the first transactions
        df_synth.loc[first_transactions, 'td'] = 0

        df_synth['cumulative_td'] = df_synth.groupby('account_id')['td'].cumsum()

        for i, account_id in enumerate(df_synth['account_id'].unique()):
            start_date_str = sampled_start_dates[i].strftime('%Y-%m-%d')
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')

            # Filter the rows for the current account_id
            account_rows = df_synth[df_synth['account_id'] == account_id]

            # Calculate the date for each transaction
            for index, row in account_rows.iterrows():
                if index == 0:
                    df_synth.at[index, 'td'] = 0
                    df_synth.at[index, 'datetime'] = start_date
                else:
                    transaction_date = start_date + datetime.timedelta(days=row['cumulative_td'])
                    df_synth.at[index, 'datetime'] = transaction_date
        return df_synth




