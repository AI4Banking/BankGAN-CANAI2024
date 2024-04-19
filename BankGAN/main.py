from myctgan import CTGAN
from data_transformer import DataTransformer
from data_sampler import DataSampler
from train import Train
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import pickle
import datetime
import calendar
from scipy.stats import norm
import os
import random

from prepare_data import preprocess_data_czech
# Set seeds
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

def process_group(group):
    n = 80
    chunks = [group.iloc[i:i + n] for i in range(0, len(group), n)]
    processed_chunks = [chunk for chunk in chunks if len(chunk) == n]
    return processed_chunks

def _apply_activate2(transformer, data):
    """Apply proper activation function to the output of the generator."""
    data_t = []
    st = 0
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn == 'tanh':
                ed = st + span_info.dim
                data_t.append(tf.math.tanh(data[:, st:ed]))
                st = ed
            elif span_info.activation_fn == 'softmax':
                ed = st + span_info.dim
                transformed = tf.nn.softmax(data[:, st:ed])
                data_t.append(transformed)
                st = ed
            else:
                raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

    return tf.concat(data_t, axis=1)

# def main():
#     with tf.device('/gpu:0'):
#         n = 5000
#         embedding_dim = 100
#         df = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
#         print('preprocessing data....')
#         raw_data, LOG_AMOUNT_SCALE, TD_SCALE = preprocess_data_czech(df)
#         raw = raw_data.copy()

#         final_raw = raw[['log_amount_sc', 'tcode', 'td']]

#         # Process each group
#         processed_groups = raw.groupby('account_id').apply(process_group)

#         # Flatten the list of DataFrames and assign new account IDs
#         new_account_id = 0
#         processed_dfs = []
#         for group in processed_groups:
#             for chunk in group:
#                 chunk['new_account_id'] = new_account_id
#                 processed_dfs.append(chunk)
#                 new_account_id += 1

#         # Concatenate all processed DataFrames
#         final_df = pd.concat(processed_dfs, ignore_index=True)
#         #grouped = final_df.groupby('new_account_id')
#         synth_bf = pd.read_csv('../lstmModel/synth_lstm_tcode.csv')
#         synth_bf = synth_bf.rename(columns={'transaction_code': 'tcode'})
#         grouped = synth_bf.groupby('account_id')
        
#         generator = tf.saved_model.load("gen_ctgan_type2")
#         # Reading the object back from the file
#         with open('transformerobject.pkl', 'rb') as file:
#             transformer = pickle.load(file)

#         with open('samplerobject.pkl', 'rb') as file:
#             sampler = pickle.load(file)

#         #one hot encoder object
#         for column_info in transformer._column_transform_info_list:
#             if column_info.column_name == 'tcode':
#                ohe = column_info.transform
#         #construct list of starting indexes
#         START_DATE = raw_data['datetime'].min()
#         # start_dates = raw_data.groupby('account_id')['datetime'].min()
#         # sampled_start_dates = start_dates.sample(n)
#         start_date_opts = raw_data.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
#         sampled_start_dates = np.random.choice(start_date_opts, size=n) # sample start dates from real data
#         # Convert sampled_start_dates to a pandas Series of Timestamps
#         sampled_start_dates_series = pd.Series(pd.to_datetime(sampled_start_dates))

#         diff_days = (sampled_start_dates_series - START_DATE).dt.days
#         date_inds = diff_days.tolist()

#         data=[]
#         account_ids = []  # Initialize an empty list for account_ids
#         for seq_i, (name, group) in enumerate(grouped):
#             # name is the unique account_id
#             # group is a DataFrame with all rows for that account_id
#             print(seq_i)
#             if seq_i > n-1:
#                 break
#             account_id = name
#             m = len(group)
#             tcode = group['tcode']
#             si = date_inds[seq_i] 
#             # Add the account_id repeated m times to the account_ids list
#             account_ids.extend([account_id] * m)
#             for i in range(m):

#                 #def sample_external_condvec(self, tcode=tcode.iloc[i], oheobject=ohe)
                
#                 condvec = sampler.sample_external_condvec(tcode.iloc[i], ohe)
#                 length_of_seq = 1
#                 mean = tf.zeros(shape=(1, embedding_dim), dtype=tf.float32)
#                 std = mean + 1
#                 fakez = tf.random.normal(shape=(length_of_seq, embedding_dim),mean=mean, stddev=std)
#                 c1 = condvec
#                 c1 = tf.convert_to_tensor(np.array(c1))
#                 c1 = tf.cast(c1, dtype=tf.float32)
#                 fakez = tf.concat([fakez, c1], axis=1)
#                 fake = generator(fakez)
#                 fakeact = _apply_activate2(transformer, fake)
#                 data.append(fakeact.numpy())
#         data = np.concatenate(data, axis=0)
#         synth = transformer.inverse_transform(data)
#         synth['account_id'] = account_ids  # Add the account_id column
#         synth['td'] = synth['td'].apply(lambda x: 0 if x < 0 else round(x))
        

#          # Identify the first transaction for each account
#         first_transactions = synth.groupby('account_id').head(1).index
#         # Set 'td' to 0 only for the first transactions
#         synth.loc[first_transactions, 'td'] = 0


#         # Step 2: Calculate cumulative sum of 'td' for each account_id
#         synth['cumulative_td'] = synth.groupby('account_id')['td'].cumsum()

#         for i, account_id in enumerate(synth['account_id'].unique()):
#             start_date_str = sampled_start_dates[i].strftime('%Y-%m-%d')
#             start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')

#             # Filter the rows for the current account_id
#             account_rows = synth[synth['account_id'] == account_id]

#             # Calculate the date for each transaction
#             for index, row in account_rows.iterrows():
#                 if index == 0:
#                   synth.at[index, 'datetime'] = start_date
#                 else:
#                   transaction_date = start_date + datetime.timedelta(days=row['cumulative_td'])
#                   synth.at[index, 'datetime'] = transaction_date
        
#         experiment_name = 'synth_ctgan_type2_lstm'
#         filename = '../DATA/' + experiment_name +'.csv'
#         synth.to_csv(filename, index=False)


# def main():
#     with tf.device('/gpu:0'):
#         n = 5000
#         embedding_dim = 100
#         greedy_decode=False
#         df = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
#         print('preprocessing data....')
#         raw_data, LOG_AMOUNT_SCALE, TD_SCALE = preprocess_data_czech(df)
#         raw = raw_data.copy()

#         final_raw = raw[['log_amount_sc', 'tcode', 'td', 'day', 'dow', 'dtme', 'month']]
#         #final_raw = raw[['log_amount_sc', 'tcode', 'td']]

#         # Process each group
#         processed_groups = raw.groupby('account_id').apply(process_group)

#         # Flatten the list of DataFrames and assign new account IDs
#         new_account_id = 0
#         processed_dfs = []
#         for group in processed_groups:
#             for chunk in group:
#                 chunk['new_account_id'] = new_account_id
#                 processed_dfs.append(chunk)
#                 new_account_id += 1

#         # Concatenate all processed DataFrames
#         final_df = pd.concat(processed_dfs, ignore_index=True)
#         #grouped = final_df.groupby('new_account_id')
#         synth_bf = pd.read_csv('../Banksformer/synth_tcode.csv')
#         synth_bf = synth_bf.rename(columns={'transaction_code': 'tcode'})
#         grouped = synth_bf.groupby('account_id')

#         generator = tf.saved_model.load("gen_ctgan_type1_onehot")

#         # Reading the object back from the file
#         with open('transformerobject_onehot.pkl', 'rb') as file:
#             transformer = pickle.load(file)

#         with open('samplerobject_onehot.pkl', 'rb') as file:
#             sampler = pickle.load(file)

#         #one hot encoder object
#         for column_info in transformer._column_transform_info_list:
#             if column_info.column_name == 'tcode':
#                ohe = column_info.transform
#         get_dtme = lambda d: calendar.monthrange(d.year, d.month)[1] - d.day
#         START_DATE = raw_data['datetime'].min()
#         MAX_YEARS_SPAN = 25
#         END_DATE = START_DATE.replace(year = START_DATE.year+ MAX_YEARS_SPAN)
#         ALL_DATES = [START_DATE + datetime.timedelta(i) for i in range((END_DATE - START_DATE).days)]
#         #AD = np.array([(d.month % 12, d.day % 31, d.weekday() % 7, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
#         AD = np.array([(d.month -1, d.day -1, d.weekday() -1, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
#         max_days = 100
#         #construct list of starting indexes
#         # start_dates = raw_data.groupby('account_id')['datetime'].min()
#         # sampled_start_dates = start_dates.sample(n)
#         # diff_days = (sampled_start_dates - START_DATE).dt.days

#         start_date_opts = raw_data.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
#         sampled_start_dates = np.random.choice(start_date_opts, size=n) # sample start dates from real data
#         # Convert sampled_start_dates to a pandas Series of Timestamps
#         sampled_start_dates_series = pd.Series(pd.to_datetime(sampled_start_dates))

#         diff_days = (sampled_start_dates_series - START_DATE).dt.days



#         date_inds = diff_days.tolist()

#         recovered_data_list = []
#         visualize_list = []
#         #construct td_ps
#         # Empty array to store the pdf values
#         td_ps = []


#         for seq_i, (name, group) in enumerate(grouped):
#             # name is the unique account_id
#             # group is a DataFrame with all rows for that account_id
#             print(seq_i)
#             if seq_i > n-1:
#                 break
#             account_id = name
#             m = len(group)
#             tcode = group['tcode']
#             si = date_inds[seq_i] 
#             for i in range(m):

#                 #def sample_external_condvec(self, tcode=tcode.iloc[i], oheobject=ohe)
                
#                 condvec = sampler.sample_external_condvec(tcode.iloc[i], ohe)
#                 length_of_seq = 1
#                 mean = tf.zeros(shape=(1, embedding_dim), dtype=tf.float32)
#                 std = mean + 1
#                 fakez = tf.random.normal(shape=(length_of_seq, embedding_dim),mean=mean, stddev=std)
#                 c1 = condvec
#                 c1 = tf.convert_to_tensor(np.array(c1))
#                 c1 = tf.cast(c1, dtype=tf.float32)
#                 fakez = tf.concat([fakez, c1], axis=1)
#                 fake = generator(fakez)
#                 fakeact = _apply_activate2(transformer, fake)
#                 recovered_data, day_ps, dtme_ps, dow_ps, month_ps = transformer.generate_raw_ps(fakeact,LOG_AMOUNT_SCALE)
                
#                 mean = recovered_data['td']
#                 std = 3.06
#                 td_ps = norm.pdf(AD[si:si+max_days,3]-si, mean, std)
#                 ps = month_ps[AD[si:si+max_days,0]]*day_ps[AD[si:si+max_days,1]]*dow_ps[AD[si:si+max_days,2]] *dtme_ps[AD[si:si+max_days,-1]] * td_ps
                
#                 # Check for NaN values in ps
#                 if np.isnan(ps).any():
#                     print("Warning: NaN values found in ps")
#                     ps = np.nan_to_num(ps)  # Replaces NaNs with zero and Infs with large finite numbers

#                 # Check for sum(ps) is zero
#                 if sum(ps) == 0:
#                     print("Warning: sum of ps is zero")
#                     ps = ps + 1e-10  # To prevent division by zero
#                 if greedy_decode:
#                     timesteps = np.argmax(ps)
#                 else:
#                     timesteps = np.random.choice(max_days, p=ps/sum(ps))
#                 if i == 0:
#                     timesteps = 0
#                 recovered_data['date'] = ALL_DATES[timesteps + si]
#                 recovered_data['account_id'] = account_id
#                 recovered_data_list.append(recovered_data)
#                 si = timesteps + si
#                 iteration_dict = {
#                         'day_ps': day_ps,
#                         'dtme_ps': dtme_ps,
#                         'dow_ps': dow_ps,
#                         'month_ps': month_ps,
#                         'td_ps': td_ps,
#                         'account_id': account_id,
#                         'mean':mean,
#                         'std': std,
#                         'si':si,
#                         'timesteps':timesteps,
#                         'ps':ps
#                     }
#                 visualize_list.append(iteration_dict)
#         final_recovered_data = pd.concat(recovered_data_list)
#         experiment_name = 'synth_ctgan_type1_onehot_trans'
#         filename = '../DATA/' + experiment_name +'.csv'
#         final_recovered_data.to_csv(filename, index=False)
#         print('finish')

# def main():
#     with tf.device('/gpu:1'):
#         n = 2300
#         embedding_dim = 100
#         greedy_decode=False
#         df = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
#         print('preprocessing data....')
#         raw_data, LOG_AMOUNT_SCALE, TD_SCALE = preprocess_data_czech(df)
#         raw = raw_data.copy()

#         final_raw = raw[['log_amount_sc', 'tcode', 'td', 'day', 'dow', 'dtme', 'month']]
#         #final_raw = raw[['log_amount_sc', 'tcode', 'td']]

#         # Process each group
#         processed_groups = raw.groupby('account_id').apply(process_group)

#         # Flatten the list of DataFrames and assign new account IDs
#         new_account_id = 0
#         processed_dfs = []
#         for group in processed_groups:
#             for chunk in group:
#                 chunk['new_account_id'] = new_account_id
#                 processed_dfs.append(chunk)
#                 new_account_id += 1

#         # Concatenate all processed DataFrames
#         final_df = pd.concat(processed_dfs, ignore_index=True)
#         #grouped = final_df.groupby('new_account_id')
#         synth_bf = pd.read_csv('../Banksformer/synth_tcode.csv')
#         synth_bf = synth_bf.rename(columns={'transaction_code': 'tcode'})
#         grouped = synth_bf.groupby('account_id')

#         generator = tf.saved_model.load("gen_ctgan_type1_clock")

#         # Reading the object back from the file
#         with open('transformerobject_clock.pkl', 'rb') as file:
#             transformer = pickle.load(file)

#         with open('samplerobject_clock.pkl', 'rb') as file:
#             sampler = pickle.load(file)

#         #one hot encoder object
#         for column_info in transformer._column_transform_info_list:
#             if column_info.column_name == 'tcode':
#                ohe = column_info.transform
#         get_dtme = lambda d: calendar.monthrange(d.year, d.month)[1] - d.day
#         START_DATE = raw_data['datetime'].min()
#         MAX_YEARS_SPAN = 25
#         END_DATE = START_DATE.replace(year = START_DATE.year+ MAX_YEARS_SPAN)
#         ALL_DATES = [START_DATE + datetime.timedelta(i) for i in range((END_DATE - START_DATE).days)]
#         #AD = np.array([(d.month % 12, d.day % 31, d.weekday() % 7, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
#         AD = np.array([(d.month -1, d.day -1, d.weekday() -1, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
#         max_days = 100
#         #construct list of starting indexes
#         # start_dates = raw_data.groupby('account_id')['datetime'].min()
#         # sampled_start_dates = start_dates.sample(n)
#         # diff_days = (sampled_start_dates - START_DATE).dt.days

#         start_date_opts = raw_data.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
#         sampled_start_dates = np.random.choice(start_date_opts, size=n) # sample start dates from real data
#         # Convert sampled_start_dates to a pandas Series of Timestamps
#         sampled_start_dates_series = pd.Series(pd.to_datetime(sampled_start_dates))

#         diff_days = (sampled_start_dates_series - START_DATE).dt.days



#         date_inds = diff_days.tolist()

#         recovered_data_list = []
#         visualize_list = []
#         #construct td_ps
#         # Empty array to store the pdf values
#         td_ps = []


#         for seq_i, (name, group) in enumerate(grouped):
#             # name is the unique account_id
#             # group is a DataFrame with all rows for that account_id
#             print(seq_i)
#             if seq_i > n-1:
#                 break
#             account_id = name
#             m = len(group)
#             tcode = group['tcode']
#             si = date_inds[seq_i] 
#             for i in range(m):

#                 #def sample_external_condvec(self, tcode=tcode.iloc[i], oheobject=ohe)
                
#                 condvec = sampler.sample_external_condvec(tcode.iloc[i], ohe)
#                 length_of_seq = 1
#                 mean = tf.zeros(shape=(1, embedding_dim), dtype=tf.float32)
#                 std = mean + 1
#                 fakez = tf.random.normal(shape=(length_of_seq, embedding_dim),mean=mean, stddev=std)
#                 c1 = condvec
#                 c1 = tf.convert_to_tensor(np.array(c1))
#                 c1 = tf.cast(c1, dtype=tf.float32)
#                 fakez = tf.concat([fakez, c1], axis=1)
#                 fake = generator(fakez)
#                 fakeact = _apply_activate2(transformer, fake)
#                 recovered_data, day_ps, dtme_ps, dow_ps, month_ps = transformer.generate_raw_ps(fakeact,LOG_AMOUNT_SCALE)
                
#                 mean = recovered_data['td']
#                 std = 3.06
#                 td_ps = norm.pdf(AD[si:si+max_days,3]-si, mean, std)
#                 ps = month_ps[AD[si:si+max_days,0]]*day_ps[AD[si:si+max_days,1]]*dow_ps[AD[si:si+max_days,2]] *dtme_ps[AD[si:si+max_days,-1]] * td_ps
                
#                 # Check for NaN values in ps
#                 if np.isnan(ps).any():
#                     print("Warning: NaN values found in ps")
#                     ps = np.nan_to_num(ps)  # Replaces NaNs with zero and Infs with large finite numbers

#                 # Check for sum(ps) is zero
#                 if sum(ps) == 0:
#                     print("Warning: sum of ps is zero")
#                     ps = ps + 1e-10  # To prevent division by zero
#                 if greedy_decode:
#                     timesteps = np.argmax(ps)
#                 else:
#                     timesteps = np.random.choice(max_days, p=ps/sum(ps))
#                 if i == 0:
#                     timesteps = 0
#                 recovered_data['date'] = ALL_DATES[timesteps + si]
#                 recovered_data['account_id'] = account_id
#                 recovered_data_list.append(recovered_data)
#                 si = timesteps + si
#                 iteration_dict = {
#                         'day_ps': day_ps,
#                         'dtme_ps': dtme_ps,
#                         'dow_ps': dow_ps,
#                         'month_ps': month_ps,
#                         'td_ps': td_ps,
#                         'account_id': account_id,
#                         'mean':mean,
#                         'std': std,
#                         'si':si,
#                         'timesteps':timesteps,
#                         'ps':ps
#                     }
#                 visualize_list.append(iteration_dict)
#         final_recovered_data = pd.concat(recovered_data_list)
#         experiment_name = 'synth_ctgan_type1_clock_trans_sm'
#         filename = '../DATA/' + experiment_name +'.csv'
#         final_recovered_data.to_csv(filename, index=False)
#         print('finish')



def main():
    """ The order of columns is important, 'tcode' should be the first discrete column"""
    with tf.device('/gpu:0'):

        df = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
        print('preprocessing data....')
        raw_data, LOG_AMOUNT_SCALE, TD_SCALE = preprocess_data_czech(df)
        raw = raw_data.copy()

        final_raw = raw[['log_amount_sc', 'tcode', 'td', 'day', 'dow', 'dtme', 'month']]
        #final_raw = raw[['log_amount_sc', 'tcode', 'td']]

        # Process each group
        processed_groups = raw.groupby('account_id').apply(process_group)

        # Flatten the list of DataFrames and assign new account IDs
        new_account_id = 0
        processed_dfs = []
        for group in processed_groups:
            for chunk in group:
                chunk['new_account_id'] = new_account_id
                processed_dfs.append(chunk)
                new_account_id += 1

        # Concatenate all processed DataFrames
        final_df = pd.concat(processed_dfs, ignore_index=True)



        synth_bf = pd.read_csv('../Banksformer/synth_tcode.csv')
        grouped = synth_bf.groupby('account_id')
        #grouped = final_df.groupby('new_account_id')

        

        print('transforming data...')
        transformer = DataTransformer(date_transformation='clock')
        #transformer = DataTransformer(date_transformation='clock')
        
        transformer.fit(final_raw, discrete_columns=('tcode'), date_columns= ('day', 'dow', 'dtme', 'month'))     
        data_t   = transformer.transform(final_raw)                #matrix of transformed data
        output_info = transformer.output_info_list

        account_id_counts = raw_data['account_id'].value_counts().sort_index()
        trans_sizes = np.array(account_id_counts)
        assert sum(trans_sizes) == data_t.shape[0]
        transactions = np.split(data_t, np.cumsum(trans_sizes)[:-1])   #transactions is the list of arrays, each array is for an individual customer 


        log_frequency = True
        sampler = DataSampler(data_t, transactions, output_info, log_frequency)
        
     
        with open('transformerobject_clock.pkl', 'wb') as file:
            pickle.dump(transformer, file)

        with open('samplerobject_clock.pkl', 'wb') as file:
            pickle.dump(sampler, file)

        model = CTGAN()
        generator  = model.make_generator(sampler, transformer)
        discriminator = model.make_discriminator(sampler, transformer)

        train = Train(transformer, sampler, generator, discriminator, epochs=100)
        experiment_name = 'ctgan_type1_clock_trans_two'
        train.train(raw_data, experiment_name)
        print('synthesize data')
        data, visdata = train.synthesise_data_bank_externaltcode_banksformer(5000, raw_data, grouped, LOG_AMOUNT_SCALE)
        print('finish')
        filename = '../DATA/synth_' + experiment_name +'.csv'
        data.to_csv(filename, index=False)

        filename_vis = '../DATA/'+ experiment_name + '_vis' + '.pickle'
        with open(filename_vis, 'wb') as f:
            pickle.dump(visdata, f)
    
    
   



if __name__ == "__main__":
    main()


    

