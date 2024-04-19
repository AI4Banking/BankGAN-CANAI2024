
import numpy as np
import time
import tensorflow as tf
#from sklego.preprocessing import RepeatingBasisFunction
from .field_info import FieldInfo  # Assuming you named the other file as field_info.py
import random
import os

# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


class TensorEncoder:
    def __init__(self, df, info, max_seq_len, min_seq_len):
        """df: preprocessed real data   """
        self.df = df
        self.info = info
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        self.n_seqs = self.count_seqs_in_df()   #14354
        self.n_seqs_overlap = self.count_seqs_with_overlap()    #72764
        self.n_seqs_overlap_vary = self.count_variable_length_seqs_with_overlap()   #99101

        self.n_feat_inp = sum(self.info.FIELD_DIMS_IN.values())      #number of features
        self.n_feat_tar = sum(self.info.FIELD_DIMS_TAR.values())

        # rbf_dow = RepeatingBasisFunction(n_periods=3, column="dow", input_range=(0,6), remainder="drop")
        # rbf_day = RepeatingBasisFunction(n_periods=3, column="day", input_range=(0,30), remainder="drop")
        # rbf_dtme = RepeatingBasisFunction(n_periods=3, column="dtme", input_range=(0,30), remainder="drop")
        # rbf_month = RepeatingBasisFunction(n_periods=3, column="month", input_range=(0,11), remainder="drop")

        # rbf_dow.fit(df)
        # rbf_day.fit(df)
        # rbf_dtme.fit(df)
        # rbf_month.fit(df)

        # self.rbf_dict = {'dow': rbf_dow, 'day': rbf_day, 'dtme' : rbf_dtme, 'month':rbf_month}
       

    def count_seqs_in_df(self):
        """sequences do not have overlap, the length of sequences are between 'min_seq_len' and 'max_seq_len' """
        gb_aid = self.df.groupby("account_id")["account_id"]
        full_seqs_per_acct = gb_aid.count() // self.max_seq_len
        n_full_seqs = sum(full_seqs_per_acct)
        n_part_seqs = sum(gb_aid.count() - full_seqs_per_acct * self.max_seq_len >= self.min_seq_len)
        return n_full_seqs + n_part_seqs  #14354
    
    
    def count_seqs_with_overlap(self,  slide_step = 10):
        """Count sequences in dataset with overlap using a sliding window.
            sequences are of fixed length=max_seq_len"""
        
        total_seqs = 0
        # Unique account IDs
        valid_ids = self.df['account_id'].value_counts().index.tolist()

        for acc_id in valid_ids:
            acc_data_len = len(self.df[self.df['account_id'] == acc_id])
            
           
            num_seqs_for_this_id = (acc_data_len - self.max_seq_len) // slide_step + 1

            if num_seqs_for_this_id > 0:
                total_seqs += num_seqs_for_this_id

        return total_seqs      #72764
    
    def count_variable_length_seqs_with_overlap(self, slide_step = 10):
        """Count sequences in dataset with overlap using a sliding window for variable sequence lengths, but
         the sequence length are greater than minimum sequence length"""
        
        # The number of sequences will be stored in this variable
        total_seqs = 0
        
        # Unique account IDs
        valid_ids = self.df['account_id'].value_counts().index.tolist()

        for acc_id in valid_ids:
            acc_data_len = len(self.df[self.df['account_id'] == acc_id])
            
            start_idx = 0
            while start_idx + self.max_seq_len <= acc_data_len:
                total_seqs += 1
                start_idx += slide_step

            while acc_data_len - start_idx >= self.min_seq_len:
                total_seqs += 1
                start_idx += slide_step

        return total_seqs       #99101
    
    @staticmethod
    def bulk_encode_time_value(val, max_val):
        x = np.sin(2 * np.pi / max_val * val)
        y = np.cos(2 * np.pi / max_val * val)
        return np.stack([x, y], axis=1)

    def seq_to_inp_tensor(self, seq, seq_i, seq_len):
        """seq is group.iloc[start:start+seq_len], group is df.groupby('account_id')"""
        for k in self.info.DATA_KEY_ORDER:
            depth = self.info.FIELD_DIMS_IN[k]
            st = self.info.FIELD_STARTS_IN[k]
            enc_type = self.info.INP_ENCODINGS[k]
            #if enc_type == "oh_tcode":
            if enc_type.startswith('oh'):
                x = tf.one_hot(seq[k], depth).numpy()
            elif enc_type == "cl":
                max_val = self.info.CLOCK_DIMS[k]
                x = self.bulk_encode_time_value(seq[k], max_val)
                #print(x.shape)
            elif enc_type == "rbf":
                x = self.rbf_dict[k].transform(seq)
                #print(x.shape)
            elif enc_type == "raw":
                x = np.expand_dims(seq[k], 1)
            else:
                raise Exception(f"Got invalid enc_type: {enc_type}")
            self.inp_tensor[seq_i, :seq_len, st:st + depth] = x

    def seq_to_targ_tensor(self, seq, seq_i, seq_len):
   
        for k in self.info.DATA_KEY_ORDER:
            depth = self.info.FIELD_DIMS_TAR[k]
            st = self.info.FIELD_STARTS_TAR[k]
            enc_type = self.info.TAR_ENCODINGS[k]
            if enc_type == "cl-i":
                #max_val = self.info.CLOCK_DIMS[k]
                #x = np.expand_dims(seq[k] % max_val, 1)
                x = np.expand_dims(seq[k], 1)
            elif enc_type == "raw":
                x = np.expand_dims(seq[k], 1)
            elif enc_type == "cl":
                max_val = self.info.CLOCK_DIMS[k]
                x = self.bulk_encode_time_value(seq[k], max_val)
            else:
                raise Exception(f"Got invalid enc_type: {enc_type}")
            self.tar_tensor[seq_i, :seq_len, st:st + depth] = x


    def encode(self, add_attribute_row = True):
        """Compatible with count_seqs_in_df() """
        self.inp_tensor = np.zeros((self.n_seqs, self.max_seq_len, self.n_feat_inp))
        self.tar_tensor = np.zeros((self.n_seqs, self.max_seq_len, self.n_feat_tar))
        self.attributes = np.zeros(self.n_seqs)
        seq_i = 0
        rows_per_acct = {}
        alert_every = 2000
        start_time = time.time()
        for acct_id, group in self.df.groupby("account_id"):
            rows_per_acct[acct_id] = []
            for i in range(len(group) // self.max_seq_len + 1):
                n_trs = len(group)
                start = i * self.max_seq_len
                seq_len = min(self.max_seq_len, n_trs - start)
                if seq_len >= self.min_seq_len:
                    self.seq_to_inp_tensor(group.iloc[start:start + seq_len], seq_i, seq_len)
                    self.seq_to_targ_tensor(group.iloc[start:start + seq_len], seq_i, seq_len)
                    self.attributes[seq_i] = group["age_sc"].iloc[0]
                    rows_per_acct[acct_id].append(seq_i)
                    seq_i += 1
                    if seq_i % alert_every == 0:
                        print(f"Finished encoding {seq_i} of {self.n_seqs} seqs")
        if add_attribute_row:
            self.inp_tensor = np.concatenate([np.repeat(self.attributes[:, None, None], self.n_feat_inp, axis=2), self.inp_tensor], axis=1)
            
        print(f"Took {time.time() - start_time:.2f} secs")

    def encode_with_overlap(self, slide_step=10, add_attribute_row = True):
        """compatible with function count_seqs_with_overlap() """
        self.n_seqs_overlap = self.count_seqs_with_overlap(slide_step)
        self.inp_tensor = np.zeros((self.n_seqs_overlap, self.max_seq_len, self.n_feat_inp))
        self.tar_tensor = np.zeros((self.n_seqs_overlap, self.max_seq_len, self.n_feat_tar))
        self.attributes = np.zeros(self.n_seqs_overlap)
        seq_i = 0
        rows_per_acct = {}
        alert_every = 2000
        start_time = time.time()
        valid_ids = self.df['account_id'].value_counts().index.tolist()

        for acc_id in valid_ids:
            group = self.df[self.df['account_id'] == acc_id]
            rows_per_acct[acc_id] = []
            acc_data_len = len(group)
            
            for start_idx in range(0, acc_data_len - self.max_seq_len + 1, slide_step):
                self.seq_to_inp_tensor(group.iloc[start_idx:start_idx + self.max_seq_len], seq_i, self.max_seq_len)
                self.seq_to_targ_tensor(group.iloc[start_idx:start_idx + self.max_seq_len], seq_i, self.max_seq_len)
                self.attributes[seq_i] = group["age_sc"].iloc[0]
                rows_per_acct[acc_id].append(seq_i)
                seq_i += 1
                if seq_i % alert_every == 0:
                    print(f"Finished encoding {seq_i} of {self.n_seqs_overlap} seqs")
        if add_attribute_row:
            self.inp_tensor = np.concatenate([np.repeat(self.attributes[:, None, None], self.n_feat_inp, axis=2), self.inp_tensor], axis=1)
            
        print(f"Took {time.time() - start_time:.2f} secs")
    
    def encode_variable_length_with_overlap(self, slide_step=40,  add_attribute_row = True):
        """compatible with the function count_variable_length_seqs_with_overlap() """
        self.inp_tensor = np.zeros((self.n_seqs_overlap_vary, self.max_seq_len, self.n_feat_inp))
        self.tar_tensor = np.zeros((self.n_seqs_overlap_vary, self.max_seq_len, self.n_feat_tar))
        self.attributes = np.zeros( self.n_seqs_overlap_vary )

        seq_i = 0
        rows_per_acct = {}
        alert_every = 2000

        valid_ids = self.df['account_id'].value_counts().index.tolist()

        for acc_id in valid_ids:
            group = self.df[self.df['account_id'] == acc_id]
            rows_per_acct[acc_id] = []
            acc_data_len = len(group)
            
            start_idx = 0
            while start_idx + self.max_seq_len <= acc_data_len:
                self.seq_to_inp_tensor(group.iloc[start_idx:start_idx + self.max_seq_len], seq_i, self.max_seq_len)
                self.seq_to_targ_tensor(group.iloc[start_idx:start_idx + self.max_seq_len], seq_i, self.max_seq_len)
                self.attributes[seq_i] = group["age_sc"].iloc[0]
                rows_per_acct[acc_id].append(seq_i)
                seq_i += 1
                if seq_i % alert_every == 0:
                    print(f"Finished encoding {seq_i} of {self.n_seqs} seqs")
                start_idx += slide_step

            
            while acc_data_len - start_idx >= self.min_seq_len:
                seq_len = min(self.max_seq_len, acc_data_len - start_idx)
                self.seq_to_inp_tensor(group.iloc[start_idx:], seq_i, seq_len)
                self.seq_to_targ_tensor(group.iloc[start_idx:], seq_i, seq_len)
                self.attributes[seq_i] = group["age_sc"].iloc[0]
                rows_per_acct[acc_id].append(seq_i)
                seq_i += 1
                if seq_i % alert_every == 0:
                    print(f"Finished encoding {seq_i} of {self.n_seqs_overlap_vary} seqs")
                start_idx += slide_step
        if add_attribute_row:
            self.inp_tensor = np.concatenate([np.repeat(self.attributes[:, None, None], self.n_feat_inp, axis=2), self.inp_tensor], axis=1)
            

   