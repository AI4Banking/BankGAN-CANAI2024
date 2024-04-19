#The same as https://github.com/sdv-dev/CTGAN/blob/master/ctgan/data_sampler.py with a correction


import numpy as np
import random
import os
import numpy as np
import tensorflow as tf
# Set seeds
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""
    
    #output_info is transformer.output_info_list : 
    def __init__(self, data, transactions, output_info, log_frequency):

        """order of input features: ['log_amount_sc', 'tcode', 'td', 'day', 'dow', 'dtme', 'month']
        output_info:
        [[SpanInfo(dim=1, activation_fn='tanh'), SpanInfo(dim=10, activation_fn='softmax')],---> log_amount_sc
        [SpanInfo(dim=16, activation_fn='softmax')],                                        ----> tcode
        [SpanInfo(dim=1, activation_fn='tanh'),SpanInfo(dim=8, activation_fn='softmax')],   -----> td
        [SpanInfo(dim=31, activation_fn='softmax')],                                        -----> day
        [SpanInfo(dim=7, activation_fn='softmax')],                                         -----> dow
        [SpanInfo(dim=31, activation_fn='softmax')],                                        -----> dtme
        [SpanInfo(dim=12, activation_fn='softmax')]]                                        -----> month
        """
        
        self.transactions = transactions
        self._data = data
        
        #column_info is an element of transformer.output_info_list
        def is_discrete_column(column_info):
            return (len(column_info) == 1 and column_info[0].activation_fn == 'softmax')

        n_discrete_columns = sum([1 for column_info in output_info if is_discrete_column(column_info)])   # 5

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_matrix_indexes = np.zeros(n_discrete_columns, dtype='int32')   #added by myself

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max([column_info[0].dim for column_info in output_info if is_discrete_column(column_info)], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([column_info[0].dim for column_info in output_info if is_discrete_column(column_info)])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob  #array of shape(5,31=max_category_length): 
                                                                                                 #probability of sampling each category from each categorical feature
                self._discrete_column_cond_st[current_id] = current_cond_st    #offset indexes in conditional vector(array([ 0, 16, 47, 54, 85])
                self._discrete_column_n_category[current_id] = span_info.dim   #number of categoriries in each discrete column(array([16, 31,  7, 31, 12])
                self._discrete_column_matrix_indexes[current_id] = st    #starting index of categorical column in data_t (array([ 11,  36,  67,  74, 105])
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.
        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (self._discrete_column_cond_st[discrete_column_id] + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.
        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]

    
    def sample_original_condvec(self, idx, row_idx):
        """ sample conditional vector based on the original data , for generating data"""
        cond = np.zeros((1, self._n_categories), dtype='float32')
        col_idx = 0               #assumed that transaction code is the first discrete column
        matrix_st = self._discrete_column_matrix_indexes[col_idx]             #corrected by myself
        matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
        pick = np.argmax(self.transactions[idx][row_idx, matrix_st:matrix_ed])
        cond[0, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def sample_external_condvec(self, tcode, oheobject):
        """ sample conditional vector based on another generated transaction codes, for generating data"""
        cond = np.zeros((1, self._n_categories), dtype='float32')
        col_idx = 0            #assumed that transaction code is the first discrete column
        pick = oheobject.dummies.index(tcode)
        cond[0, pick + self._discrete_column_cond_st[col_idx]] = 1
        return cond

    