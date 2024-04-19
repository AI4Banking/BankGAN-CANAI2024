# Based on https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
import pandas as pd
#from datetime import datetime
import os
import datetime
import calendar
import scipy.stats as st
import matplotlib.pyplot as plt
from calendar import monthrange
from scipy.stats import norm
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import pickle
import random
import os
import numpy as np
# Set seeds
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def bulk_encode_time_value(val, max_val):
    x = tf.sin(2 * tf.constant(np.pi, dtype=tf.float32) / max_val * val)
    y = tf.cos(2 * tf.constant(np.pi, dtype=tf.float32) / max_val * val)
    return tf.stack([x, y], axis=1)



def clock_to_probs(pt, pts):
    EPS_CLOCKP = tf.constant(0.01, dtype=tf.float32)
    ds = pts - pt
    sq_ds = tf.reduce_sum(tf.square(ds + EPS_CLOCKP), axis=1)
    raw_ps = 1 / sq_ds
    return raw_ps / tf.reduce_sum(raw_ps)



class Train(object):
    """
    generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300."""
            
    def __init__(self, transformer, sampler, generator, discriminator, embedding_dim=100, batch_size=700, generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6,discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=100, pac=10):
        assert batch_size % 2 == 0
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._embedding_dim = embedding_dim
        self._transformer = transformer
        self._sampler = sampler
        self.pac = pac
        self.generator = generator
        self.discriminator = discriminator

        

    
    
    def _gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        """Samples from the Gumbel-Softmax distribution
        :cite:`maddison2016concrete`, :cite:`jang2016categorical` and
        optionally discretizes.
        Parameters
        ----------
        logits: tf.Tensor
            Un-normalized log probabilities.
        tau: float, default=1.0
            Non-negative scalar temperature.
        hard: bool, default=False
            If ``True``, the returned samples will be discretized as
            one-hot vectors, but will be differentiated as soft samples.
        dim: int, default=1
            The dimension along which softmax will be computed.
        Returns
        -------
        tf.Tensor
            Sampled tensor of same shape as ``logits`` from the
            Gumbel-Softmax distribution. If ``hard=True``, the returned samples
            will be one-hot, otherwise they will be probability distributions
            that sum to 1 across ``dim``.
        """

        gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
        gumbels = gumbel_dist.sample(tf.shape(logits))
        gumbels = (logits + gumbels) / tau
        output = tf.nn.softmax(gumbels, dim)

        if hard:
            index = tf.math.reduce_max(output, 1, keepdims=True)
            output_hard = tf.cast(tf.equal(output, index), output.dtype)
            output = tf.stop_gradient(output_hard - output) + output
        return output
        
    
    
    
    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(tf.math.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'sigmoid':
                    ed = st + span_info.dim
                    data_t.append(tf.math.sigmoid(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return tf.concat(data_t, axis=1)

    def _apply_activate2(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(tf.math.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'sigmoid':
                    ed = st + span_info.dim
                    data_t.append(tf.math.sigmoid(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = tf.nn.softmax(data[:, st:ed])
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return tf.concat(data_t, axis=1)

    def cross_entropy_loss(self, data, c, m, output_info):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in output_info:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    #c is the conditional vector
                    labels=c[:, st_c:ed_c]
                    #data is generated by generator
                    logits=data[:, st:ed]
                    tmp = tf.nn.softmax_cross_entropy_with_logits(
                        labels,
                        logits)
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = tf.stack(loss, axis=1)
        #we are interested in the loss for the feature that is conditioned on 
        m1 = tf.cast(m, dtype=tf.float32)
        return tf.reduce_mean(loss * m1)


    def calc_gradient_penalty(self, real_cat, fake_cat, gp_lambda=1):
        #random alpha(between 0 and 1) for each input batch to discriminator
        alpha = tf.random.uniform([real_cat.shape[0] // self.pac, 1, 1], 0., 1.)
        alpha = tf.tile(alpha, tf.constant([1, self.pac, real_cat.shape[1]], tf.int32))
        alpha = tf.reshape(alpha, [-1, real_cat.shape[1]])
        
        interpolates = alpha * real_cat + ((1 - alpha) * fake_cat)
        pacdim = self.pac * real_cat.shape[1]
        interpolates_disc = tf.reshape(interpolates,[-1, pacdim])

        with tf.GradientTape() as tape:
            tape.watch(interpolates_disc)
            pred_interpolates = self.discriminator(interpolates_disc, training=True)

        gradients = tape.gradient(pred_interpolates, interpolates_disc)

        gradients_view = tf.norm(tf.reshape(gradients, [-1, self.pac * real_cat.shape[1]]), axis=1) - 1
        gradient_penalty = tf.reduce_mean(tf.square(gradients_view)) * gp_lambda
        return gradient_penalty
    
    def train(self, raw_data, experiment_name):
        
        optimizerG = tf.keras.optimizers.Adam(learning_rate = self._generator_lr, beta_1=0.5, beta_2=0.9, decay = self._generator_decay)
        optimizerD = tf.keras.optimizers.Adam(learning_rate = self._discriminator_lr, beta_1=0.5, beta_2=0.9, decay = self._discriminator_decay)
        
        # l2_norm_clip = 1.0
        # noise_multiplier = 2.2
        # num_microbatches = self._batch_size
        
        # optimizerG = DPKerasAdamOptimizer(
        #         l2_norm_clip=l2_norm_clip,
        #         noise_multiplier=noise_multiplier,
        #         num_microbatches=num_microbatches,
        #         learning_rate=self._generator_lr)


        # optimizerD = DPKerasAdamOptimizer(
        #         l2_norm_clip=l2_norm_clip,
        #         noise_multiplier=noise_multiplier,
        #         num_microbatches=num_microbatches,
        #         learning_rate=self._discriminator_lr)
        
        
        #csv_file = '../DATA/ctgan_test2_losses.csv'
        csv_file = '../DATA/' + experiment_name + '_losses' +'.csv'
        results_df = pd.DataFrame(columns=['Generator Loss', 'Discriminator Loss_average', 'Discriminator Loss_fake', 'Discriminator Loss_real',
                                           'Cross Entropy', 'Gradient Penalty'])

        mean = tf.zeros(shape=(self._batch_size, self._embedding_dim), dtype=tf.float32)
        std = mean + 1
        steps_per_epoch = max(len(raw_data)// self._batch_size, 1)

        data_dim = self._transformer.output_dimensions
        dim_cond_vec = self._sampler.dim_cond_vec()
        #dim of the input to discriminator
        input_dim = data_dim + dim_cond_vec
        pacdim = input_dim * self.pac
        print('start of training')
        for i in range(self._epochs):
            epoch_start = datetime.datetime.now()
            for id_ in range(steps_per_epoch):
                # step_per_epoch_start = datetime.now()
                for n in range(self._discriminator_steps):
                    fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
                    condvec = self._sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = tf.convert_to_tensor(np.array(c1))
                        c1 = tf.cast(c1, dtype=tf.float32)
                        #c1 = tf.identity(c1, name=None) # Optional, just to ensure a new tensor is created

                        m1 = tf.convert_to_tensor(np.array(m1))
                        m1 = tf.cast(m1, dtype=tf.int32)
                        #m1 = tf.identity(m1, name=None) # Optional, just to ensure a new tensor is created
                        fakez = tf.concat([fakez, c1], axis=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._sampler.sample_data(self._batch_size, col[perm], opt[perm])
                        c2 = tf.gather(c1, indices=perm)
                        
                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                       fake_cat = tf.concat([fakeact, c1], axis=1)
                       real_cat = tf.concat([real, c2], axis=1)
                    else:
                       fake_cat = fakeact
                       real_cat = real

                    # reshape the data for packed discriminator
                    fake_cat_disc = tf.reshape(fake_cat,[-1, pacdim])
                    real_cat_disc = tf.reshape(real_cat, [-1, pacdim])

                    with tf.GradientTape() as tape:
                        y_fake = self.discriminator(fake_cat_disc, training=True)
                        y_real = self.discriminator(real_cat_disc, training=True)
                        pen = self.calc_gradient_penalty(real_cat, fake_cat, 1)
                        loss_d = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake)) + pen
                        loss_d_real = -tf.reduce_mean(y_real)
                        loss_d_fake = tf.reduce_mean(y_fake)
                    grads_disc = tape.gradient(loss_d, self.discriminator.trainable_variables)
                    optimizerD.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

                fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
                condvec = self._sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    
                else:
                    c1, m1, col, opt = condvec
                    c1 = tf.convert_to_tensor(np.array(c1))
                    c1 = tf.cast(c1, dtype=tf.float32)

                    m1 = tf.convert_to_tensor(np.array(m1))
                    m1 = tf.cast(m1, dtype=tf.int32)
                
                    fakez = tf.concat([fakez, c1], axis=1)

                with tf.GradientTape() as tape:
                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                       fake_temp = tf.concat([fakeact, c1], axis=1)
                       y_fake = self.discriminator(tf.reshape(fake_temp,[-1, pacdim]))
                    else: 
                       y_fake = self.discriminator(tf.reshape(fakeact,[-1, pacdim]))

                    if condvec is None:
                       cross_entropy = 0
                    else:
                       output_info = self._transformer.output_info_list
                       cross_entropy = self.cross_entropy_loss(fake, c1, m1, output_info)

                    # mse_loss = tf.keras.losses.MSE(real[:,:1], fakeact[:,:1])
                    # average_mse_loss = tf.reduce_mean(mse_loss)
                    # amt_dim = output_info[0][1].dim + 1
                    # labels = real_cat[:, 1:amt_dim]
                    # logits = fake[:,1:amt_dim]
                    # tmp = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
                    # amt_cross_entropy = tf.reduce_mean(tmp)
                    loss_g = -tf.reduce_mean(y_fake) + cross_entropy 
                    #loss_g = tf.reduce_mean(y_fake) + cross_entropy
                grads_gen = tape.gradient(loss_g, self.generator.trainable_variables)
                optimizerG.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
                # step_per_epoch_end = datetime.now()
                # step_per_epoch_time = step_per_epoch_end - step_per_epoch_start
                # print(f'Step per Epoch time {id_}, time: {step_per_epoch_time}')
            epoch_end = datetime.datetime.now()
            epoch_time = epoch_end - epoch_start
            print(f'Epoch {i+1}, Loss G: {loss_g.numpy(): .4f}, Loss D: {loss_d.numpy(): .4f}, cross_entropy: {cross_entropy.numpy(): .2f}',flush=True)
            print(f'Epoch time {i+1}, time: {epoch_time}')
            #recording of generator loss and discriminator loss in dataframe : losses.csv
            loss_g_numpy = round(loss_g.numpy(), 2)
            loss_d_numpy = round(loss_d.numpy(), 2)
            cross_entropy_numpy = round(cross_entropy.numpy(), 2)
            pen_numpy = round(pen.numpy(), 2)
            loss_d_real_numpy = round(loss_d_real.numpy(), 2)
            loss_d_fake_numpy = round(loss_d_fake.numpy(), 2)
            epoch = i+1
            # columns=['Generator Loss', 'Discriminator Loss_average', 'Discriminator Loss_fake', 'Discriminator Loss_real','Cross Entropy', 'Gradient Penalty'])
            results_df.loc[epoch] = [loss_g_numpy, loss_d_numpy, loss_d_fake_numpy, loss_d_real_numpy, cross_entropy_numpy , pen_numpy]

        
        # Save the dataframe to a file
        results_df.to_csv(csv_file)
        tf.saved_model.save(self.generator, "gen_ctgan_type1_rbf_dp")
        tf.saved_model.save(self.discriminator, "disc_ctgan_type1_rbf_dp")


            
    def synthesise_data_bank_externaltcode_td(self, n, raw_data ,grouped):
        """ 
        conditional vector is constructed based on the generated sequence of transactions,
        for generating data, Start date for each transaction is chosen randomly from the account's starting dates, 
        and the remaining datetimes is constructed based on the 'td'. 
        Args:
            n (int):
                Number of transaction sequences.
            raw_data (dataframe):
                Real Date
        Returns:
                pandas.DataFrame   """
        
        #one hot encoder object
        for column_info in self._transformer._column_transform_info_list:
            if column_info.column_name == 'tcode':
               ohe = column_info.transform
        #construct list of starting indexes
        START_DATE = raw_data['datetime'].min()
        # start_dates = raw_data.groupby('account_id')['datetime'].min()
        # sampled_start_dates = start_dates.sample(n)
        start_date_opts = raw_data.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
        sampled_start_dates = np.random.choice(start_date_opts, size=n) # sample start dates from real data
        # Convert sampled_start_dates to a pandas Series of Timestamps
        sampled_start_dates_series = pd.Series(pd.to_datetime(sampled_start_dates))

        diff_days = (sampled_start_dates_series - START_DATE).dt.days
        date_inds = diff_days.tolist()

        data=[]
        account_ids = []  # Initialize an empty list for account_ids
        for seq_i, (name, group) in enumerate(grouped):
            # name is the unique account_id
            # group is a DataFrame with all rows for that account_id
            print(seq_i)
            if seq_i > n-1:
                break
            account_id = name
            m = len(group)
            tcode = group['tcode']
            si = date_inds[seq_i] 
            # Add the account_id repeated m times to the account_ids list
            account_ids.extend([account_id] * m)
            for i in range(m):

                #def sample_external_condvec(self, tcode=tcode.iloc[i], oheobject=ohe)
                
                condvec = self._sampler.sample_external_condvec(tcode.iloc[i], ohe)
                length_of_seq = 1
                mean = tf.zeros(shape=(1, self._embedding_dim), dtype=tf.float32)
                std = mean + 1
                fakez = tf.random.normal(shape=(length_of_seq, self._embedding_dim),mean=mean, stddev=std)
                c1 = condvec
                c1 = tf.convert_to_tensor(np.array(c1))
                c1 = tf.cast(c1, dtype=tf.float32)
                fakez = tf.concat([fakez, c1], axis=1)
                fake = self.generator(fakez)
                fakeact = self._apply_activate2(fake)
                data.append(fakeact.numpy())
        data = np.concatenate(data, axis=0)
        synth = self._transformer.inverse_transform(data)
        synth['account_id'] = account_ids  # Add the account_id column
        synth['td'] = synth['td'].apply(lambda x: 0 if x < 0 else round(x))
        

         # Identify the first transaction for each account
        first_transactions = synth.groupby('account_id').head(1).index
        # Set 'td' to 0 only for the first transactions
        synth.loc[first_transactions, 'td'] = 0


        # Step 2: Calculate cumulative sum of 'td' for each account_id
        synth['cumulative_td'] = synth.groupby('account_id')['td'].cumsum()

        for i, account_id in enumerate(synth['account_id'].unique()):
            start_date_str = sampled_start_dates[i].strftime('%Y-%m-%d')
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')

            # Filter the rows for the current account_id
            account_rows = synth[synth['account_id'] == account_id]

            # Calculate the date for each transaction
            for index, row in account_rows.iterrows():
                if index == 0:
                  synth.at[index, 'datetime'] = start_date
                else:
                  transaction_date = start_date + datetime.timedelta(days=row['cumulative_td'])
                  synth.at[index, 'datetime'] = transaction_date

        return synth
                
                


    def synthesise_data_bank_externaltcode_banksformer(self, n, raw_data ,grouped, LOG_AMOUNT_SCALE, greedy_decode=False):
        """grouped = externaltcode_df.groupby('account_id)
        conditional vector is constructed based on a generated sequence of trandsactions, transaction dates are computed based on banksformer method
        p(Time Delta) is constructed based on a fixed std = 3.06, and mean= predicted time delta
        """
        #one hot encoder object
        for column_info in self._transformer._column_transform_info_list:
            if column_info.column_name == 'tcode':
               ohe = column_info.transform
        get_dtme = lambda d: calendar.monthrange(d.year, d.month)[1] - d.day
        START_DATE = raw_data['datetime'].min()
        MAX_YEARS_SPAN = 25
        END_DATE = START_DATE.replace(year = START_DATE.year+ MAX_YEARS_SPAN)
        ALL_DATES = [START_DATE + datetime.timedelta(i) for i in range((END_DATE - START_DATE).days)]
        #AD = np.array([(d.month % 12, d.day % 31, d.weekday() % 7, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
        AD = np.array([(d.month -1, d.day -1, d.weekday() -1, i, d.year, get_dtme(d)) for i, d in enumerate(ALL_DATES)])
        max_days = 100
        #construct list of starting indexes
        # start_dates = raw_data.groupby('account_id')['datetime'].min()
        # sampled_start_dates = start_dates.sample(n)
        # diff_days = (sampled_start_dates - START_DATE).dt.days

        start_date_opts = raw_data.groupby("account_id")["datetime"].min().dt.date.to_list()   #len = 4500
        sampled_start_dates = np.random.choice(start_date_opts, size=n) # sample start dates from real data
        # Convert sampled_start_dates to a pandas Series of Timestamps
        sampled_start_dates_series = pd.Series(pd.to_datetime(sampled_start_dates))

        diff_days = (sampled_start_dates_series - START_DATE).dt.days



        date_inds = diff_days.tolist()

        recovered_data_list = []
        visualize_list = []
        #construct td_ps
        # Empty array to store the pdf values
        td_ps = []


        for seq_i, (name, group) in enumerate(grouped):
            # name is the unique account_id
            # group is a DataFrame with all rows for that account_id
            print(seq_i)
            if seq_i > n-1:
                break
            account_id = name
            m = len(group)
            tcode = group['tcode']
            si = date_inds[seq_i] 
            for i in range(m):

                #def sample_external_condvec(self, tcode=tcode.iloc[i], oheobject=ohe)
                
                condvec = self._sampler.sample_external_condvec(tcode.iloc[i], ohe)
                length_of_seq = 1
                mean = tf.zeros(shape=(1, self._embedding_dim), dtype=tf.float32)
                std = mean + 1
                fakez = tf.random.normal(shape=(length_of_seq, self._embedding_dim),mean=mean, stddev=std)
                c1 = condvec
                c1 = tf.convert_to_tensor(np.array(c1))
                c1 = tf.cast(c1, dtype=tf.float32)
                fakez = tf.concat([fakez, c1], axis=1)
                fake = self.generator(fakez)
                fakeact = self._apply_activate2(fake)
                recovered_data, day_ps, dtme_ps, dow_ps, month_ps = self._transformer.generate_raw_ps(fakeact,LOG_AMOUNT_SCALE)
                
                mean = recovered_data['td']
                std = 3.06
                td_ps = norm.pdf(AD[si:si+max_days,3]-si, mean, std)
                ps = month_ps[AD[si:si+max_days,0]]*day_ps[AD[si:si+max_days,1]]*dow_ps[AD[si:si+max_days,2]] *dtme_ps[AD[si:si+max_days,-1]] * td_ps
                
                # Check for NaN values in ps
                if np.isnan(ps).any():
                    print("Warning: NaN values found in ps")
                    ps = np.nan_to_num(ps)  # Replaces NaNs with zero and Infs with large finite numbers

                # Check for sum(ps) is zero
                if sum(ps) == 0:
                    print("Warning: sum of ps is zero")
                    ps = ps + 1e-10  # To prevent division by zero
                if greedy_decode:
                    timesteps = np.argmax(ps)
                else:
                    timesteps = np.random.choice(max_days, p=ps/sum(ps))
                if i == 0:
                    timesteps = 0
                recovered_data['date'] = ALL_DATES[timesteps + si]
                recovered_data['account_id'] = account_id
                recovered_data_list.append(recovered_data)
                si = timesteps + si
                iteration_dict = {
                        'day_ps': day_ps,
                        'dtme_ps': dtme_ps,
                        'dow_ps': dow_ps,
                        'month_ps': month_ps,
                        'td_ps': td_ps,
                        'account_id': account_id,
                        'mean':mean,
                        'std': std,
                        'si':si,
                        'timesteps':timesteps,
                        'ps':ps
                    }
                visualize_list.append(iteration_dict)
        final_recovered_data = pd.concat(recovered_data_list)
        return final_recovered_data, visualize_list    

    