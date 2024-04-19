# The same as https://github.com/sdv-dev/CTGAN/blob/master/ctgan/data_transformer.py 

from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
from sklego.preprocessing import RepeatingBasisFunction
import random
import os
import numpy as np
# Set seeds
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)




class DataTransformer(object):
    """Data Transformer.
    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using OneHotEncoder from rdt.transformers.
    """

    def __init__(self,date_transformation = None, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.
        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self.CLOCK_DIMS = {"day": 31,"dtme": 31,"dow": 7,"month": 12}
        self.CLOCKS = {}
        self.RBF = {}
        self.date_transformation = date_transformation
        self.rbf_configurations = {
            'dow': {'input_range': (0, 6)},
            'month': {'input_range': (0, 11)},
            'day': {'input_range': (0, 30)},
            'dtme' : {'input_range': (0, 30)}
        }
        for k, val in self.CLOCK_DIMS.items():
            self.CLOCKS[k] = tf.constant(self.bulk_encode_time_value(np.arange(val), val), dtype=tf.float32)

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns(td, amount)
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        if column_name == 'td':
           gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))
        else: 
           gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))
        #gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))  
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)


    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)
    
    def _fit_date(self, data):
        """Fit clock transformation of rbf transformation for date column(month, day, dtme, dow).
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        if self.date_transformation == 'clock':
            return ColumnTransformInfo(
                column_name=column_name, column_type='date', transform = None,
                output_info=[SpanInfo(2, 'tanh')],
                output_dimensions= 2)
        elif self.date_transformation == 'rbf':
            config = self.rbf_configurations.get(column_name)
            # Dynamically create RBF with the configuration
            rbf_transform = RepeatingBasisFunction(n_periods=2, column=column_name, input_range=config['input_range'], remainder= "drop")
            rbf_transform.fit(data)
            return ColumnTransformInfo(column_name=column_name, column_type='date', transform=rbf_transform, output_info=[SpanInfo(2, 'tanh')], output_dimensions=2)
            


    def fit(self, raw_data, discrete_columns=(), date_columns=()):
        """Fit the ``DataTransformer``.
        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns, and clock or rbf transformation for date columns 
        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            elif column_name in date_columns:
                column_transform_info = self._fit_date(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    # convert a continuous column of data to appropriate format for discriminator
    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output
    
    # convert a discrete column of data to appropriate format for discriminator
    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    @staticmethod
    def bulk_encode_time_value(val, max_val):
        x = np.sin(2 * np.pi / max_val * val)
        y = np.cos(2 * np.pi / max_val * val)
        return np.stack([x, y], axis=1)
    
    @staticmethod
    def clock_to_probs(pt, pts):
        EPS_CLOCKP = 0.01
        # Cast `pts` to the same dtype as `pt`
        pts_casted = tf.cast(tf.constant(pts), dtype=pt.dtype)
        ds = pts_casted - pt
        sq_ds = np.sum(tf.square(ds+EPS_CLOCKP), axis=1)
        raw_ps = 1/ sq_ds   
        
        return raw_ps / np.sum(raw_ps)

        
    def _transform_date(self, column_transform_info, data):
         
         if self.date_transformation == 'clock':
            column_name = data.columns[0]
            clock_dim = self.CLOCK_DIMS[column_name]
            return self.bulk_encode_time_value(data[column_name], clock_dim)
         elif self.date_transformation == 'rbf':
            rbf = column_transform_info.transform
            return rbf.transform(data)

       

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.
        Outputs a list with Numpy arrays.
        """
        
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            elif column_transform_info.column_type == 'date':
                column_data_list.append(self._transform_date(column_transform_info, data))
                if column_transform_info.transform is not None:
                   rbf = column_transform_info.transform
                   val = self.CLOCK_DIMS[column_name]
                   df = pd.DataFrame(np.arange(val), columns=[column_name])
                   transformed_array = rbf.transform(df)
                   self.RBF[column_name] = transformed_array
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))
            
        return column_data_list

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        column_data_list = self._synchronous_transform(raw_data,self._column_transform_info_list)
       
        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.
        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        
        return recovered_data

    def date_ps(self, oheIns, date_ps_raw):
        ohe_dummies = oheIns.dummies
        sorted_categories = np.sort(ohe_dummies)
        sort_index = np.argsort(ohe_dummies)
        date_ps_raw = date_ps_raw.numpy()[0]
        date_ps = date_ps_raw[sort_index]
        return date_ps

    def generate_raw_ps(self, data, LOG_AMOUNT_SCALE):
        sigmas=None
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_name == 'log_amount_sc':
                recovered_column_data = self._inverse_transform_continuous(column_transform_info, column_data,sigmas, st)
                recovered_column_data = recovered_column_data * LOG_AMOUNT_SCALE
                recovered_column_data = 10 ** recovered_column_data
                recovered_column_data = np.round(recovered_column_data - 1.0, 2)
                recovered_column_data_list.append(recovered_column_data)
                column_names.append(column_transform_info.column_name)

            elif  column_transform_info.column_name == 'tcode':  
                recovered_column_data = self._inverse_transform_discrete(column_transform_info, column_data)
                recovered_column_data_list.append(recovered_column_data)
                column_names.append(column_transform_info.column_name)

            elif column_transform_info.column_name == 'amount':
                recovered_column_data = self._inverse_transform_continuous(column_transform_info, column_data,sigmas, st)
                recovered_column_data_list.append(recovered_column_data)
                column_names.append(column_transform_info.column_name)

            elif column_transform_info.column_name == 'month': 
                if column_transform_info.column_type == 'discrete':
                    month_ps_raw = column_data
                    ohe = column_transform_info.transform
                    month_ps = self.date_ps(ohe, month_ps_raw)
                elif column_transform_info.transform is not None:
                    month_ps = self.clock_to_probs(column_data, self.RBF['month'])
                else:
                    month_ps = self.clock_to_probs(column_data, self.CLOCKS['month'])

            elif column_transform_info.column_name == 'day': 
                if column_transform_info.column_type == 'discrete':
                    day_ps_raw = column_data
                    ohe = column_transform_info.transform
                    day_ps = self.date_ps(ohe, day_ps_raw)
                elif column_transform_info.transform is not None:
                     day_ps = self.clock_to_probs(column_data, self.RBF['day'])
                else:
                    day_ps = self.clock_to_probs(column_data, self.CLOCKS['day'])

            elif column_transform_info.column_name == 'dow': 
                if column_transform_info.column_type == 'discrete':
                    dow_ps_raw = column_data
                    ohe = column_transform_info.transform
                    dow_ps = self.date_ps(ohe, dow_ps_raw)
                elif column_transform_info.transform is not None:
                    dow_ps = self.clock_to_probs(column_data, self.RBF['dow'])
                else:
                    dow_ps = self.clock_to_probs(column_data, self.CLOCKS['dow'])

            elif column_transform_info.column_name == 'dtme': 
                if column_transform_info.column_type == 'discrete':
                    dtme_ps_raw = column_data
                    ohe = column_transform_info.transform
                    dtme_ps = self.date_ps(ohe, dtme_ps_raw)
                elif column_transform_info.transform is not None:
                    dtme_ps = self.clock_to_probs(column_data, self.RBF['dtme'])
                else:
                    dtme_ps = self.clock_to_probs(column_data, self.CLOCKS['dtme'])

            elif column_transform_info.column_name == 'td':
                td_raw = column_data
                #gaussian_component = np.argmax(td_raw[:, 1:], axis=1)[0]
                recovered_column_data = self._inverse_transform_continuous(column_transform_info, column_data,sigmas, st)
                #recovered_column_data = np.round(recovered_column_data * TD_SCALE ).astype(int)
                recovered_column_data_list.append(recovered_column_data)
                column_names.append(column_transform_info.column_name)
                
            st += dim
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names).astype(self._column_raw_dtypes.loc[['log_amount_sc','tcode']]))

        return recovered_data, day_ps, dtme_ps, dow_ps, month_ps
        #return recovered_data
















    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }

    