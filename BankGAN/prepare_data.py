
from datetime import datetime
import calendar
import numpy as np
import pandas as pd
import random
import os
import numpy as np
import tensorflow as tf

# Set seeds
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def preprocess_data_czech(df):
    #df = pd.read_csv('tr_by_acct_w_age.csv')

    czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
    df["datetime"] = df["date"].apply(czech_date_parser)
    #df["datetime"] = pd.to_datetime(df["datetime"])

    df["month"] = df["datetime"].dt.month 
    df["day"] = df["datetime"].dt.day 
    df["dow"] =  df["datetime"].dt.dayofweek 
    df["year"] = df["datetime"].dt.year
    df["doy"] = df["datetime"].dt.dayofyear
    
    df["td"] = df[["account_id", "datetime"]].groupby("account_id").diff()
    df["td"] = df["td"].apply(lambda x: x.days)
    df["td"].fillna(0.0, inplace=True)
    

    # dtme - days till month end
    df["dtme"] = df.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day)

    df['raw_amount'] = df.apply(lambda row: row['amount'] if row['type'] == 'CREDIT' else -row['amount'], axis=1)


    cat_code_fields = ['type', 'operation', 'k_symbol']
    TCODE_SEP = "__"
    # create tcode by concating fields in "cat_code_fields"
    tcode = df[cat_code_fields[0]].astype(str)
    for ccf in cat_code_fields[1:]:
        tcode += TCODE_SEP + df[ccf].astype(str)

    df["tcode"] = tcode

    conditions = [
    (df['day'] >= 1) & (df['day'] <= 10),
    (df['day'] > 10) & (df['day'] <= 20),
    (df['day'] > 20) & (df['day'] <= 31)
      ]
      
    categories = ['first', 'middle', 'last']

    # Use numpy.select() to map the numbers to categories
    df['DoM_cat'] = np.select(conditions, categories)

    bin_edges = [17, 30, 40, 50, 60, 81]
    labels = ['18-30', '31-40', '41-50', '51-60', '61+']

    # Use pd.cut() to convert ages to categorical groups
    df['age_group'] = pd.cut(df['age'], bins=bin_edges, labels=labels, right=False)
    df['age_group'] = df['age_group'].astype('object')

    result = df.groupby('account_id')['datetime'].agg(['min', 'max'])
    result['duration'] = result['max'] - result['min']
    result_sorted = result.sort_values('duration', ascending=False)

    df['log_amount'] = np.log10(df['amount'] + 1)
    LOG_AMOUNT_SCALE = df['log_amount'].std()
    df['log_amount_sc'] = df['log_amount']/ LOG_AMOUNT_SCALE
    TD_SCALE = df['td'].std()
    df['td_sc'] = df['td']/TD_SCALE

    return df, LOG_AMOUNT_SCALE , TD_SCALE