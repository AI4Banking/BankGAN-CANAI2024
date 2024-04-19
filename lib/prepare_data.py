
from datetime import datetime
import calendar
import numpy as np
import pandas as pd
import random
import os

# random.seed(0)
# np.random.seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


def preprocess_data_czech(df):
    #df = pd.read_csv('tr_by_acct_w_age.csv')

    czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
    df["datetime"] = df["date"].apply(czech_date_parser)
    #df["datetime"] = pd.to_datetime(df["datetime"])

    df["month"] = df["datetime"].dt.month % 12
    df["day"] = df["datetime"].dt.day % 31
    df["dow"] =  df["datetime"].dt.dayofweek % 7
    df["year"] = df["datetime"].dt.year
    
    df["td"] = df[["account_id", "datetime"]].groupby("account_id").diff()
    df["td"] = df["td"].apply(lambda x: x.days)
    df["td"].fillna(0.0, inplace=True)
    

    # dtme - days till month end
    df["dtme"] = df.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day) % 31

    df['raw_amount'] = df.apply(lambda row: row['amount'] if row['type'] == 'CREDIT' else -row['amount'], axis=1)


    cat_code_fields = ['type', 'operation', 'k_symbol']
    TCODE_SEP = "__"
    # create tcode by concating fields in "cat_code_fields"
    tcode = df[cat_code_fields[0]].astype(str)
    for ccf in cat_code_fields[1:]:
        tcode += TCODE_SEP + df[ccf].astype(str)

    df["tcode"] = tcode

    ATTR_SCALE = df["age"].std()
    df["age_sc"] = df["age"] / ATTR_SCALE

    df["log_amount"] = np.log10(df["amount"]+1)
    LOG_AMOUNT_SCALE = df["log_amount"].std()
    df["log_amount_sc"] = df["log_amount"] / LOG_AMOUNT_SCALE
        
    TD_SCALE = df["td"].std()
    df["td_sc"] = df["td"] / TD_SCALE

    cat_fields = ['type', 'operation', 'k_symbol', 'tcode']

    field_mappings = {}
    for field in cat_fields:
        
        # Create the category to number mapping
        cat_to_num = dict([(tc, i) for i, tc in enumerate(df[field].unique())])
        
        # Store the mappings in the field_mappings dictionary
        field_mappings[f"{field}_to_num".upper()] = cat_to_num
        field_mappings[f"num_to_{field}".upper()] = dict([(i, tc) for i, tc in enumerate(df[field].unique())])

        df[field + "_num"] = df[field].apply(lambda x: cat_to_num[x])
        
        # add '_' to nan and blank so they are always interpreted as strings
        df[field] = df[field].astype(str).apply(lambda x: "_" + x  if x in ["nan", ""]  else x)

    START_DATE = df["datetime"].min()
    return df, LOG_AMOUNT_SCALE, TD_SCALE, ATTR_SCALE, START_DATE, field_mappings