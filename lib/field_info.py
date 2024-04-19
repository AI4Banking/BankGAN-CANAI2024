from abc import ABC, abstractmethod

class EncodingStrategy(ABC):
    """ cl - clock encoding (2d)
        oh - One-hot encoding
        raw - no encoding
        cl-i -  clock integer: transforms [1, 2, ..., n] -> [1, 2, ..., n-1, 0]"""
    
    @abstractmethod
    def get_input_encoding(self):
        pass

    @abstractmethod
    def get_target_encoding(self):
        pass

    @abstractmethod
    def get_net_encoding(self):
        pass
    
    
    def get_loss_type_activ(self, net_encodings):
        loss_types = {}
        activation_types = {}
        for field, encoding in net_encodings.items():
            if 'oh' in encoding:
                loss_types[field] = 'scce'    # 'scce': sparse categorical cross entropy``
            elif encoding == 'dist_cont':     # output is mean and std of a gaussian distribution
                loss_types[field] = 'pdf'
                activation_types[field] = 'relu'
            elif encoding == 'raw' or encoding == 'cl':
                loss_types[field] = 'mse'
            else:
                raise ValueError(f"Unknown encoding type for field {field}: {encoding}")
        return loss_types, activation_types
    

     
    
     

class BanksFormerEncoding(EncodingStrategy):

    """ input encoding of time-related informations is cyclical encoding with sine/cosine transformation
        output of time-related information is probabilities  """
    
    def get_input_encoding(self):
        return  {
            "day": "cl",
            "dtme": "cl",
            "dow": "cl",
            "month": "cl",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "oh_tcode"}
    
            # "k_symbol_num":"oh_symbol",
            # "operation_num":"oh_operation",
            # "type_num":"oh_type" }
        
    
    def get_target_encoding(self):
        return {
            "day": "cl-i",
            "dtme": "cl-i",
            "dow": "cl-i",
            "month": "cl-i",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "raw" }
    
            # "k_symbol_num":"raw",
            # "operation_num":"raw",
            # "type_num":"raw" }
        
    
    def get_net_encoding(self):
        return {
            "day": "oh_day",
            "dtme": "oh_dtme",
            "dow": "oh_dow",
            "month": "oh_month",
            "td_sc": "dist_cont",      #mean and std of continuous distribution
            "log_amount_sc": "dist_cont",
            "tcode_num": "oh_tcode"}
        
            # "k_symbol_num":"oh_symbol",
            # "operation_num":"oh_operation",
            # "type_num":"oh_type" }

        
    
class BanksFormerEncoding_v2(BanksFormerEncoding):
    def get_net_encoding(self):
        net_encoding = super().get_net_encoding()
        net_encoding["log_amount_sc"] = "raw"
        return net_encoding
    

class date_onehot_Encoding(EncodingStrategy):
      
      """ input encoding of time-related informations is one-hot encoding
        output of time-related information is probabilities  """
      
      def get_input_encoding(self):
        return  {
            "day": "oh_day",
            "dtme": "oh_dtme",
            "dow": "oh_dow",
            "month": "oh_month",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "oh_tcode" 
            }
      def get_target_encoding(self):
        return {
             "day": "cl-i",
            "dtme": "cl-i",
            "dow": "cl-i",
            "month": "cl-i",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "raw"
        }
    
      def get_net_encoding(self):
        return {
             "day": "oh_day",
            "dtme": "oh_dtme",
            "dow": "oh_dow",
            "month": "oh_month",
            "td_sc": "dist_cont",      #mean and std of continuous distribution
            "log_amount_sc": "dist_cont",
            "tcode_num": "oh_tcode"

        }
      


class date_rbf_Encoding(EncodingStrategy):
      
      """ input encoding of time-related informations is one-hot encoding
        output of time-related information is probabilities  """
      
      def get_input_encoding(self):
        return  {
            "day": "rbf",
            "dtme": "rbf",
            "dow": "rbf",
            "month": "rbf",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "oh_tcode" 
            }
      
      def get_target_encoding(self):
        return {
             "day": "cl-i",
            "dtme": "cl-i",
            "dow": "cl-i",
            "month": "cl-i",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "raw"
        }
    
      def get_net_encoding(self):
        return {
             "day": "oh_day",
            "dtme": "oh_dtme",
            "dow": "oh_dow",
            "month": "oh_month",
            "td_sc": "dist_cont",      #mean and std of continuous distribution
            "log_amount_sc": "dist_cont",
            "tcode_num": "oh_tcode"

        }
      
class date_clock_Encoding(EncodingStrategy):
    """date features both in input and output are in two-dimensional format"""
    def get_input_encoding(self):
        return  {
            "day": "cl",
            "dtme": "cl",
            "dow": "cl",
            "month": "cl",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "oh_tcode" 
            }
    
    def get_target_encoding(self):
        return {
             "day": "cl",
            "dtme": "cl",
            "dow": "cl",
            "month": "cl",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "raw"
        }
    
    def get_net_encoding(self):
        return {
             "day": "cl",
            "dtme": "cl",
            "dow": "cl",
            "month": "cl",
            "td_sc": "dist_cont",      #mean and std of continuous distribution
            "log_amount_sc": "dist_cont",
            "tcode_num": "oh_tcode"

        }
    
class  date_clock_encoding_v2(date_clock_Encoding):
    def get_net_encoding(self):
        net_encoding = super().get_net_encoding()
        net_encoding["log_amount_sc"] = "dist_cont"
        return net_encoding
    
    

class StrategyFactory:
    @staticmethod
    def get_strategy(scenario):
        if scenario == 'banksformer':
            return BanksFormerEncoding()
        elif scenario == "banksformer_v2":
            return BanksFormerEncoding_v2()
        elif scenario == 'dateonehot':
            return date_onehot_Encoding()
        elif scenario == 'dateclock':
            return date_clock_Encoding()
        elif scenario == 'daterbf':
            return date_rbf_Encoding()
        elif scenario == 'dateclock_v2':
            return date_clock_encoding_v2()
        
class FIELD_INFO_TCODE():
      def __init__(self):
         
         self.CAT_FIELDS = ['tcode_num']
         self.DATA_KEY_ORDER = self.CAT_FIELDS

         self.INP_ENCODINGS =  {"tcode_num": "oh_tcode"}
         self.TAR_ENCODINGS = {"tcode_num": "raw"}
         self.NET_ENCODINGS = {"tcode_num": "oh_tcode"}

         self.LOSS_TYPES = {"tcode_num": "scce"}

         self.ACTIVATIONS = {"tcode_num": None}

         self.FIELD_DIMS_IN  = {"tcode_num": 16}
         self.FIELD_DIMS_TAR = {"tcode_num" : 1}
         self.FIELD_DIMS_NET = {"tcode_num":16}
         self.FIELD_STARTS_IN = {"tcode_num": 0}
         self.FIELD_STARTS_NET = {"tcode_num": 0}
         self.FIELD_STARTS_TAR = {"tcode_num": 0}

class FIELD_INFO_CATFIELD:
    def __init__(self):
        self.CAT_FIELDS = [ 'k_symbol_num',  'operation_num', 'type_num']
        self.DATA_KEY_ORDER = self.CAT_FIELDS

        self.INP_ENCODINGS =   {"k_symbol_num":"oh_symbol",
                            "operation_num":"oh_operation",
                              "type_num":"oh_type" }
        
        self.TAR_ENCODINGS = {"k_symbol_num":"raw",
                              "operation_num":"raw",
                              "type_num":"raw" }
        
        self.NET_ENCODINGS = { "k_symbol_num":"oh_symbol",
                               "operation_num":"oh_operation",
                              "type_num":"oh_type" }
        
        self.LOSS_TYPES =  {"k_symbol_num":"scce",
                               "operation_num":"scce",
                                 "type_num": "scce"}
        
        self.ACTIVATIONS = {"k_symbol_num": None,
                               "operation_num":None,
                              "type_num":None }
        
        self.FIELD_DIMS_IN, self.FIELD_DIMS_TAR, self.FIELD_DIMS_NET, self.FIELD_STARTS_IN, self.FIELD_STARTS_TAR, self.FIELD_STARTS_NET= self._get_field_dims_and_starts()

    def _get_field_dims_and_starts(self):
        ENCODING_DIMS_BY_TYPE = {'oh_type':2,'oh_operation':6, 'oh_symbol':9, 'raw':1 }
        FIELD_DIMS_IN  = {}
        FIELD_DIMS_TAR = {}
        FIELD_DIMS_NET = {}

        for k in self.DATA_KEY_ORDER:
            FIELD_DIMS_IN[k] = ENCODING_DIMS_BY_TYPE[self.INP_ENCODINGS[k]]
            FIELD_DIMS_TAR[k] = ENCODING_DIMS_BY_TYPE[self.TAR_ENCODINGS[k]]
            FIELD_DIMS_NET[k] = ENCODING_DIMS_BY_TYPE[self.NET_ENCODINGS[k]]

        FIELD_STARTS_IN = self._compute_field_starts(FIELD_DIMS_IN)
        FIELD_STARTS_TAR = self._compute_field_starts(FIELD_DIMS_TAR)
        FIELD_STARTS_NET = self._compute_field_starts(FIELD_DIMS_NET)

            
        return FIELD_DIMS_IN, FIELD_DIMS_TAR, FIELD_DIMS_NET, FIELD_STARTS_IN, FIELD_STARTS_TAR, FIELD_STARTS_NET
        
    def _compute_field_starts(self, field_dims):
        field_starts = {}
        start = 0
        for k in self.DATA_KEY_ORDER:
            field_starts[k] = start
            start += field_dims[k]
        return field_starts

class FieldInfo_type2:
    def __init__(self):

        self.DATA_KEY_ORDER = ['tcode_num', 'td_sc', 'log_amount_sc']    
        self.INP_ENCODINGS = { "td_sc": "raw","log_amount_sc": "raw","tcode_num": "oh_tcode" }    
        self.TAR_ENCODINGS ={"td_sc": "raw","log_amount_sc": "raw", "tcode_num": "raw" }
        self.NET_ENCODINGS = {"td_sc": "dist_cont", "log_amount_sc": "dist_cont","tcode_num": "oh_tcode"}
        self.LOSS_TYPES = {"td_sc": "pdf", "log_amount_sc": "pdf","tcode_num": "scce"}
        self.ACTIVATIONS =  {"td_sc": "relu", "log_amount_sc": "relu","tcode_num": None}
        self.FIELD_DIMS_IN, self.FIELD_DIMS_TAR, self.FIELD_DIMS_NET, self.FIELD_STARTS_IN, self.FIELD_STARTS_TAR, self.FIELD_STARTS_NET= self._get_field_dims_and_starts()

    def _get_field_dims_and_starts(self):
        ENCODING_DIMS_BY_TYPE = {'oh_tcode':16,'raw':1, 'dist_cont':2 }
        FIELD_DIMS_IN  = {}
        FIELD_DIMS_TAR = {}
        FIELD_DIMS_NET = {}

        for k in self.DATA_KEY_ORDER:
            FIELD_DIMS_IN[k] = ENCODING_DIMS_BY_TYPE[self.INP_ENCODINGS[k]]
            FIELD_DIMS_TAR[k] = ENCODING_DIMS_BY_TYPE[self.TAR_ENCODINGS[k]]
            FIELD_DIMS_NET[k] = ENCODING_DIMS_BY_TYPE[self.NET_ENCODINGS[k]]

        FIELD_STARTS_IN = self._compute_field_starts(FIELD_DIMS_IN)
        FIELD_STARTS_TAR = self._compute_field_starts(FIELD_DIMS_TAR)
        FIELD_STARTS_NET = self._compute_field_starts(FIELD_DIMS_NET)

            
        return FIELD_DIMS_IN, FIELD_DIMS_TAR, FIELD_DIMS_NET, FIELD_STARTS_IN, FIELD_STARTS_TAR, FIELD_STARTS_NET
    
    def _compute_field_starts(self, field_dims):
        field_starts = {}
        start = 0
        for k in self.DATA_KEY_ORDER:
            field_starts[k] = start
            start += field_dims[k]
        return field_starts
        
        
class FieldInfo:
    def __init__(self, scenario):
         
        #self.CAT_FIELDS = [ 'k_symbol_num',  'operation_num', 'type_num']
        self.CAT_FIELDS = ['tcode_num']
        self.DATA_KEY_ORDER = self.CAT_FIELDS + ['dow', 'month', "day", 'dtme', 'td_sc', 'log_amount_sc']
        self.CLOCK_DIMS = {"day": 31,
                "dtme": 31,
                "dow": 7,
                "month": 12}

        encoding_strategy = StrategyFactory.get_strategy(scenario)
        
        self.INP_ENCODINGS = encoding_strategy.get_input_encoding()
        self.TAR_ENCODINGS = encoding_strategy.get_target_encoding()
        self.NET_ENCODINGS = encoding_strategy.get_net_encoding()
        

        self.LOSS_TYPES, self.ACTIVATIONS = encoding_strategy.get_loss_type_activ(self.NET_ENCODINGS)

        self.FIELD_DIMS_IN, self.FIELD_DIMS_TAR, self.FIELD_DIMS_NET, self.FIELD_STARTS_IN, self.FIELD_STARTS_TAR, self.FIELD_STARTS_NET= self._get_field_dims_and_starts()

    def _get_field_dims_and_starts(self):
        ENCODING_DIMS_BY_TYPE = {'cl-i': 1, 
                                     'raw': 1,
                                      'rbf':3,
                                      'cl': 2,
                                      'dist_cont':2,
                                      'oh_tcode':16, 'oh_type':2,'oh_operation':6, 'oh_symbol':9 ,'oh_day':31, 'oh_month':12, 'oh_dtme':31, 'oh_dow': 7}
        FIELD_DIMS_IN  = {}
        FIELD_DIMS_TAR = {}
        FIELD_DIMS_NET = {}

        for k in self.DATA_KEY_ORDER:
            FIELD_DIMS_IN[k] = ENCODING_DIMS_BY_TYPE[self.INP_ENCODINGS[k]]
            FIELD_DIMS_TAR[k] = ENCODING_DIMS_BY_TYPE[self.TAR_ENCODINGS[k]]
            FIELD_DIMS_NET[k] = ENCODING_DIMS_BY_TYPE[self.NET_ENCODINGS[k]]

        FIELD_STARTS_IN = self._compute_field_starts(FIELD_DIMS_IN)
        FIELD_STARTS_TAR = self._compute_field_starts(FIELD_DIMS_TAR)
        FIELD_STARTS_NET = self._compute_field_starts(FIELD_DIMS_NET)

            
        return FIELD_DIMS_IN, FIELD_DIMS_TAR, FIELD_DIMS_NET, FIELD_STARTS_IN, FIELD_STARTS_TAR, FIELD_STARTS_NET
    
    def _compute_field_starts(self, field_dims):
        field_starts = {}
        start = 0
        for k in self.DATA_KEY_ORDER:
            field_starts[k] = start
            start += field_dims[k]
        return field_starts