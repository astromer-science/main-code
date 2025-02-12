import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import polars as pl
import numpy as np
import logging
import shutil
import random
import glob
import math
import toml
import os

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
from typing import List, Dict, Any
from io import BytesIO
from tqdm import tqdm

# Set up logging configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_dtype(value, data_type):
    if type(value) in [int, float] and data_type == 'integer':
        return _int64_feature([int(value)])
    if type(value) in [int, float] and data_type == 'float':
        return _float_feature([value])
    if type(value) == str and data_type == 'string':
        return _bytes_feature([str(value).encode()])

    if type(value) == list:
        if type(value[0]) == int and data_type == 'integer':
            return _int64_feature(value)
        if type(value[0]) == float and data_type == 'float':
            return _float_feature(value)
        if type(value[0]) == str and data_type == 'string':
            return _bytes_feature(value)
    
    raise ValueError('[ERROR] {} with type {}/{} could not be parsed. Please use <str>, <int>, or <float>'.format(value, type(value), data_type))

def substract_frames(frame1, frame2, on):
    frame1 = frame1[~frame1[on].isin(frame2[on])]
    return frame1

def write_config(context_features: List[str], sequential_features: List[str], config_path: str) -> None:
    """
    Writes the configuration to a toml file.

    Args:
        context_features (list): List of context features.
        sequential_features (list): List of sequential features.
        config_path (str): Path to the output config.toml file.
    """
    config = {
        "context_features": context_features,
        "sequential_features": sequential_features
    }

    # Make directory if it does not exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    try:
        with open(config_path, 'w') as config_file:
            toml.dump(config, config_file)
        logging.info(f'Successfully wrote config to {config_path}')
    except Exception as e:
        logging.error(f'Error while writing the config file: {str(e)}')
        raise e

def create_config_toml(parquet_id='newID', 
                       target='./data/precords/test', 
                       context_features=['ID', 'Label'],
                       sequential_features=['mjd', 'mag', 'err'],
                       context_dtypes=['string', 'integer'],
                       sequential_dtypes=['float', 'float', 'float']):
    
    os.makedirs(target, exist_ok=True)
    config_dict = {
        'id_column':{
            'value': parquet_id,
            'dtype': 'integer'
        },
        'target':{
            'value': target,
            'dtype': 'string'
        },
        'context_features': {
            'value': context_features,
            'dtypes': context_dtypes, 
        },
        'sequential_features':{
            'value': sequential_features,
            'dtypes': sequential_dtypes,
        }

    }
    with open(os.path.join(target, 'config.toml'), 'w') as handle:
        toml.dump(config_dict, handle)
    print('[INFO] Toml file succefully created at: {}'.format(target))
    
class DataPipeline:
    """
    Args:
        metadata
        config_path
    """

    def __init__(self,
                 metadata=None,
                 config_path= "./config.toml"):

        

        #get context and sequential features from config file 
        if not os.path.isfile(config_path):
            logging.error("The specified config path does not exist")
            raise FileNotFoundError("The specified config path does not exist")

        # Read the config file
        with open(config_path, 'r') as f:
            config = toml.load(f)

        # Saving class variables
        self.metadata                  = metadata
        self.config_path               = config_path
        self.config                    = config
        self.context_features          = config['context_features']['value']
        self.context_features_dtype    = config['context_features']['dtypes']
        self.sequential_features       = config['sequential_features']['value']
        self.sequential_features_dtype = config['sequential_features']['dtypes']
        self.output_folder             = config['target']['path']
        self.obs_path                  = config['sequential_features']['path']
        self.id_column                 = config['id_column']['value']

        self.check_dtypes()        
        assert self.metadata[self.id_column].dtype == int, \
        'ID column should be an integer Serie but {} was given'.format(self.metadata[self.id_column].dtype)
        
        if metadata is not None:
            print('[INFO] {} samples loaded'.format(metadata.shape[0]))

#         self.metadata['subset_0'] = ['full']*self.metadata.shape[0]

        os.makedirs(self.output_folder, exist_ok=True)

    def check_dtypes(self):
        sample = pd.read_parquet(
            os.path.join(self.config['sequential_features']['path'], 'shard_000.parquet'))
                
        a = [self.context_features, self.sequential_features]
        b = [self.context_features_dtype, self.sequential_features_dtype]
        c = ['context', 'sequence']
        d = [self.metadata, sample]
        
        for features, fdtype, name, df in zip(a, b, c, d):
            partial = []
            for key in features:
                if df[key].dtype in [object, 'string']:
                    partial.append('string')

                if np.issubdtype(df[key].dtype, np.floating):
                    partial.append('float')

                if np.issubdtype(df[key].dtype, np.integer):
                    partial.append('integer')

            if partial != fdtype:
                print('[WARN] Inconsistent data types in {}. Overwritting config...'.format(name))
                if name == 'context':
                    self.context_features_dtype = partial
                    self.config['context_features']['dtypes'] = partial
                                    
                if name == 'sequence':
                    self.sequential_features_dtype = partial
                    self.config['sequential_features']['dtypes'] = partial

        with open(self.config_path, 'w') as f:
            toml.dump(self.config, f)
        
    @staticmethod
    def aux_serialize(sel : pl.DataFrame, 
                      path : str, 
                      context_features: list, 
                      context_features_dtype: list,
                      sequential_features: list,
                      sequential_features_dtype: list) -> None:
        if not isinstance(sel, pl.DataFrame):
            logging.error("Invalid data type provided to aux_serialize")
            raise ValueError("Invalid data type provided to aux_serialize")

        with tf.io.TFRecordWriter(path) as writer:
            for row  in sel.iter_rows(named=True):
                ex = DataPipeline.get_example(row, context_features, context_features_dtype, 
                                              sequential_features, sequential_features_dtype)
                writer.write(ex.SerializeToString())
         
        
    @staticmethod
    def get_example(row: dict, 
                    context_features: list, 
                    context_features_dtype: list,
                    sequential_features: list,
                    sequential_features_dtype: list) -> tf.train.SequenceExample:
        """
        Converts a given row into a TensorFlow SequenceExample.

        Args:
            row (pd.Series): Row of data to be converted.

        Returns:
            tf.train.SequenceExample: The converted row as a SequenceExample.
        """
        dict_features = {}
        # Parse each context feature based on its dtype and add to the features dictionary
        for name, data_type in zip(context_features, context_features_dtype):
            dict_features[name] = parse_dtype(row[name], data_type=data_type)

        # Create a context for the SequenceExample using the features dictionary
        element_context = tf.train.Features(feature=dict_features)

        dict_sequence = {}
        # Create a sequence of features for each dimension of the lightcurve
        for col, data_type in zip(sequential_features, sequential_features_dtype):
            seqfeat = parse_dtype(row[col][:], data_type=data_type)
            seqfeat = tf.train.FeatureList(feature=[seqfeat])
            dict_sequence[col] = seqfeat

        # Add the sequence to the SequenceExample
        element_lists = tf.train.FeatureLists(feature_list=dict_sequence)

        # Create the SequenceExample
        ex = tf.train.SequenceExample(context=element_context, feature_lists=element_lists)
        # logging.info("Successfully converted to SequenceExample.")
        return ex
    
    def inspect_records(self, dir_path:str = './records/output/', num_records: int = 1):
        """
        Function to inspect the first 'num_records' from a random TFRecord file in the given directory.

        Args:
            dir_path (str): Directory path where TFRecord files are located.
            num_records (int): Number of records to inspect.

        Returns:
            NoReturn
        """
        # Use glob to get all the .record files in the directory
        file_paths = glob.glob(dir_path + '*.record')

        # Select a random file path
        file_path = random.choice(file_paths)

        try:
            raw_dataset = tf.data.TFRecordDataset(file_path)
            for raw_record in raw_dataset.take(num_records):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

            logging.info(f'Successfully inspected {num_records} records from {file_path}.')
        except Exception as e:
            logging.error(f'Error while inspecting records. Error message: {str(e)}')
            raise e


    def prepare_data(self, container : np.ndarray, elements_per_shard : int, subset : str, fold_n : int):
        """Prepare the data to be saved as records"""
        if container is None or not hasattr(container, '__iter__'):
            raise ValueError("Invalid container provided to prepare_data")
        
        # Number of objects in the split
        N = len(container)
        # Compute the number of shards
        n_shards = math.ceil(N/elements_per_shard)

        # Number of characters of the padded number of shards
        name_length = len(str(n_shards))

        # Create fold's directory
        root = os.path.join(self.output_folder, 'fold_'+str(fold_n), subset)
        os.makedirs(root, exist_ok=True)

        # Keep config file in the records folder to desirialize later
        shutil.copyfile(self.config_path, os.path.join(root, 'config.toml'))

        # Create one file per shard
        shard_paths = []
        shards_data = []
        for shard in range(n_shards):
            # Get the shard number padded with 0s
            shard_name = str(shard).rjust(name_length, '0')
            # Get the shard store name
            shard_path= os.path.join(root, '{}.record'.format(shard_name))
            # Get shard observations
            sel = container[shard*elements_per_shard:(shard+1)*elements_per_shard]
            # Save it into a list
            shard_paths.append(shard_path)
            shards_data.append(sel) 

        return shards_data, shard_paths
    
    def lightcurve_step(self, inputs):
        """
        Preprocessing applied to each light curve separately
        """
        # First feature is time
        inputs = inputs.sort(self.sequential_features[0])    
        return inputs

    def observations_step(self):
        """
        Preprocessing applied to all observations. Filter only
        """
        fn_0 = pl.col("errmag") < 1.  # Clean the data on the big lazy dataframe
        fn_1 = pl.col("errmag") > 0.  
        return fn_0 & fn_1

    def read_all_parquets(self, observations_path : str) -> pd.DataFrame:
        """
        Read the files from given paths and filters it based on err_threshold and ID from metadata
        Args:
            observations_path (str): Directory path of parquet files
        Returns:
            new_df (pl.DataFrame): Processed dataframe
        """
        # logging.info("Reading parquet files")

        if not os.path.exists(observations_path):
            logging.error("The specified parquets path does not exist")
            raise FileNotFoundError("The specified parquets path does not exist")


        # Read the parquet filez lazily
        paths = ['{}/shard_{}.parquet'.format(observations_path, 
                                              str(s).rjust(3, '0')) for s in self.metadata['shard'].unique()]

        scan = pl.scan_parquet(paths)
                
        # Using partial information, extract only the necessary objects
        ID_series = pl.Series(self.metadata[self.id_column].values)
        f1 = pl.col(self.id_column).is_in(ID_series)
        scan.filter(f1)
        
        lightcurves_fn  = lambda light_curve: self.lightcurve_step(light_curve)

        # Filter, drop nulls, and sort every object
        processed_obs = scan.filter(self.observations_step()).drop_nulls().groupby(self.id_column).apply(lightcurves_fn, schema=None)

        # Select only the relevant columns
        processed_obs = processed_obs.select([self.id_column] + self.sequential_features)

        # Mix metadata and the data
        processed_obs = processed_obs.groupby(self.id_column).all()

        # First run takes more time!
        # metadata_lazy = pl.scan_parquet(metadata_path, cache=True) # First run is slower
        metadata_lazy = pl.from_pandas(self.metadata).lazy()   
        # Perform the join to get the data
        processed_obs = processed_obs.join(other=metadata_lazy, 
                                            on=self.id_column).collect(streaming=False) #streaming might be useless.                    
        
        return processed_obs
    


    def resample_folds(self, n_folds=1):
        print('[INFO] Creating {} random folds'.format(n_folds))
        print('Not implemented yet hehehe...')

    def run(self,  n_jobs : int =1, elements_per_shard : int = 5000) -> None: 
        """
        Executes the DataPipeline operations which includes reading parquet files, processing samples and writing records.
        
        Args:
            n_jobs (int): The maximum number of concurrently running jobs. Default is 1
            elements_per_shard (int): Maximum number of elements per shard. Default is 5000
        """
        if not os.path.exists(self.obs_path):
            logging.error("The specified parquets path does not exist")
            raise FileNotFoundError("The specified parquets path does not exist")
          
        # Start the operations
        logging.info("Starting DataPipeline operations")

        # threads = Parallel(n_jobs=n_jobs, backend='threading')
        fold_groups = [x for x in self.metadata.columns if 'subset' in x]
        
        
        print('[INFO] Reading parquet')
        new_df = self.read_all_parquets(self.obs_path)
        print('[INFO] Light curves loaded')
        self.new_df = new_df

        for fold_n, fold_col in enumerate(fold_groups):
            pbar = tqdm(self.metadata[fold_col].dropna().unique(), colour='#00ff00') # progress bar
            for subset in pbar:               
                pbar.set_description(f"Processing fold {fold_n+1}/{len(fold_groups)} - {subset}")
                # ============ Processing Samples ===========
                partial = self.metadata[self.metadata[fold_col] == subset]
                
                # Transform into a appropiate representation
                index = partial[self.id_column]
                b = np.isin(new_df[self.id_column].to_numpy(), index)
                container = new_df.filter(b)
                
                # ============ Writing Records ===========                
                shards_data, shard_paths = self.prepare_data(container, elements_per_shard, subset, fold_n)
                
                for shard, shard_path in zip(shards_data,shard_paths):
                    DataPipeline.aux_serialize(shard, shard_path, 
                                    self.context_features, self.context_features_dtype, 
                                    self.sequential_features, self.sequential_features_dtype)

                # with ThreadPoolExecutor(n_jobs) as exe:
                #     # submit tasks to generate files
                #     _ = [exe.submit(DataPipeline.aux_serialize, shard, shard_path, 
                #                     self.context_features, self.context_features_dtype, 
                #                     self.sequential_features, self.sequential_features_dtype) \
                #              for shard, shard_path in zip(shards_data, shard_paths)]
    

        logging.info('Finished execution of DataPipeline operations')




def get_tf_dtype(data_type, is_sequence=False):
    if not is_sequence:
        if data_type == 'integer': return tf.io.FixedLenFeature([], dtype=tf.int64)
        if data_type == 'float': return tf.io.FixedLenFeature([], dtype=tf.float32)
        if data_type == 'string': return tf.io.FixedLenFeature([], dtype=tf.string)
    else:
        if data_type == 'integer': return tf.io.VarLenFeature(dtype=tf.int64)
        if data_type == 'float': return tf.io.VarLenFeature(dtype=tf.float32)
        if data_type == 'string': return tf.io.VarLenFeature(dtype=tf.string32)

def deserialize(sample, records_path=None):
    """
    Reads a serialized sample and converts it to tensor.
    Context and sequence features should match the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    try:
        with open(os.path.join(records_path, 'config.toml'), 'r') as f:
            config = toml.load(f)
    except:
        with open('./data/config_v0.toml', 'r') as f:
            config = toml.load(f)

    # Define context features as strings
    context_features = {}
    for feat, data_type in zip(config['context_features']['value'], config['context_features']['dtypes']):
        context_features[feat]= get_tf_dtype(data_type)

    sequence_features = {}
    # Define sequence features as floating point numbers
    for feat, data_type in zip(config['sequential_features']['value'], config['sequential_features']['dtypes']):
        sequence_features[feat]= get_tf_dtype(data_type, is_sequence=True)

    # Parse the serialized sample into context and sequence features
    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )
    
    # Cast context features to strings
    input_dict = {k: context[k] for k in config['context_features']['value']}

    # Cast and store sequence features
    casted_inp_parameters = []
    for k in config['sequential_features']['value']:
        seq_dim = sequence[k]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    # Add sequence to the input dictionary
    input_dict['input'] = tf.stack(casted_inp_parameters, axis=2)[0]
    
    try:
        input_dict['length'] = tf.shape(input_dict['input'])[0]
        # Compatibility with keys in the following code
        input_dict['lcid'] = input_dict.pop('ID')
        input_dict['label'] = input_dict.pop('Label')
    except Exception as e:
        input_dict['lcid'] = input_dict.pop('id')
        
    try:    
        del input_dict['Class']
        del input_dict['Band']
    except:
        pass
        
    return input_dict