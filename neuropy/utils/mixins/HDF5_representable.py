from functools import wraps, partial
from pathlib import Path
from copy import deepcopy
from typing import Sequence, Union, Type, Optional, List, Dict, Callable
from attrs import define, field, Factory
import numpy as np
import pandas as pd
import h5py # for to_hdf and read_hdf definitions
from datetime import datetime, timedelta
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define
import attrs
from attrs import field, asdict, fields
from neuropy.utils.result_context import IdentifyingContext

# ==================================================================================================================== #
# 2023-07-30 HDF5 General Object Serialization Classes                                                                 #
# ==================================================================================================================== #
""" Imports:

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

        
"""



@define(slots=False, repr=False)
class HDFSerializationRegister:
	""" 2024-01-10 - A dramatically simplified HDF serialization type handler that avoids all of the hard crap and just allows users to register conversion functions directly for items of different types
	
	a_register = HDFSerializationRegister()

	a_register.converion_registery[pd.DataFrame] = lambda x, *hdf_args, **hdf_kwargs: x.to_hdf(*hdf_args, **hdf_kwargs)
	a_register.converion_registery[Epoch] = lambda x, *hdf_args, **hdf_kwargs: x.to_dataframe().to_hdf(*hdf_args, **hdf_kwargs)


	# works!
	a_register.to_hdf(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.filter_epochs, 'output/all_directional_laps_filter_epochs_decoder_result-filter_epochs.hdf', 'filter_epochs')


	"""
	converion_registery: Dict[Type, Callable] = field(default=Factory(dict))

	def to_hdf(self, v, *hdf_args, **hdf_kwargs):
		found_conversion_fn = self.converion_registery.get(type(v), None)
		if found_conversion_fn is None:
			print(f'could not find conversion function for v of type {type(v)} in registery.')
			return None
		
		return found_conversion_fn(v, *hdf_args, **hdf_kwargs) # call the function with value directly
		# return found_conversion_fn





# Deserialization ____________________________________________________________________________________________________ #

def post_deserialize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._is_post_deserialize = True
    return wrapper


class HDF_DeserializationMixin(AttrsBasedClassHelperMixin):
    def deserialize(self, *args, **kwargs):
        # Your deserialization logic here
        
        # Call post-deserialization methods
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_is_post_deserialize'):
                attr()


    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs):
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = cls.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        raise NotImplementedError # implementor must override!

        # # Read the DataFrame using pandas
        # pos_df = pd.read_hdf(file_path, key=key)

        # # Open the file with h5py to read attributes
        # with h5py.File(file_path, 'r') as f:
        #     dataset = f[key]
        #     metadata = {
        #         'time_variable_name': dataset.attrs['time_variable_name'],
        #         'sampling_rate': dataset.attrs['sampling_rate'],
        #         't_start': dataset.attrs['t_start'],
        #         't_stop': dataset.attrs['t_stop'],
        #     }

        # # Reconstruct the object using the class constructor
        # _out = cls(pos_df=pos_df, metadata=metadata)
        # _out.filename = file_path # set the filename it was loaded from
        # return _out



""" Usage of DeserializationMixin

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDF_Converter, HDFMixin


class MyClass(DeserializationMixin):
    def __init__(self, value):
        self.value = value

    @post_deserialize
    def setup(self):
        print(f"Post-deserialization setup for value: {self.value}")

        
"""


# ==================================================================================================================== #
# HDF Conversion Helpers                                                                                               #
# ==================================================================================================================== #



class HDFConvertableEnum:
    """ indicates conformers should be converted to an HDF-enumeration if they are used as a dataframe column. 
    
    from neuropy.utils.mixins.HDF5_representable import HDFConvertableEnum
    
    TODO: see notes on
    
    NESTED TYPES CAN'T BE PICKED, so I"m moving this out
    
    .apply(lambda x: NeuronType.from_hdf_coding_string(x)).astype(object) #.astype(str) # interestingly this leaves the dtype of this column as 'category' still, but _spikes_df["neuron_type"].to_numpy() returns the correct array of objects... this is better than it started before saving, but not the same. 
    
    """
    @classmethod
    def get_pandas_categories_type(cls) -> Type:
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def convert_to_hdf(cls, value) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_hdf_coding_string(cls, string_value: str) -> "HDFConvertableEnum":
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def convert_dataframe_columns_for_hdf(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ Convert any Enum-typed dataframe columns to the HDF5-compatible categorical type if needed 
        
        NOTE: works to fix errors with custom enum-type columns in DataWrangler too.
            As an alternative to:
                plot_neuron_replay_stats_df['track_membership'] = pd.Categorical(plot_neuron_replay_stats_df['track_membership'].map(lambda x: SplitPartitionMembership.convert_to_hdf(x)), ordered=True) # Fixes that column for DataWrangler
                plot_neuron_replay_stats_df['neuron_type'] = pd.Categorical(plot_neuron_replay_stats_df['neuron_type'].map(lambda x: NeuronType.convert_to_hdf(x)), ordered=True)

                       
        Usage:
            from neuropy.utils.mixins.HDF5_representable import HDFConvertableEnum
    
            plot_neuron_replay_stats_df = HDFConvertableEnum.convert_dataframe_columns_for_hdf(plot_neuron_replay_stats_df)
            plot_neuron_replay_stats_df
            
        """
        # [col for col in df.columns if issubclass(type(df[col].iloc[0]), HDFConvertableEnum)]
        convertable_cols = [col for col in df.columns if (hasattr(type(df[col].iloc[0]), 'get_pandas_categories_type') and hasattr(type(df[col].iloc[0]), 'convert_to_hdf'))] # this works, ['track_membership', 'neuron_type']

        for col in convertable_cols:
            col_type = type(df[col].iloc[0])
            cat_type = col_type.get_pandas_categories_type()
            if df[col].dtype != cat_type:
                df[col] = df[col].apply(col_type.convert_to_hdf).astype(cat_type)
                
        return df
    
    @classmethod
    def restore_hdf_dataframe_column_original_type(cls, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """ Restores the original type to the specified column (column_name) in the dataframe after loading from an HDF5 file.
    
        Usage:

                    
        """
        assert column_name in df.columns
        df[column_name] = df[column_name].apply(cls.from_hdf_coding_string)
        return df





class HDF_Converter:
    """ holds static functions that convert specific types to an HDF compatible datatype. 

    Created custom `HDF_Converter` to hold static attribute conversions. Might be a little more elogant to register convertable types but they might need to be converted in different ways in different circumstances

    Ideally can have a bunch of types like:
        Path = str(v)

    """
    
    
    # Static Conversion Functions ________________________________________________________________________________________ #
    

    # TO HDF _____________________________________________________________________________________________________________ #

    @staticmethod
    def _convert_dict_to_hdf_attrs_fn(f, key: str, value):
        """ value: dict-like """
        if isinstance(f, h5py.File):
            for sub_k, sub_v in value.items():
                f[f'{key}/{sub_k}'] = sub_v
        else:
            with h5py.File(f, "a") as f:
                for sub_k, sub_v in value.items():
                    f[f'{key}/{sub_k}'] = sub_v


    # @staticmethod
    # def _convert_dict_to_hdf_attrs_fn(f, key: str, value):
    #     """ value: dict-like """
    #     # if isinstance(f, h5py.File):
    #     with h5py.File(f, "a") as f:
    #         for sub_k, sub_v in value.items():
    #             f[f'{key}/{sub_k}'] = sub_v

    #         # with f.create_group(key) as g:
    #         #     for sub_k, sub_v in value.items():
    #         #         g[f'{key}/{sub_k}'] = sub_v

    # @staticmethod
    # def _convert_optional_ndarray_to_hdf_attrs_fn(f, key: str, value):
    #     """ value: dict-like """
    #     # if isinstance(f, h5py.File):
    #     with h5py.File(f, "a") as f:
    #         if value is not None:
    #             f[f'{key}'] = value
    #         else:
    #             f[f'{key}'] = np.ndarray([])


    # @staticmethod
    # def _convert_Path_dict_to_hdf_attrs_fn(f, key: str, value):
    #     for sub_k, sub_v in value.items():
    #         f[f'{key}/{sub_k}'] = str(sub_v)

    #     # (lambda f, k, v: f[f'{key}/{sub_k}'] = str(sub_v) for sub_k, sub_v in value.items())
    #     assert isinstance(value, Path)
    #     return str(value)

    # Value Type Conversion functions: `_prepare_{A_TYPE}_value_to_for_hdf_fn`
    ## Do not write to f themselves, simply convert values of a specific type to a HDF or PyTables compatable type:
    @staticmethod
    def _prepare_datetime_timedelta_value_to_for_hdf_fn(f, key: str, value) -> np.int64:
        # Convert timedelta to seconds and then to nanoseconds
        assert isinstance(value, timedelta)
        time_in_seconds = value.total_seconds()
        time_in_nanoseconds = int(time_in_seconds * 1e9)
        # Convert to np.int64 (64-bit integer) for tb.Time64Col()
        time_as_np_int64 = np.int64(time_in_nanoseconds)
        return time_as_np_int64

    @classmethod
    def prepare_neuron_indexed_dataframe_for_hdf(cls, neuron_indexed_df: pd.DataFrame, active_context: IdentifyingContext, aclu_column_name: Optional[str]='aclu') -> pd.DataFrame:
        """ prepares a neuron-indexed dataframe (one with an entry for each neuron and an aclu column) for export to hdf5 by converting specific columns to the categorical type if needed """

        if aclu_column_name is None:
            # if aclu_column_name is None, use the index:
            sess_specific_aclus = list(neuron_indexed_df.index.to_numpy())
        else:
            sess_specific_aclus = list(neuron_indexed_df[aclu_column_name].to_numpy())

        

        # Adds column columns=['neuron_uid', 'session_uid', 'aclu']
        neuron_indexed_df['aclu'] = neuron_indexed_df.get('aclu', sess_specific_aclus)  # add explicit 'aclu' column from index if it doesn't exist

        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
        neuron_indexed_df['session_uid'] = session_ctxt_key # add fixed 'session_uid' column 
        neuron_indexed_df['neuron_uid'] = [f"{session_ctxt_key}|{aclu}" for aclu in sess_specific_aclus]

        # Convert any Enum-typed dataframe columns to the HDF5-compatible categorical type if needed
        neuron_indexed_df = HDFConvertableEnum.convert_dataframe_columns_for_hdf(neuron_indexed_df)

        return neuron_indexed_df

    @staticmethod
    def _prepare_pathlib_Path_for_hdf_fn(f, key: str, value):
        assert isinstance(value, Path)
        return str(value)
    
    @staticmethod
    def _try_default_to_hdf_conversion_fn(file_path, key: str, value, file_mode:str='a'):
        """ naievely attempts to save the value `a_value` out to hdf based on its type. Even if it works it might not be correct or deserializable due to datatype issues. 

        #TODO 2023-07-31 06:10: - [ ] This currently clobbers any existing dataset due to HDF5 limitations on overwriting/replacing Datasets with ones of different type or size. Make sure this is what I want.
            # using `require_dataset` instead of `create_dataset` allows overwriting the existing dataset. Otherwise if it exists it throws: `ValueError: Unable to create dataset (name already exists)`
        
        """
        with h5py.File(file_path, file_mode) as f:
            # if isinstance(a_value, dict):
                # for attribute, value in a_value.items():
            # Only flat (non-recurrsive) types allowed.

            # Remove old, differently sized dataset if it exists because HDF5 doesn't allow overwriting datasets with one of a different type or size:
            if key in f:
                print(f'WARNING: clobbering existing dataset in {file_path} at {key}! Will be replaced by new value.')
                # could at least backup metadata first to restore later:
                # attrs_backup = deepcopy(f[key].attrs)
                del f[key]

        if isinstance(value, pd.DataFrame):
            value.to_hdf(file_path, key=key) #TODO 2023-07-31 06:07: - [ ] Is using pandas built-in .to_hdf an instance of double saving (h5py.File and pandas accessing same file concurrently?). In that cause I'd need to bring it out of the loop.
        elif isinstance(value, np.ndarray):
            with h5py.File(file_path, file_mode) as f:
                f.create_dataset(key, data=value)
            
        ## TODO: LOL THESE DO NOTHING BELOW HERE, no writes
        elif isinstance(value, (list, tuple)):
            # convert to np.array before saving
            value = np.array(value)
            with h5py.File(file_path, file_mode) as f:
                f.create_dataset(key, data=value)
        else:
            # ... handle other attribute types as needed
            raise NotImplementedError


    # FROM HDF ___________________________________________________________________________________________________________ #
    @classmethod
    def _restore_dataframe_byte_strings_to_strings(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ converts columns containing byte strings (b'aString') to normal strings ('aString') """
        for col in df:
            if isinstance(df[col][0], bytes):
                # print(col, "will be transformed from bytestring to string")
                df[col] = df[col].str.decode("utf8")  # or any other encoding


    @classmethod
    def expand_dataframe_session_context_column(cls, non_expanded_context_df: pd.DataFrame, session_uid_column_name:str='session_uid') -> pd.DataFrame:
        """ expands a column (session_uid_column_name) containing a str representation of the session context (e.g. 'kdiba|gor01|one|2006-6-08_14-26-15') into its four separate component ['format_name', 'animal', 'exper_name', 'session_name'] columns.
        Additionally adds the 'session_datetime' column if it can be parsed from the 'session_name' column.
         """
        def check_all_columns_exist(df, required_columns):
            return all(col in df.columns for col in required_columns)


        assert session_uid_column_name in non_expanded_context_df.columns
        assert len(non_expanded_context_df[session_uid_column_name]) > 0 # must have at least one element
        if isinstance(non_expanded_context_df[session_uid_column_name][0], str):
            # String representations of session contexts ('session_uid'-style):
            non_expanded_context_df = non_expanded_context_df.astype({session_uid_column_name: 'string'})
            all_sess_context_tuples = [tuple(a_session_uid.split('|', maxsplit=4)) for a_session_uid in non_expanded_context_df[session_uid_column_name]]
        elif isinstance(non_expanded_context_df[session_uid_column_name][0], IdentifyingContext):
            # IdentifyingContext type objects:
            all_sess_context_tuples = [a_ctx.as_tuple() for a_ctx in non_expanded_context_df[session_uid_column_name]] #[('kdiba', 'gor01', 'one', '2006-6-07_11-26-53'), ('kdiba', 'gor01', 'one', '2006-6-08_14-26-15'), ('kdiba', 'gor01', 'one', '2006-6-09_1-22-43'), ...]
        else:
            raise TypeError         
        
        # Check if the dataframe already has the appropriate rows:
        if not check_all_columns_exist(non_expanded_context_df, required_columns=IdentifyingContext._get_session_context_keys()):
            expanded_context_df = pd.DataFrame.from_records(all_sess_context_tuples, columns=IdentifyingContext._get_session_context_keys())
            out_df = pd.concat((expanded_context_df, non_expanded_context_df), axis=1)
        else:
            out_df = non_expanded_context_df
            
        if not 'session_datetime' in out_df.columns:
            # parse session date if possible:
            # Apply the extract_date function to the 'session_name' column to create a new 'session_date' column
            out_df['session_datetime'] = out_df['session_name'].apply(IdentifyingContext.try_extract_date_from_session_name)

        return out_df

    @classmethod
    def restore_native_column_types_manual_if_needed(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ 2023-08-24
        Usage:
            restore_native_column_types_manual_if_needed(_out_table)
        """
        # restore native column types:
        from neuropy.core.neurons import NeuronType
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership

        # manual conversion is required for some reason:
        
        if ("track_membership" in df.columns) and (df.dtypes["track_membership"] == np.int8):
            df["track_membership"] = df["track_membership"].apply(lambda x: SplitPartitionMembership.hdf_coding_ClassNames()[x]).astype(object)
        if ("neuron_type" in df.columns) and ((df.dtypes["neuron_type"] == np.int8) or (df.dtypes["neuron_type"] == np.uint8)):
            df["neuron_type"] = df["neuron_type"].apply(lambda x: NeuronType.hdf_coding_ClassNames()[x]).astype(object)

        cls._restore_dataframe_byte_strings_to_strings(df)
        return df

    @classmethod
    def general_post_load_restore_table_as_needed(cls, df: pd.DataFrame, session_uid_column_name='session_uid') -> pd.DataFrame:
        """ 2023-08-24 should be generally safe to apply on loaded PyTables tables loaded as dataframes.

        Usage:
            _out_table = general_post_load_restore_table_as_needed(_out_table)
        """
        cls.restore_native_column_types_manual_if_needed(df)
        df = cls.expand_dataframe_session_context_column(df, session_uid_column_name=session_uid_column_name)
        return df


# ==================================================================================================================== #
# HDF Serialization (saving to HDF5 file)                                                                              #
# ==================================================================================================================== #
_ALLOW_GLOBAL_NESTED_EXPANSION:bool = False
# _ALLOW_GLOBAL_NESTED_EXPANSION:bool = True

class HDF_SerializationMixin:
    """
    Inherits `get_serialized_dataset_fields` from AttrsBasedClassHelperMixin

    Used to be
        class HDF_SerializationMixin(AttrsBasedClassHelperMixin):
    but this didn't work and was found to be the source of the pickling issues. 

    Now I'll need to trace everything and find:

        AttrsBasedClassHelperMixin

        

    2. Actual attrs classes that didn't inherity from AttrsBasedClassHelperMixin because it would be redundant with their conformance to HDFMixin or HDF_SerializationMixin. Now they'll need to directly conform to `AttrsBasedClassHelperMixin` as well
    3. Some non-attrs classes that didn't provide a custom .to_hdf(...) function but happened to have it work anyway will need to have their custom .to_hdf(...) re-written.
    

    """
    
    @classmethod
    def is_hdf_serializable(cls):
        """ returns whether the class is completely hdf serializable. """
        return True

    # Static Conversion Functions ________________________________________________________________________________________ #

   
    # Main Methods _______________________________________________________________________________________________________ #

    
    def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        enable_hdf_testing_mode: bool - default False - if True, errors are not thrown for the first field that cannot be serialized, and instead all are attempted to see which ones work.
        
    
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        
        # file_mode:str = kwargs.get('file_mode', 'w') # default to overwrite
        file_mode:str = kwargs.get('file_mode', 'a') # default to append
        
        if not attrs.has(type(self)):
            raise NotImplementedError # implementor must override!
    
        if not isinstance(self, AttrsBasedClassHelperMixin):
            raise NotImplementedError # automatic `to_hdf` only supported on `AttrsBasedClassHelperMixin`-derived objects.
        
        ## Automatic implementation if class is `AttrsBasedClassHelperMixin`
        if debug_print:
            print(f'WARNING: experimental automatic `to_hdf` implementation for object of type {type(self)} to file_path: {file_path}, with key: {key}:')
            
        # ==================================================================================================================== #
        # Serializable Dataset HDF5 Fields:                                                                                    #
        # ==================================================================================================================== #
        hdf_dataset_fields, hdf_dataset_fields_filter_fn = self.get_serialized_dataset_fields('hdf')
        active_hdf_dataset_fields_dict = {a_field.name:a_field for a_field in hdf_dataset_fields} # a dict that allows accessing the actual attr by its name
        # use `asdict` to get a dictionary-representation of self only for the `hdf` serializable fields
        try:
            _active_obj_dataset_values_dict = asdict(self, filter=hdf_dataset_fields_filter_fn, recurse=False)
        except AttributeError as err:
            # happens when the type of `self` is modified after a pickled version is saved. The unpickled result seems to be lacking the property and asdict fails
            print(f'WARN: to_hdf(..., key: {key}, ...): \n\tasdict(...) failed with error: {err}')
            _active_obj_dataset_values_dict = asdict(self, filter=hdf_dataset_fields_filter_fn, recurse=False)
        except BaseException:
            raise


        # print(f'_temp_obj_dict: {_active_obj_values_dict}')
        if enable_hdf_testing_mode:
            unserializable_fields = {}

        for a_field_name, a_value in _active_obj_dataset_values_dict.items():
            a_field_attr = active_hdf_dataset_fields_dict[a_field_name]
            if debug_print:
                print(f'a_field: {a_field_attr.name}')
            a_field_key:str = f'{key}/{a_field_attr.name}'
            if debug_print:
                print(f'\ta_field_key: {a_field_key}')
            
            is_custom_serializable:bool = False
            try:
                is_custom_serializable = a_field_attr.type.is_hdf_serializable()
            except AttributeError as e:
                # AttributeError: type object 'numpy.ndarray' has no attribute 'is_hdf_serializable' is expected when not native serializing type
                is_custom_serializable = False
            except Exception as e:
                raise e # unhandled exception!!
        
            custom_serialization_fn = a_field_attr.metadata.get('custom_serialization_fn', None) # (lambda f, k, v: a_value)
            if custom_serialization_fn is not None:
                # use the custom serialization function:
                custom_serialization_fn(file_path, a_field_key, a_value)
            else:
                # No custom serialization function.
                if is_custom_serializable:
                    ## Known `hdf_serializable` field, meaning it properly implements its own `to_hdf(...)` function! We can just call that! 
                    ## use that fields' to_hdf function
                    if debug_print:
                        print(f'\t field is serializable! Calling a_value.to_hdf(...)...')
                    a_value.to_hdf(file_path=file_path, key=a_field_key)
                else:
                    ## field is not known to be hdf_serializable! It might not serialize correctly even if this method doesn't throw an error.
                    if debug_print:
                        print(f'\t field not custom serializable! a_field_attr.type: {a_field_attr.type}.')                    
                        print(f'WARNING: {a_field_key} is not custom serializable, but we will try HDF_Converter._try_default_to_hdf_conversion_fn(file_path=file_path, key=a_field_key, value=a_value) with the value. Will raise a NotImplementedException if this fails.')

                    # if the user set the "is_hdf_handled_custom" field, it will be handled in the overriden .to_hdf(...)
                    is_handled_in_overriden_to_hdf = a_field_attr.metadata.get('is_hdf_handled_custom', None) or False
                    if not is_handled_in_overriden_to_hdf:
                        try:
                            HDF_Converter._try_default_to_hdf_conversion_fn(file_path=file_path, key=a_field_key, value=a_value, file_mode=file_mode)
                        except NotImplementedError as e:
                            ## If not handled in the default method, try global expansion if that is alowed. Otherwise we're out of options.
                            if _ALLOW_GLOBAL_NESTED_EXPANSION:
                                if hasattr(a_value, 'to_dict'):
                                    a_value = a_value.to_dict() # convert to dict using the to_dict() method.
                                elif hasattr(a_value, '__dict__'):
                                    a_value = a_value.__dict__ #TODO 2023-07-30 19:31: - [ ] save the type as metadata
                                else:
                                    # Last resort is to try dir(a_value)
                                    a_value = {attr: getattr(a_value, attr) for attr in dir(a_value) if not callable(getattr(a_value, attr)) and not attr.startswith('__')} # More general way (that works for DynamicProperties, etc) I think `dir()` or __dir__() or something?`

                                if isinstance(a_value, dict):
                                    with h5py.File(file_path, file_mode) as f:
                                        for attribute, value in a_value.items():
                                            sub_field_key:str = f"{a_field_key}/{attribute}"
                                            HDF_Converter._try_default_to_hdf_conversion_fn(file_path=file_path, key=sub_field_key, value=a_value, file_mode=file_mode)
                                else:
                                    if enable_hdf_testing_mode:
                                        print(f'NotImplementedError("a_field_attr: {a_field_attr} could not be serialized with value a_value: {a_value} because it could not be converted to a dict via any known method")')
                                        unserializable_fields[a_field_attr] = f"a_field_attr: {a_field_attr} could not be serialized with value a_value: {a_value} because it could not be converted to a dict via any known method"
                                    else:
                                        raise NotImplementedError(f"a_field_attr: {a_field_attr} could not be serialized with value a_value: {a_value} because it could not be converted to a dict via any known method") # really not implemented

                                    
                            else:
                                # _ALLOW_GLOBAL_NESTED_EXPANSION is not allowed.
                                if enable_hdf_testing_mode:
                                    print(f'NotImplementedError("a_field_attr: {a_field_attr} could not be serialized and _ALLOW_GLOBAL_NESTED_EXPANSION is not allowed.")')
                                    unserializable_fields[a_field_attr] = f"a_field_attr: {a_field_attr} could not be serialized and _ALLOW_GLOBAL_NESTED_EXPANSION is not allowed." # NotImplementedError(f"a_field_attr: {a_field_attr} could not be serialized and _ALLOW_GLOBAL_NESTED_EXPANSION is not allowed.")
                                else:
                                    raise NotImplementedError(f"a_field_attr: {a_field_attr} could not be serialized and _ALLOW_GLOBAL_NESTED_EXPANSION is not allowed.")

                        except Exception as e:
                            # Unhandled exception
                            if enable_hdf_testing_mode:
                                unserializable_fields[a_field_attr] = e
                            else:
                                raise e

                    else:
                        if debug_print:
                            print(f'field "{a_field_name}" with key "{a_field_key}" has "is_hdf_handled_custom" set, meaning the inheritor from this class must handle it in the overriden method.')


            # Post serializing the dataset, set any hdf_metadata properties it might have:
            ## NOTE: this is still within the datasets loop and just sets the metadata for the specific dataset assigned a value above!
            custom_dataset_field_hdf_metadata = a_field_attr.metadata.get('hdf_metadata', {})
            if len(custom_dataset_field_hdf_metadata) > 0: # don't open the file for no reason
                with h5py.File(file_path, file_mode) as f:
                    group = f[key]
                    for an_hdf_attr_name, a_value in custom_dataset_field_hdf_metadata.items():
                        group.attrs[an_hdf_attr_name] = a_value


        # ==================================================================================================================== #
        # Serializable HDF5 Attributes Fields (metadata set on the HDF5 Group corresponding to this object):                   #
        # ==================================================================================================================== #
        ## Get attributes fields as well
        hdf_attr_fields, hdf_attr_fields_filter_fn = self.get_serialized_attribute_fields('hdf')
        active_hdf_attributes_fields_dict = {a_field.name:a_field for a_field in hdf_attr_fields} # a dict that allows accessing the actual attr by its name
        _active_obj_attributes_values_dict = asdict(self, filter=hdf_attr_fields_filter_fn, recurse=False) # want recurse=True for this one?
        
        # Actually assign the attributes to the group:
        if len(_active_obj_attributes_values_dict) > 0: # don't open the file for no reason
            # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
            with h5py.File(file_path, 'r+') as f:
                group = f[key]
                for a_field_name, a_value in _active_obj_attributes_values_dict.items():
                    a_field_attr = active_hdf_attributes_fields_dict[a_field_name]
                    if debug_print:
                        print(f'an_attribute_field: {a_field_attr.name}')

                    custom_serialization_fn = a_field_attr.metadata.get('custom_serialization_fn', None) # (lambda f, k, v: a_value)
                    if custom_serialization_fn is not None:
                        # use the custom serialization function:
                        custom_serialization_fn(group.attrs, a_field_attr.name, a_value)
                    else:
                        # set that group attribute to a_value
                        group.attrs[a_field_attr.name] = a_value #TODO 2023-07-31 05:50: - [ ] Assumes that the value is valid to be used as an HDF5 attribute without conversion.




""" Example to_hdf/read_hdf based only on the type of the values:

def to_hdf(self, file_path):
    with h5py.File(file_path, 'w') as f:
        for attribute, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                value.to_hdf(file_path, key=attribute)
            elif isinstance(value, np.ndarray):
                f.create_dataset(attribute, data=value)
            # ... handle other attribute types as needed ...

@classmethod
def read_hdf(cls, file_path):
    with h5py.File(file_path, 'r') as f:
        attrs_dict = {}
        for attribute in cls.__annotations__:
            if attribute in f:
                if pd.api.types.is_categorical_dtype(f[attribute]):
                    attrs_dict[attribute] = pd.read_hdf(file_path, key=attribute)
                else:
                    attrs_dict[attribute] = np.array(f[attribute])
            # ... handle other attribute types as needed ...
    return cls(**attrs_dict)
"""    


        # self.position.to_hdf(file_path=file_path, key=f'{key}/pos')
        # if self.epochs is not None:
        #     self.epochs.to_hdf(file_path=file_path, key=f'{key}/epochs') #TODO 2023-07-30 11:13: - [ ] What if self.epochs is None?
        # else:
        #     # if self.epochs is None
        #     pass
        # self.spikes_df.spikes.to_hdf(file_path, key=f'{key}/spikes')

        # # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
        # with h5py.File(file_path, 'r+') as f:
        #     ## Unfortunately, you cannot directly assign a dictionary to the attrs attribute of an h5py group or dataset. The attrs attribute is an instance of a special class that behaves like a dictionary in some ways but not in others. You must assign attributes individually
        #     group = f[key]
        #     group.attrs['position_srate'] = self.position_srate
        #     group.attrs['ndim'] = self.ndim

        #     # can't just set the dict directly
        #     # group.attrs['config'] = str(self.config.to_dict())  # Store as string if it's a complex object
        #     # Manually set the config attributes
        #     config_dict = self.config.to_dict()
        #     group.attrs['config/speed_thresh'] = config_dict['speed_thresh']
        #     group.attrs['config/grid_bin'] = config_dict['grid_bin']
        #     group.attrs['config/grid_bin_bounds'] = config_dict['grid_bin_bounds']
        #     group.attrs['config/smooth'] = config_dict['smooth']
        #     group.attrs['config/frate_thresh'] = config_dict['frate_thresh']
            


# General/Combined ___________________________________________________________________________________________________ #

class HDFMixin(HDF_DeserializationMixin, HDF_SerializationMixin):
    # Common methods for serialization and deserialization
    pass


"""
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

class SpecialClassHDFMixin(HDFMixin):
    # Custom methods for a specific class

class MyClass(SpecialClassHDFMixin):
    # MyClass definition

"""



# class HDF5_Initializable(FileInitializable):
# 	""" Implementors can be initialized from a file path (from which they are loaded)
# 	"""
# 	@classmethod
# 	def from_file(cls, f):
# 		assert isinstance(f, (str, Path))
# 		raise NotImplementedError


# class HDF5_Representable(HDF5_Initializable, FileRepresentable):
# 	""" Implementors can be loaded or saved to a file
# 	"""
# 	@classmethod
# 	def to_file(cls, data: dict, f):
# 		raise NotImplementedError

 
# 	def save(self):
# 		raise NotImplementedError



# ==================================================================================================================== #
# 2023-08-01 Item Conversions for HDF5 Representations                                                                 #
# ==================================================================================================================== #
""" The above Mixins regard objects that should independently load/save themselves out to an HDF5 file.

In addition, some class (like enums) might need a way to specify how their value should be converted to an HDF5 datatype and how it can be recognized to be deserialized.

`as_hdf_representable` - returns an HDF representation of itself along with metadata to indicate how it can be reconstructed.


value: the object represented as an HDF5 datatype, potentially a compound type but not something advanced enough to warrent its own `to_hdf` and `from_hdf` methods

metadata: to be attached to the property
    - class_name
    - serialization_version - constrains what deserialization version should/can be used to reconstruct the object
    - serialization_datetime
    - 


`Class.init_from_hdf_representation` - the inverse deserialization method



How should a List[NeuronType] or a Dict[int:NeuronType] be written out to HDF5? Probably as a np.array of simpler representations, right? Otherwise we get stupid with the hierarchy keys






"""


