#!/usr/bin/env python3

## Main API of the project
# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# Classes :
# - PreProcessor -> SkLearn Pipeline compatible class interface
#
# Fonctions :
# - get_preprocessor -> Returns a PreProcessor class instance
# - preprocess_pipeline -> Preprocessing pipeline
# - check_pipeline_order -> Checks the sequence of transformations for unexpected behaviours
# - listing_count_words -> Word count and listing
# - list_one_appearance_word -> Lists words occuring only once in the documents


import os
import gc
import copy
import json
import functools
import numpy as np
import pandas as pd
from typing import Union

from words_n_fun import utils
from words_n_fun.preprocessing import basic


# Get logger
import logging

logger = logging.getLogger(__name__)

# Dict of all the available preprocessing features
# All these transformations are one to one (one input will lead to only input) such that these
# transformations can be chained while preserving the input dataset shape
# Some advices are specified in configs/pipeline_usage_order.json about how to order these transformations
# They also act as warnings when some sequences of transformations might have some unexpected results
USAGE = {
    'notnull': basic.notnull,  # Replaces null values by an empty character
    'remove_non_string': basic.remove_non_string,  # Replaces all non strings by an empty character
    'get_true_spaces': basic.get_true_spaces,  # Replaces all whitespaces by a single space
    'remove_accents': basic.remove_accents,  # Removes all accents and special characters (ç..)
    'remove_stopwords': functools.partial(basic.remove_stopwords, opt='all'),  # Removes stopwords
    'trim_string': basic.trim_string,  # Trims spaces: multiple spaces become one
    'remove_leading_and_ending_spaces': basic.remove_leading_and_ending_spaces,  # Removes leading and trailing spaces
    'remove_punct': functools.partial(basic.remove_punct, del_parenthesis=True, replacement_char=' '),  # Replaces all non alpha-numeric characters by spaces
    'remove_punct_except_parenthesis': functools.partial(basic.remove_punct, del_parenthesis=False, replacement_char=' '),   # Replaces all non alpha-numeric characters by spaces EXCEPT parenthesis and slashes
    'pe_matching': basic.pe_matching,  # Specific one-to-one tokens replacements
    'to_lower': functools.partial(basic.to_lower, threshold_nb_chars=0),  # Transforms the string to lower case
    'to_lower_except_singleletters': functools.partial(basic.to_lower, threshold_nb_chars=2),  # Transforms longer than 2 characters string to lower cases
    'remove_numeric': functools.partial(basic.remove_numeric, replacement_char=' '),  # Replaces numeric strings by a space
    'remove_gender_synonyms': basic.remove_gender_synonyms,  # [French] Removes gendered synonyms
    'lemmatize': basic.lemmatize,  # Lemmatizes the document
    'stemmatize': basic.stemmatize,  # Stemmatizes the words of the document
    'add_point': basic.add_point,  # Adds a dot at the end of each line
    'add_space_around_special': basic.deal_with_specific_characters,  # Ads spaces before and after some punctuations (, : ; .)
    'replace_urls' : basic.replace_urls,  # Replaces URLs by spaces
    'replace_urls_with_domains' : functools.partial(basic.replace_urls, replace_with_domain=True),  # Replaces URLs by their domain part
    'fix_text': basic.fix_text,  # Fixes numerous inconsistencies within a text (via ftfy)
}


# Default pipeline
DEFAULT_PIPELINE = ['remove_non_string', 'get_true_spaces', 'to_lower_except_singleletters', 'pe_matching',
                    'remove_gender_synonyms', 'remove_punct_except_parenthesis', 'remove_numeric',
                    'remove_stopwords', 'stemmatize', 'remove_accents', 'trim_string', 'remove_leading_and_ending_spaces']


class PreProcessor():
    '''Class PreProcessor:
    This class implements fit & transform methods and can therefore be used
    to insert a preprocessing pipeline into a Sklearn pipeline
    '''

    def __init__(self, pipeline: Union[list, None] = DEFAULT_PIPELINE, prefered_column: str = 'docs',
                 modify_data: bool = True, chunksize: int = 0, first_row: str = 'header',
                 columns: list = ['docs', 'tags'], sep: str = ',', nrows: int = 0, **pandas_args) -> None:
        '''Class constructor
        The purpose of a lot of these arguments are to handle the case when the input of the transform method is a path to
        a csv file. While handy, this use case is not advised.

        Kwargs:
            pipeline (list): List of transformations to apply (from the USAGE dict) (default: DEFAULT_PIPELINE)
            prefered_column (str): Default column name to consider as the document container when working with a pandas dataframe or csv file (default: 'docs')
            modify_data (boolean): When working with a pandas dataframe or csv file, specifies wether the input data is modified or a new column is created (default: True)
            chunksize (int): If not 0 the pipeline is processed chunkwise and this parameter specifies the chunksize (dafault : 0)
            first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled -'header', 'data' or 'skip' (default : 'header')
            columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use, if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing (default : ['docs', 'tags'])
            sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
            nrows (int) : When working with a pandas dataframe or csv file, specifies the maximum number of lines to read (default: 0 we take it all)
            pandas_args : When working with a pandas dataframe or csv file, specifies arguments to pass to pandas
        Raises:
            ValueError: If chunksize < 0
            ValueError: If first_row is different than 'header', 'data' or 'skip'
            ValueError: If nrows < 0
        '''
        if chunksize < 0:
            raise ValueError("chunksize parameter must be >= 0")
        if first_row not in ['header', 'data', 'skip']:
            raise ValueError('first_row parameter must be one of header, data, or skip')
        if nrows < 0:
            raise ValueError('nrows parameter must be >= 0')
        if modify_data != True:
            logger.warning("modify data must be True for the preprocessor class to remain Sklearn compatible")
        # Set properties
        self.pipeline = pipeline
        self.prefered_column = prefered_column
        self.modify_data = modify_data
        self.chunksize = chunksize
        self.first_row = first_row
        self.columns = columns
        self.sep = sep
        self.nrows = nrows
        self.pandas_args = pandas_args

    def fit(self):
        '''Required to be compatible with Sklearn pipelines'''
        pass

    def transform(self, docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame]) -> Union[str, list, np.ndarray, pd.Series, pd.DataFrame]:
        '''Wrapper around preprocess_pipeline

        Args:
            docs (?): Documents to be preprocessed (compatible types : str ending by .csv, str, list, np.ndarray, pd.Series, pd.DataFrame)
        Returns:
            ?: Preprocessed documents (the initial type is preserved except for str ending by .csv -> pd.DataFrame)
        '''
        if not isinstance(docs, pd.Series):
            logger.warning("pd.Series is the prefered type for api.Preprocessor, other types might not be compatible with some Sklearn pipelines ")
        return preprocess_pipeline(docs, pipeline=self.pipeline, prefered_column=self.prefered_column, modify_data=self.modify_data,
                                   chunksize=self.chunksize, first_row=self.first_row, columns=self.columns, sep=self.sep,
                                   nrows=self.nrows, **self.pandas_args)


def get_preprocessor(pipeline: list = DEFAULT_PIPELINE, prefered_column: str = 'docs', modify_data: bool = True,
                     chunksize: int = 0, first_row: str = 'header', columns: list = ['docs', 'tags'], sep: str = ',',
                     nrows: int = 0, **pandas_args) -> PreProcessor:
    '''Retourne une instance de PreProcessor

    Kwargs:
        pipeline (list): List of transformations to apply (from the USAGE dict) (default: DEFAULT_PIPELINE)
        prefered_column (str): Default column name to consider as the document container when working with a pandas dataframe or csv file (default: 'docs')
        modify_data (boolean): When working with a pandas dataframe or csv file, specifies wether the input data is modified or a new column is created (default: True)
        chunksize (int): If not 0 the pipeline is processed chunkwise and this parameter specifies the chunksize (default : 0)
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled -'header', 'data' or 'skip' (default : 'header')
        columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use, if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing (default : ['docs', 'tags'])
        sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
        nrows (int) : When working with a pandas dataframe or csv file, specifies the maximum number of lines to read (default: 0 we take it all)
        pandas_args : When working with a pandas dataframe or csv file, specifies arguments to pass to pandas
    Returns:
        PreProcessor: A PreProcessor instance with its pipeline set
    '''
    logger.debug('Calling api.get_preprocessor')
    return PreProcessor(pipeline=pipeline, prefered_column=prefered_column, modify_data=modify_data,
                        chunksize=chunksize, first_row=first_row, columns=columns, sep=sep,
                        nrows=nrows, **pandas_args)


def preprocess_pipeline(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame],
                        pipeline: list = DEFAULT_PIPELINE, prefered_column: str = 'docs',
                        modify_data: bool = True, chunksize: int = 0, first_row: str = 'header',
                        columns: list = ['docs', 'tags'], sep: str = ',', nrows: int = 0,
                        **pandas_args) -> Union[str, list, np.ndarray, pd.Series, pd.DataFrame]:
    '''Preprocessing pipeline

    Args:
        docs (?): Documents to be preprocessed (compatible types : str ending by .csv, str, list, np.ndarray, pd.Series, pd.DataFrame)
    Kwargs:
        pipeline (list): List of transformations to apply (from the USAGE dict) (default: DEFAULT_PIPELINE)
        prefered_column (str): Default column name to consider as the document container when working with a pandas dataframe or csv file (default: 'docs')
        modify_data (boolean): When working with a pandas dataframe or csv file, specifies wether the input data is modified or a new column is created (default: True)
        chunksize (int): If not 0 the pipeline is processed chunkwise and this parameter specifies the chunksize (default : 0)
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled -'header', 'data' or 'skip' (default : 'header')
        columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use, if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing (default : ['docs', 'tags'])
        sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
        nrows (int) : When working with a pandas dataframe or csv file, specifies the maximum number of lines to read (default: 0 we take it all)
        pandas_args : When working with a pandas dataframe or csv file, specifies arguments to pass to pandas
    Raises:
        ValueError: If chunksize < 0
        ValueError: If first_row is different than 'header', 'data' or 'skip'
        ValueError: If nrows < 0
    Returns:
        ?: Preprocessed documents (the initial type is preserved except for str ending by .csv -> pd.DataFrame)
    '''
    logger.debug('Calling api.preprocess_pipeline')
    if chunksize < 0:
        raise ValueError("chunksize parameter must be >= 0")
    if first_row not in ['header', 'data', 'skip']:
        raise ValueError('first_row parameter must be one of header, data, or skip')
    if nrows < 0:
        raise ValueError('nrows parameter must be >= 0')
    # Check the order of transformations in the pipeline, warnings are displayed if unexpected behaviours could occur
    check_pipeline_order(pipeline)
    # Get docs type
    docs_type = utils.get_docs_type(docs)
    # Get nb of elements to process
    docs_length = utils.get_docs_length(docs, first_row=first_row, sep=sep, nrows=nrows)
    max_chunksize = min(chunksize, docs_length) if chunksize != 0 else docs_length
    # We need to deepcopy the data if it is a pandas dataframe
    if docs_type in ('pd.DataFrame', 'file_path'):
        docs_copy = copy.deepcopy(docs)
    else:
        docs_copy = docs  # Not really a copy, no need & avoid memory waste
    gen = utils.get_generator(docs_copy, chunksize=chunksize, first_row=first_row,
                              columns=columns, sep=sep, nrows=nrows, **pandas_args)
    # Get the columns name that need to be processed (if working with a dataframe or csv file)
    docs_column = utils.get_column_to_be_processed(docs_copy, prefered_column=prefered_column,
                                                   first_row=first_row, columns=columns, sep=sep)
    # If we are working with a file, we get a new csv file to store the output
    # Otherwise we get a new column
    if docs_type == 'file_path':
        new_csv_file = utils.get_new_csv_name(docs_copy)
        if not modify_data:
            column_to_write = utils.get_new_column_name(utils.get_columns_to_use(docs_copy, first_row=first_row, columns=columns, sep=sep), docs_column)
        else:
            column_to_write = docs_column
    elif docs_type == 'pd.DataFrame':
        column_to_write = utils.get_new_column_name(list(docs_copy.columns), docs_column) if not modify_data else docs_column
    docs_outputs = []  # Will contain the reults of the preprocessing pipeline if we are note working with csv files
    # Chunk iteration
    for i, docs_gen in enumerate(gen):
        if chunksize != 0:
            logger.info(f"Processing chunck n°{i + 1}:")
        # For files or dataframes, we get the column to work with
        if docs_type in ('pd.DataFrame', 'file_path'):
            docs_input = docs_gen[docs_column]
        else:
            docs_input = docs_gen
        # Sequential processing of all the pipeline transformations
        for item in pipeline:
            # If item is a string, we apply the corresponding function from USAGE
            if item in USAGE.keys():
                logger.info(f"Preprocessing: step {item}")
                docs_input = USAGE[item](docs_input)
            # If it's a callable, it is directly called
            elif callable(item):
                logger.info(f"Preprocessing: step {item}")
                docs_input = item(docs_input)
            # gc collect if more than a thousand elements (improve memory usage)
            if max_chunksize >= 1000:
                gc.collect()
        # If working with a file, we append the processed chunk to the newly created result file
        if docs_type == 'file_path':
            docs_gen[column_to_write] = docs_input
            with_header = True if i == 0 else False
            with open(new_csv_file, 'a', encoding='utf-8') as f:
                docs_gen.to_csv(f, header=with_header, sep=sep, index=False)
        # Otherwise it is appended to docs_outputs
        else:
            docs_outputs.append(docs_input)
    # Manage return types, either a str, a path to the result file, a list, a np.ndarray, a pd.Series or a Dataframe
    if docs_type == 'file_path':
        return new_csv_file
    elif docs_type == 'str':
        return docs_outputs[0]
    elif docs_type == 'list':
        return [elem for docs_output in docs_outputs for elem in docs_output]
    elif docs_type == 'np.ndarray':
        return np.array([elem for docs_output in docs_outputs for elem in docs_output])
    elif docs_type in ['pd.Series', 'pd.DataFrame']:
        series_output = pd.concat(docs_outputs)  # Element pd.Series
        if docs_type == 'pd.Series':
            return series_output
        elif docs_type == 'pd.DataFrame':
            docs_copy[column_to_write] = series_output
            return docs_copy


def check_pipeline_order(pipeline: list) -> None:
    '''Checks the order of transformations in the pipeline, warnings are displayed if unexpected behaviours could occur

    Args:
        pipeline (list): Pipeline to check
    '''
    logger.debug('Calling api.check_pipeline_order')

    # We get pipeline_usage_order. It contains some advices about the sequence of transformations within a pipeline
    conf_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'configs', 'pipeline_usage_order.json')
    with open(conf_file_path, 'r') as f:
        usage_order = json.load(f)

    # Iterates over all the transformations within the pipeline
    for current_number, current_function in enumerate(pipeline):
        if callable(current_function):
            if hasattr(current_function, '__name__'):
                function_name = current_function.__name__
            else:
                function_name = str(current_function)
            logger.info(f"The pipeline contains a custom funtion : {function_name}.")
            continue
        # Skips if the current function is not in USAGE
        if current_function not in USAGE.keys():
            logger.warning(f"Function {current_function}is unknown: SKIP.")
            continue
        # Skips if the current function is not in usage_order
        if current_function not in usage_order.keys():
            continue

        # We get the usage specifics about the current_function
        # It tracks unexpected behaviors (eg: removing stopwords after a stemmatizer is called might not work as expected
        # if words that should be considered as stopwords are truncated)
        order_dict = usage_order[current_function]

        # Cases where the current function has one or more "before" case
        if 'before' in order_dict.keys():
            for before_function in order_dict['before'].keys():
                # If the function before_function exists in the pipeline before current_function, a warning is raised
                if min([len(pipeline)] + [i for i, _ in enumerate(pipeline) if _ == before_function]) < current_number:
                    if order_dict['before'][before_function] != None:
                        logger.warning(f"/!\ /!\ /!\: {order_dict['before'][before_function]}")
                    # usage_order is cleared to avoid duplicate warnings
                    try:
                        usage_order[current_function]['before'][before_function] = None
                        usage_order[before_function]['after'][current_function] = None
                    except:
                        continue

        # Cases where the current function has one or more "after" case
        if 'after' in order_dict.keys():
            for after_function in order_dict['after'].keys():
                # If the function after_function exists in the pipeline after current_function, a warning is raised
                if max([0] + [i for i, _ in enumerate(pipeline) if _ == after_function]) > current_number:
                    if order_dict['after'][after_function] != None:
                        logger.warning(f"/!\ /!\ /!\: {order_dict['after'][after_function]}")
                    # usage_order is cleared to avoid duplicate warnings
                    try:
                        usage_order[current_function]['after'][after_function] = None
                        usage_order[after_function]['before'][current_function] = None
                    except:
                        continue

        # Cases where the current function has one or more "not_before" case
        if 'not_before' in order_dict.keys():
            for not_before_function in order_dict['not_before'].keys():
                if not not_before_function.startswith('OR'):
                    # Checks if the prerequisites to current_function are missing
                    if min([len(pipeline)] + [i for i, _ in enumerate(pipeline) if _ == not_before_function]) > current_number:
                        logger.warning(f"/!\ /!\ /!\: {order_dict['not_before'][not_before_function]}")
                # 'OR' cases:
                else:
                    warn_top = True
                    # Checks if at least one of the prerequisites to current_function is present
                    for or_function in order_dict['not_before'][not_before_function].keys():
                        if min([len(pipeline)] + [i for i, _ in enumerate(pipeline) if _ == or_function]) > current_number:
                            warn_top = False
                            break
                    if warn_top:
                        for or_function in order_dict['not_before'][not_before_function].keys():
                            logger.warning(f"/!\ /!\ /!\: {order_dict['not_before'][not_before_function][or_function]}")


@utils.data_agnostic_input
def listing_count_words(docs: pd.Series) -> pd.DataFrame:
    '''Words listing and counts

    Args:
        docs (pd.Series): Documents to process
    Returns:
        pd.DataFrame: Dataframe listing all the words appearing in the documents along with their respective count
    '''
    logger.debug('Calling api.listing_count_words')
    words = list([word for sentence in docs.str.split() for word in sentence])
    return pd.DataFrame(words, columns=['word']).groupby('word').size().to_frame('count').reset_index()


@utils.data_agnostic_input
def list_one_appearance_word(docs: pd.Series) -> pd.Series:
    '''Lists the words appearing only once in the whole corpus

    Args:
        docs (pd.Series): Documents to process
    Returns:
        pd.Series: List of the words appearing only once
    '''
    logger.debug('Calling fonction api.list_one_appearance_word')
    count_words = listing_count_words(docs)
    # Return result (le reset index permet juste d'avoir un index continue)
    return count_words[count_words['count'] == 1]['word'].reset_index(drop=True)


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
