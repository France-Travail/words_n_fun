#!/usr/bin/env python3

## Utils
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
# Fonctions :
# - timer -> Decorator to monitor the execution time of a function
# - data_agnostic -> Decorator to manage type casting from and to pd.Series
# - data_agnostic_input -> Decorator to manage type casting to pd.Series
# - get_docs_type -> Returns the type of a list
# - get_docs_length -> Returns the number of elements within a set of documents
# - get_file_length -> Returns... the file length !
# - get_new_csv_name -> Returns a new filename ("processed") from a given filename
# - get_generator -> Returns a generator given the type of document to process and the chunksize
# - get_df_generator_from_csv -> Returns a dataFrame generator by chunk over a file
# - get_columns_to_use -> Returns the names of the columns to use while loading a csv file
# - get_new_column_name -> Returns a new column name from a list of existing columns and a column name
# - get_column_to_be_processed -> Returns the name of the column to process given the type of the "docs" element
# - regroup_data_series ->Wrapper to regroup identical data of a pd.Series before being processed
# - regroup_data_df -> Wrapper to regroup identical data of a pd.DataFrame before being processed
# - get_regex_match_words -> Returns a generic regex matching one or more words

import os
import re
import csv
import time
import copy
import errno
import ntpath
import numpy as np
import pandas as pd
from functools import wraps
from datetime import datetime
from typing import Callable, Union, List

# Get logger
import logging

logger = logging.getLogger(__name__)


def timer(function: Callable) -> Callable:
    '''Decorator to monitor the execution time of a function

    Args:
        function (func): Function to decorate
    Returns:
        function: Decorated function
    '''
    logger.debug('Calling utils.timer')

    # Get wrapper
    def wrapper(*args, **kwargs):
        '''Wrapper'''
        start_time = time.time()
        results = function(*args, **kwargs)
        end_time = time.time()
        logger.info(f"TIME-IT: exec time -> {round(end_time - start_time, 4)} seconds.")
        return results

    return wrapper


def data_agnostic(function: Callable, prefered_column: str = "docs", sep: str = ',') -> Callable:
    '''Decorator to manage type casting from and to pd.Series

    Supported types:
        - str ending by .csv /!\ Not advised /!\
        - str
        - list
        - np.ndarray
        - pd.Series
        - pd.DataFrame

    Args:
        function (func): Function to decorate
    Kwargs:
        prefered_column (str): Default column name to consider as the document container when working with a pandas dataframe or csv file (default: 'docs')
        sep: Separator to use if loading from a csv file
    Raises:
        ValueError: If the input is a path to an empty csv file
        FileNotFoundError: If the input is a path to a file that does not exist
    Returns:
        function: Decorated function
    '''
    @wraps(function)
    def wrapper(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame], *args, **kwargs):
        '''Wrapper

        Args:
            docs (?) : Arbitrary document list
        Returns:
            (?) : processed document list
        '''
        docs_type = get_docs_type(docs)

        if docs_type == 'file_path':
            logger.warning(
                f"{docs} is considered as a path to a csv file to load"
                + " This feature is not recommended."
                + " It is advised to directly work on the content of the csv file (by loading it beforehand)"
            )

            if not os.path.isfile(docs):
                raise FileNotFoundError(
                    f"File {docs} not found."
                    + f" If {docs} is a string, use [docs] or pd.Series(docs)"
                )
            if get_file_length(docs, sep=sep) == 0:
                raise ValueError(f'File {docs} is empty.')

            # Process
            logger.info(f"Loading {docs}. By default : first row is considered as the header.")
            df = pd.read_csv(docs, sep=sep)
            # If 'prefered_column' exists we use it, otherwise we fallback on the first column
            docs_column = prefered_column if prefered_column in df.columns else df.columns[0]
            logger.info(f"Selecting {docs_column} as the column to be processed.")
            docs_input = df[docs_column]
            results = function(docs_input, *args, **kwargs)
            assert results.shape[0] == docs_input.shape[0], f'The return value of  {function} must have a length of {docs_input.shape[0]}. Current length : {results.shape[0]}.'
            # Output format
            df[docs_column] = results
            # Saving the output in a new file
            saving_path = get_new_csv_name(docs)
            df.to_csv(saving_path, header=True, sep=sep, index=False)
            # Returns the path to the output file
            docs_output = saving_path

        elif docs_type == 'str':
            docs_input = pd.Series(docs)
            results = function(docs_input, *args, **kwargs)
            assert results.shape[0] == 1, f'The return value of  {function} must have a length of 1. Current length : {results.shape[0]}.'
            docs_output = results[0] if results[0] != np.nan else None

        elif docs_type == 'list':
            docs_input = pd.Series(docs)
            results = function(docs_input, *args, **kwargs)
            assert results.shape[0] == docs_input.shape[0], f'The return value of  {function} must have a length of {docs_input.shape[0]}. Current length : {results.shape[0]}.'
            docs_output = list(results.replace({np.nan: None}))

        elif docs_type == 'np.ndarray':
            docs_input = pd.Series(docs)
            results = function(docs_input, *args, **kwargs)
            assert results.shape[0] == docs_input.shape[0], f'The return value of  {function} must have a length of {docs_input.shape[0]}. Current length : {results.shape[0]}.'
            docs_output = np.array(results.replace({np.nan: None}))

        elif docs_type == 'pd.Series':
            docs_input = docs
            results = function(docs_input, *args, **kwargs)
            assert results.shape[0] == docs_input.shape[0], f'The return value of  {function} must have a length of {docs_input.shape[0]}. Current length : {results.shape[0]}.'
            docs_output = results

        elif docs_type == 'pd.DataFrame':
            # If prefered_column exists, we use it, otherwise we fall back on the first column
            docs_column = prefered_column if prefered_column in docs.columns else docs.columns[0]
            logger.info(f"Using {docs_column} as the column to be processed.")
            docs_input = docs[docs_column]
            results = function(docs_input, *args, **kwargs)
            assert results.shape[0] == docs_input.shape[0], f'The return value of  {function} must have a length of {docs_input.shape[0]}. Current length : {results.shape[0]}.'
            # Output format
            docs_output = copy.deepcopy(docs)
            docs_output[docs_column] = results

        return docs_output

    return wrapper


def data_agnostic_input(function: Callable, prefered_column: str = "docs", sep: str = ',') -> Callable:
    '''Decorator to manage type casting to pd.Series

    Supported types:
        - str ending by .csv -> chargement du csv en dataframe /!\ Unadvised /!\
        - str
        - list
        - np.ndarray
        - pd.Series
        - pd.DataFrame

    Args:
        function (func): Function to decorate
    Kwargs:
        prefered_column (str): Default column name to consider as the document container when working with a pandas dataframe or csv file (default: 'docs')
        sep: Separator to use if loading from a csv file
    Raises:
        ValueError: If the input is a path to an empty csv file
        FileNotFoundError: If the input is a path to a file that does not exist
    Returns:
        function: The decorated function
    '''

    @wraps(function)
    def wrapper(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame], *args, **kwargs) -> Union[str, list, np.ndarray, pd.Series, pd.DataFrame]:
        '''Wrapper

        Args:
            docs (?) : Arbitrary list of documents
        Returns:
            (?) : Processed documents
        '''
        docs_type = get_docs_type(docs)

        if docs_type == 'file_path':
            logger.warning(
                f"{docs} is considered as a path to a csv file to load"
                + " This feature is not recommended."
                + " It is advised to directly work on the content of the csv file (by loading it beforehand)"
            )

            if not os.path.isfile(docs):
                raise FileNotFoundError(
                    f"File {docs} not found."
                    + f" If {docs} is a string, use [docs] or pd.Series(docs)"
                )
            if get_file_length(docs, sep=sep) == 0:
                raise ValueError(f'File {docs} is empty.')
            logger.info(f"Loading {docs}. By default : first row is considered as the header.")
            df = pd.read_csv(docs, sep=sep)
            # IF prefered_column exists, we use it, otherwise we fall back on the first column
            docs_column = prefered_column if prefered_column in df.columns else df.columns[0]
            logger.info(f"Using {docs_column} as the column to be processed.")
            docs_input = df[docs_column]

        elif docs_type == 'str':
            docs_input = pd.Series(docs)

        elif docs_type == 'list':
            docs_input = pd.Series(docs)

        elif docs_type == 'np.ndarray':
            docs_input = pd.Series(docs)

        elif docs_type == 'pd.Series':
            docs_input = docs

        elif docs_type == 'pd.DataFrame':
            # If prefered_column exists, we use it, otherwise we fall back on the first column
            docs_column = prefered_column if prefered_column in docs.columns else docs.columns[0]
            logger.info(f"Using {docs_column} as the column to be processed.")
            docs_input = docs[docs_column]

        results = function(docs_input, *args, **kwargs)
        return results

    return wrapper


def get_docs_type(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame]) -> str:
    '''Returns the type of the documents within the list

    Args:
        docs (?) : Arbitrary document list
    Returns:
        (str): type of the docs list
    '''
    logger.debug('Calling utils.get_docs_type')
    if isinstance(docs, str) and docs.endswith('.csv'):
        docs_type = 'file_path'
    elif isinstance(docs, str):
        docs_type = 'str'
    elif isinstance(docs, list):
        docs_type = 'list'
    elif isinstance(docs, np.ndarray):
        docs_type = 'np.ndarray'
    elif isinstance(docs, pd.Series):
        docs_type = 'pd.Series'
    elif isinstance(docs, pd.DataFrame):
        docs_type = 'pd.DataFrame'
    else:
        raise TypeError('docs must be one of type [str, list, np.ndarray, pd.Series, pd.DataFrame]')
    return docs_type


def get_docs_length(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame],
                    first_row: str = 'header', sep: str = ',', nrows: int = 0) -> int:
    '''Returns the number of elements within a set of documents

    Args:
        docs (?): Arbitrary document list (Supported types : str ending by .csv, str, list, np.ndarray, pd.Series, pd.DataFrame)
    Kwargs:
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled
            -'header', 'data' or 'skip' (default : 'header')
        sep: Separator to use if loading from a csv file
        nrows (int) : When working with a pandas dataframe or csv file, specifies the maximum number of lines to read
            (default: 0 we take it all)
    Raises:
        ValueError: if nrows < 0
        ValueError: if first_row is not in ['header', 'data', 'skip']
    Returns:
        int: Number of documents
    '''
    logger.debug('Calling utils.get_docs_length')
    if nrows < 0:
        raise ValueError('"nrows" must be >= 0.')
    if first_row not in ['header', 'data', 'skip']:
        raise ValueError('"header" parameter must be "header", "data" or "skip".')

    # Process
    docs_type = get_docs_type(docs)

    if docs_type in ['pd.Series', 'pd.DataFrame']:
        return docs.shape[0]

    elif docs_type == 'file_path':
        file_length = get_file_length(docs, sep=sep)
        if first_row in ['header', 'skip']:
            file_length -= 1
        if nrows != 0:
            file_length = min(nrows, file_length)
        return file_length

    elif docs_type == 'str':
        return 1

    elif docs_type in ['list', 'np.ndarray']:
        return len(docs)


def get_file_length(filename: str, sep: str = ',') -> int:
    '''Returns... the file length !

    Args:
        filename (str): Path to the file
    Kwargs:
        sep: Separator to use if loading from a csv file
    Raises:
        FileNotFoundError: If the file... is not found !
    Returns:
        int: Number of rows within the file
    '''
    logger.debug('Calling utils.file_length')
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    # Check if csv file
    if filename.endswith('.csv'):
        # CSV files can contain "\n" within a data field, thus returning an incorrect number of rows
        with open(filename, 'r', encoding='utf-8') as f:
            # Using "if line" allows us to ignore empty lines
            return sum(1 for line in csv.reader(f, delimiter=sep) if line)
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            return sum(1 for line in f)


def get_new_csv_name(filename: str) -> str:
    '''Returns a new filename ("processed") from a given filename

    Args:
        filename (str): Path to the csv file (.csv)
    Raises:
        FileExistsError : If the file does not exist.
    Returns:
        (str): New filename
    '''
    logger.debug('Calling utils.get_new_csv_name')
    # Process
    # Get some paths
    file_path = os.path.abspath(filename)
    dir = os.path.dirname(os.path.abspath(filename))
    file_name = '.'.join(ntpath.basename(file_path).split('.')[:-1])
    # Get timestamp
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_file = os.path.join(dir, ''.join([file_name, '_', now, '.csv']))
    if not os.path.isfile(default_file):
        return default_file
    else:
        for i in range(2, 1000):
            new_file = os.path.join(dir, f"{file_name}_{now}_{i}.csv")
            if not os.path.isfile(new_file):
                return new_file
        raise FileExistsError('Can not find new file name (tried 1000 different names)')


def get_generator(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame], chunksize: int =0,
                  first_row: str = 'header', columns: List[str] = ['docs', 'tags'], sep: str = ',', nrows: int = 0, **pandas_args):
    '''Returns a generator given the type of document to process and the chunksize

    Args:
        docs (?): Arbitrary document list (Supported types : str ending by .csv, str, list, np.ndarray, pd.Series, pd.DataFrame)
    Kwargs:
        chunksize (int): if > 0 data is processed by chunks of chunksize size (by default : 0)
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled
            -'header', 'data' or 'skip' (default : 'header')
        columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use,
            if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing
            (default : ['docs', 'tags'])
        sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
        nrows (int) : When working with a pandas dataframe or csv file, specifies the maximum number of lines to read
            (default: 0 we take it all)
        pandas_args : When working with a pandas dataframe or csv file, specifies arguments to pass to pandas
    Raises:
        ValueError: If chunksize < 0
    Returns:
        (?): Data generator
    '''
    logger.debug('Calling utils.get_generator')
    if chunksize < 0:
        raise ValueError("Chunksize must be >= 0")

    docs_type = get_docs_type(docs)

    if docs_type == 'file_path':
        gen = get_df_generator_from_csv(docs, chunksize=chunksize, first_row=first_row,
                                        columns=columns, sep=sep, nrows=nrows, **pandas_args)

    elif docs_type == 'str':
        gen = (el for el in [docs])  # Generate only one element

    elif docs_type == 'list':
        if chunksize == 0 or chunksize >= len(docs):
            gen = (el for el in [docs])  # Generate only one element
        else:
            chunks_limits = [
                (i * chunksize, min((i + 1) * chunksize, len(docs)))
                for i in range(1 + ((len(docs) - 1) // chunksize))
            ]
            gen = (docs[chunk_limit[0] : chunk_limit[1]] for chunk_limit in chunks_limits)

    elif docs_type == 'np.ndarray':
        if chunksize == 0 or chunksize >= len(docs):
            gen = (el for el in [docs])  # Generate only one element
        else:
            chunks_limits = [
                (i * chunksize, min((i + 1) * chunksize, len(docs)))
                for i in range(1 + ((len(docs) - 1) // chunksize))
            ]
            gen = (
                docs[chunk_limit[0] : chunk_limit[1]] for chunk_limit in chunks_limits
            )

    elif docs_type in ('pd.Series', 'pd.DataFrame'):
        if chunksize == 0 or chunksize >= docs.shape[0]:
            gen = (el for el in [docs])  # Generate only one element
        else:
            chunks_limits = [
                (i * chunksize, min((i + 1) * chunksize, docs.shape[0]))
                for i in range(1 + ((docs.shape[0] - 1) // chunksize))
            ]
            gen = (
                docs.iloc[chunk_limit[0] : chunk_limit[1]]
                for chunk_limit in chunks_limits
            )

    return gen


def get_df_generator_from_csv(filename: str, chunksize: int = 0, first_row: str = 'header',
                              columns: List[str] = ['docs', 'tags'], sep: str = ',', nrows: int = 0, **pandas_args):
    '''Returns a dataFrame generator by chunk over a file
    If chunksize is 0 -> A one item generator is still returned

    Args:
        filename (str): Path to the csv file
    Kwargs:
        chunksize (int): If not 0 the pipeline is processed chunkwise and this parameter specifies the chunksize (default : 0)
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled
            -'header', 'data' or 'skip' (default : 'header')
        columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use,
            if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing
            (default : ['docs', 'tags'])
        sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
        nrows (int) : When working with a pandas dataframe or csv file, specifies the maximum number of lines to read
            (default: 0 we take it all)
        pandas_args : When working with a pandas dataframe or csv file, specifies arguments to pass to pandas
    Raises:
        ValueError: If 'first_row' is not in  ['header', 'data', 'skip']
        ValueError: If chunksize < 1
        ValueError: If nrows < 0
        ValueError: If the file is empty
        FileNotFoundError: If the file is not found
    Returns:
        (Dataframe): DataFrame Generator
    '''
    logger.debug('Calling utils.get_df_generator_from_csv')
    if chunksize < 0:
        raise ValueError('Chunksize must be >= 0.')
    if first_row not in ['header', 'data', 'skip']:
        raise ValueError("first_row must either be 'header', 'data' or 'skip'.")
    if nrows < 0:
        raise ValueError('nrows must be >= 0.')
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    # File length
    file_length = get_file_length(filename, sep=sep)
    if file_length == 0:
        raise ValueError(f"File {filename} os empty.")

    columns_to_use = get_columns_to_use(filename, first_row=first_row, columns=columns, sep=sep, file_length=file_length)

    # If there is only one row and it should be treated as the header, yields empty df
    if file_length == 1 and first_row == 'header':
        logger.warning('Empty dataframe.')
        yield pd.DataFrame([], columns=columns_to_use)
    else:
        # Manage chunks
        start_line = 0 if first_row == 'data' else 1
        end_line = file_length if nrows == 0 else min(file_length, nrows + start_line)
        # Loads everything in one pass
        if chunksize == 0:
            chunks_limits = [(start_line, end_line)]
        # Multiple pass loading (potentially only one if chunksize > end_line - start_line + 1)
        else:
            chunks_limits = [
                (
                    max(start_line, i * chunksize + start_line),
                    min((i + 1) * chunksize + start_line, end_line),
                )
                for i in range(1 + ((end_line - 1 - start_line) // chunksize))
            ]
        # Load data by chunks
        progression_index = 0
        progression_alerts_thresholds = list(range(0, 110, 10))
        for limits in chunks_limits:
            min_l = limits[0]
            max_l = limits[1]
            df = pd.read_csv(filename, encoding='utf-8', sep=sep, skiprows=min_l, nrows=max_l - min_l,
                             names=columns_to_use, header=None, **pandas_args)
            # Set correct index
            df.index = range(min_l - start_line, max_l - start_line, 1)
            # Print
            progression = max_l / end_line * 100
            if progression > progression_alerts_thresholds[progression_index]:
                logger.info(f"Loading file {filename.split('/')[-1]}: {round(progression, 2)} %")
                progression_index = [i for i, _ in enumerate(progression_alerts_thresholds) if _ <= progression][0]
            # yield
            yield df


def get_columns_to_use(filename: str, first_row: str = 'header', columns: List[str] = ['docs', 'tags'],
                       sep: str = ',', file_length: Union[int, None] = None) -> List[str]:
    '''Returns the names of the columns to use while loading a csv file

    Args:
        filename (str): Path to the csv file
    Kwargs:
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled
            -'header', 'data' or 'skip' (default : 'header')
        columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use,
            if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing
            (default : ['docs', 'tags'])
        sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
        file_length (int): Number of lines in the file (can speed up calculations by avoiding a call to get_file_length)
    Raises:
        ValueError: If 'first_row' is not in  ['header', 'data', 'skip']
        ValueError: If nrows < 0
        ValueError: If the file is empty
    Returns:
        (list<str>): Columns to use
    '''
    logger.debug('Calling utils.get_columns_to_use')
    if first_row not in ['header', 'data', 'skip']:
        raise ValueError("first_row must either be 'header', 'data' or 'skip'.")
    if file_length != None and file_length < 0:
        raise ValueError('file_length must be >= 0')
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    # Get file length if not passed to this function
    if file_length is None:
        file_length = get_file_length(filename, sep=sep)
    if file_length == 0:
        raise ValueError(f"File {filename} is empty.")

    # Open file to get first line
    with open(filename, encoding='utf8') as f:
        first_line = f.readline().replace('\n', '')

    # Get number of columns
    number_of_columns = len(first_line.split(sep))

    # Manages how the columns are handled
    if first_row == 'header':
        columns_to_use = first_line.split(sep)
    elif number_of_columns > len(columns):
        columns_to_use = columns + [str(i) for i in range(number_of_columns - len(columns))]
    elif number_of_columns < len(columns):
        columns_to_use = columns[:number_of_columns]
    else:
        columns_to_use = columns
    return columns_to_use


def get_new_column_name(df_column: list, processed_column: Union[str, int], suffix: str ='_processed') -> str:
    '''Returns a new column name from a list of existing columns and a column name

    Args:
        df_column (list) : List of the existing column names
        processed_column (str or int): Which columns to process
        suffix (str): Suffix to add at the end of the column name
    Raises:
        ValueError : If we fail to find a new column name
    Returns:
        (str): New column name
    '''
    logger.debug('Calling utils.get_new_column_name')
    default_column = ''.join([str(processed_column), suffix])
    if default_column not in df_column:
        return default_column
    else:
        for i in range(2, 1000):
            new_column = ''.join([str(processed_column), suffix, '_', str(i)])
            if new_column not in df_column:
                return new_column
        raise ValueError('Failed to find a new column name after 1000 tries')


def get_column_to_be_processed(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame], prefered_column='docs',
                               first_row: str = 'header', columns: List[str] = ['docs', 'tags'], sep: str = ',') -> Union[str, int]:
    '''Returns the name of the column to process given the type of the "docs" element

    Args:
        docs (?): Arbitrary document list (Supported types : str ending by .csv, str, list, np.ndarray, pd.Series, pd.DataFrame)
    Kwargs:
        prefered_column (str): Default column name to consider as the document container when working
            with a pandas dataframe or csv file (default: 'docs')
        first_row (str): When working with a pandas dataframe or csv file, specifies how the first line is handled
            -'header', 'data' or 'skip' (default : 'header')
        columns (list<str>) : When working with a pandas dataframe or csv file, specifies the columns to use,
            if first_row != 'header'. Truncate the data if there is too much columns & add some if they are missing
            (default : ['docs', 'tags'])
        sep (str): When working with a pandas dataframe or csv file, specifies the csv separator (default: ',')
    Returns:
        (str ou int): Index or name of the column to process
    '''
    logger.debug('Calling utils.get_column_to_be_processed')

    docs_type = get_docs_type(docs)

    if docs_type == 'pd.DataFrame':
        # We look for 'prefered_column', if it does not exist we fallback on the first column
        docs_column = prefered_column if prefered_column in docs.columns else docs.columns[0]
        logger.info(f"Using {docs_column} as a column to be processed.")
        return docs_column

    elif docs_type == 'file_path':
        available_columns = get_columns_to_use(docs, first_row=first_row, columns=columns, sep=sep)
        return available_columns[0] if prefered_column not in available_columns else prefered_column

    else:
        # Should not be used
        return prefered_column


def regroup_data_series(function: Callable, min_nb_data: int = 1000, prefix_text: Union[str, None] = None) -> Callable:
    '''Wrapper to regroup identical data of a pd.Series before being processed
    Can be used as a decorator

    Args:
        function (Callable): Function to which this wrapper is wrapped around /!\ intputs and outputs must be pd.Series, 1 to 1 /!\
    Kwargs:
        min_nb_data (int): Minimum number of rows within the document required to apply this wrapper (default : 1000)
        prefix_text (str): Prefix to add
    Returns:
        function: Decorated function
    '''
    # If no prefix is supplied we fall back to the default: __name__
    if prefix_text is None:
        try:
            prefix_text = f'{function.__name__} - '
        # Might get triggered with partials (functools.partial)
        except:
            prefix_text = f'{function} - '

    # Set wrapper
    @wraps(function)
    def wrapper(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame], *args, **kwargs) -> pd.Series:
        '''Wrapper

        Args:
            docs (pd.Series) : arbitrary pd.Series
        Returns:
            (pd.Series) : processed pd.Series
        '''
        logger.debug('Calling utils.regroup_data_series')

        init_len = len(docs)
        # If there is not enough data, the wrapper is discarded and the function returned as is
        if init_len < min_nb_data:
            return function(docs, *args, **kwargs)
        # If there is no duplicates in the data, the wrapper is discarded as well
        elif len(docs.unique()) == init_len:
            return function(docs, *args, **kwargs)
        init_name = docs.name
        init_index = docs.index
        # Put docs into a dataframe
        df = pd.DataFrame(docs)
        df.columns = ["input_data"]
        # Regroup same values together
        input_data = df["input_data"].dropna().drop_duplicates()
        logger.debug(f"{prefix_text} Reduced data to be processed by {100 * (df.shape[0] - len(input_data)) / df.shape[0]} % (grouped duplicated rows)")
        # Get output
        output_data = function(input_data, *args, **kwargs)
        # Assert lengths
        assert len(input_data) == len(output_data), f"regroup_data_series: Input data ({len(input_data)}) and Output data ({len(output_data)}) are not of equal length."
        # Prepare result
        docs_processed = pd.DataFrame({'input_data': input_data, 'output_data': output_data})
        # Merge with original dataset (use reset_index & set_index to keep original order)
        df = df.reset_index().merge(docs_processed, how='left', on='input_data').set_index('index')
        result = df["output_data"]
        # Check length & return
        assert init_len == len(result), f"regroup_data_series: Input data ({init_len}) and Output data ({len(result)}) are not of equal length."
        return result.rename(init_name).reindex(init_index)

    return wrapper


def regroup_data_df(function: Callable, columns_to_be_processed: Union[list, None] = None,
                    min_nb_data: int = 1000, prefix_text: Union[str, None] = None) -> Callable:
    '''Wrapper to regroup identical data from a dataframe before processing

    Args:
        function (function) : Function to decorate /!\ Inputs and outputs must be pd.Series, 1 to 1 /!\
    Kwargs:
        columns_to_be_processed (list): Columns to use during this processing /!\ only these columns are sent to 'function' /!\
        min_nb_data (int): Minimum number of rows within the document required to apply this wrapper (default : 1000)
        prefix_text (str): Prefix to add
    Raises:
        ValueError: If min_nb_data < 0
    Returns:
        function: Decorated function
    '''
    logger.debug('Calling utils.regroup_data_df')
    if min_nb_data <= 0:
        raise ValueError('"min_nb_data" must be > 0.')
    # If no prefix supplied we default to __name__
    if prefix_text == None:
        try:
            prefix_text = f'{function.__name__} - '
        # Might be triggered with partials (functools.partial)
        except:
            prefix_text = f'{function} - '

    # Set wrapper
    @wraps(function)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        '''Wrapper

        Args:
            df (pd.DataFrame) : DataFrame to process
        Raises:
            ValueError: If the list of columns to process can not be found within the DataFrame
        Returns:
            (pd.DataFrame) : Processed Dataframe
        '''
        # If there is not enough data, the wrapper is discarded and the function returned as is
        if df.shape[0] < min_nb_data:
            return function(df, *args, **kwargs)
        if columns_to_be_processed == None:
            final_columns_to_be_processed = list(df.columns)
        else:
            final_columns_to_be_processed = columns_to_be_processed
        # Checks whether the columns to process can be found
        for col in final_columns_to_be_processed:
            if col not in df.columns:
                raise ValueError(f"Column {col} can not be found within the supplied DataFrame.")
        # Init
        init_name = df.index.name
        init_cols = list(df.columns)
        # Regroup same values together
        # Only columns_to_be_processed are sent to the function
        input_df = (
            df[final_columns_to_be_processed]
            .dropna(subset=final_columns_to_be_processed)
            .drop_duplicates(subset=final_columns_to_be_processed)
        )
        logger.debug(f"{prefix_text} Reduced data to be processed by {100 * (df.shape[0] - input_df.shape[0]) / df.shape[0]} % (grouped duplicated rows)")
        # A unique index is added to the DataFrame to keep track of all the rows even if 'function' alters them
        join_column = get_new_column_name(list(df.columns), 'join_index', suffix='@regroup_data_df')
        join_values = [el for el in range(input_df.shape[0])]
        input_df[join_column] = join_values
        df = (
            df.reset_index()
            .merge(input_df[[join_column] + final_columns_to_be_processed], how='left', on=final_columns_to_be_processed)
            .set_index('index')
        )
        # Process function
        output_df = function(input_df, *args, **kwargs)
        # Assert lengths
        assert (input_df.shape[0] == output_df.shape[0]), f"regroup_data_df: number of inputs ({input_df.shape[0]}) and number of outputs ({output_df.shape[0]}) are not of the same length."
        # We remove some columns from the original dataframe if they are already present in the output
        df_cols = [col for col in df.columns if col not in output_df.columns or col == join_column]
        # Merge with original dataset (use reset_index & set_index to keep original order)
        df = df[df_cols].reset_index().merge(output_df, how='left', on=join_column).set_index('index')
        # Reorder cols & remove join col
        final_columns = init_cols + [col for col in list(output_df.columns) if col not in (init_cols + [join_column])]
        df = df[final_columns]
        # Rename index
        df.index.name = init_name
        # Return
        return df

    return wrapper

def get_regex_match_words(words: List[str], case_insensitive: bool = False,
                          accepted_char_ahead: str = '.?!,;:()"\'/<>=[]{}~*',
                          accepted_char_behind: str = '.?!,;:()"\'/<>=[]{}~*',
                          words_as_regex: bool = False) -> str:
    '''Returns a generic regex matching one or more words

    REGEX insights:
     - (?:^|(?<=[.?!,;:()\s])) : matches the beginning of a sentence, or whitespaces, commas, etc...
        - ?: : non capturing group
        - ^ : Start of the sentence
        - ?<= : look behind
        - [...] : Element lookup within a list

     - (?=[.?!,;:()\s]|$) : matches the end of a sentence, or whitespaces, commas, etc...
       - ?= : look ahead
       - [...] : Element lookup within a list
       - $ : end of the sentence

     - (?i) : Allows to be case insensitive

    We capture all the words to delete

    Args:
        words (list<str>): List of words/sentences to match
    Kwargs:
        case_insensitive (bool): Controls whether we should be case sensitive or not (default : False)
        accepted_char_ahead (str): List of accepted characters before these words/sentences
        accepted_char_behind (str): List of accepted characters after these words/sentences
        words_as_regex (bool): Controls whether submitted 'words' should be treated as regex (default: False)
    Returns:
        (str) : Newly generated regex
    '''
    logger.debug('Calling utils.get_regex_match_words')

    # Adding characters that are allowed after and before 'words'
    accepted_char_ahead = re.escape(accepted_char_ahead) + '\s'
    accepted_char_behind = re.escape(accepted_char_behind) + '\s'
    # Splits words up
    if words_as_regex:
        words = '|'.join(words)
    else:
        words = '|'.join([re.escape(word) for word in words])
    # Regex building
    regex = f'(?:^|(?<=[{accepted_char_ahead}]))({words})(?=[{accepted_char_behind}]|$)'
    if case_insensitive:
        regex = '(?i)' + regex
    return regex


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
