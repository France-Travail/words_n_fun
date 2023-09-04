#!/usr/bin/env python3

## Main funtions of the preprocessing API
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
# - notnull -> Replaces null values by an empty character
# - remove_non_string -> Replaces all non strings by an empty character
# - get_true_spaces -> Replaces all whitespaces by a single space
# - to_lower -> Transforms the string to lower case
# - pe_matching -> Specific one-to-one tokens replacements
# - remove_punct -> Replaces all non alpha-numeric characters by spaces
# - trim_string -> Trims spaces at the beginning and ending of the string (multiple spaces become one)
# - remove_numeric -> Replaces numeric strings by a space
# - remove_stopwords -> Removes stopwords
# - remove_accents -> Removes all accents and special characters (ç..)
# - remove_gender_synonyms -> [French] Removes gendered synonyms
# - lemmatize -> Lemmatizes the document
# - stemmatize -> Stemmatizes the words of the document
# - add_point -> Adds a dot at the end of each line
# - deal_with_specific_characters -> Ads spaces before and after some punctuations (, : ; .)
# - replace_urls -> Replaces URLs by spaces
# - remove_words -> Replaces words from a list
# - fix_text -> Fixes numerous inconsistencies within a text (via ftfy)




import ftfy
import logging
import unicodedata
import numpy as np
import pandas as pd
from typing import List, Union
from nltk.stem.snowball import FrenchStemmer

from words_n_fun import CustomTqdm as tqdm
from words_n_fun import utils
from words_n_fun.preprocessing import (lemmatizer, stopwords,
                                       synonym_malefemale_replacement)

tqdm.pandas()

# Get logger
logger = logging.getLogger(__name__)



def impl_notnull(docs: pd.Series) -> pd.Series:
    '''Replaces null values by an empty character

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.notnull')
    return docs.fillna('')

@utils.data_agnostic
def notnull(docs: pd.Series) -> pd.Series:
    '''Replaces null values by an empty character

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''    
    logger.debug('Calling basic.remove_non_string')
    return impl_notnull(docs)

def impl_remove_non_string(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Replaces all non strings by an empty character

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    if use_tqdm:
        return docs.progress_apply(lambda x: x if isinstance(x, str) else '')
    else: 
        return docs.apply(lambda x: x if isinstance(x, str) else '')
        

@utils.data_agnostic
def remove_non_string(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    return impl_remove_non_string(docs, use_tqdm)

@utils.regroup_data_series
def impl_get_true_spaces(docs: pd.Series) -> pd.Series:
    '''Replaces all whitespaces by a single space

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    return docs.str.replace(r'\s', ' ', regex=True)

@utils.data_agnostic
def get_true_spaces(docs: pd.Series) -> pd.Series:
    '''Replaces all whitespaces by a single space

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.get_true_spaces')
    return impl_get_true_spaces(docs)

@utils.regroup_data_series
def impl_to_lower(docs: pd.Series, threshold_nb_chars: int = 0, use_tqdm: bool = False) -> pd.Series:
    '''Transforms the string to lower case

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        threshold_nb_chars (int): Minimum number of characters for a token to be transformed to lowercase (def=0).

    Returns:
        pd.Series: Modified documents
    '''
    if threshold_nb_chars > 1:
        logger.debug(f"Applying lower case transform for tokens of at least {threshold_nb_chars} chars.")
        if use_tqdm:
            return docs.progress_apply(lambda x: " ".join(x.lower() if len(x) >= threshold_nb_chars else x for x in x.split((" "))) if isinstance(x, str) else None)
        else:
            return docs.apply(lambda x: " ".join(x.lower() if len(x) >= threshold_nb_chars else x for x in x.split((" "))) if isinstance(x, str) else None)
    else:
        return docs.str.lower()
    
@utils.data_agnostic
def to_lower(docs: pd.Series, threshold_nb_chars: int = 0, use_tqdm: bool = False) -> pd.Series:
    '''Transforms the string to lower case

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        threshold_nb_chars (int): Minimum number of characters for a token to be transformed to lowercase (def=0).

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.to_lower')
    return impl_to_lower(docs, threshold_nb_chars, use_tqdm)

@utils.regroup_data_series
def impl_pe_matching(docs: pd.Series) -> pd.Series:
    '''Specific one-to-one tokens replacements
    For instance 'Permis b' => 'Permisb'

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.pe_matching')
    # One can add more rules here
    regex = utils.get_regex_match_words(['(permis)\s+(b)'], case_insensitive=True, words_as_regex=True)
    docs = docs.str.replace(regex, r'\2\3', regex=True)
    return docs


@utils.data_agnostic
def pe_matching(docs: pd.Series) -> pd.Series:
    '''Specific one-to-one tokens replacements
    For instance 'Permis b' => 'Permisb'

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.pe_matching')
    return impl_pe_matching(docs)


def impl_remove_punct(docs: pd.Series, del_parenthesis: bool = True, replacement_char: str = ' ') -> pd.Series:
    '''Replaces all non alpha-numeric characters by spaces

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        del_parenthesis (bool): Whether parenthesis and slashes are removed (def= True)
        replacement_char (str): Replacement character (def= ' ')
    Returns:
        pd.Series: Modified documents
    '''
    if not del_parenthesis:
        regex = r"[^\w\s\(\)\/]|_"
    else:
        regex = r"[^\w\s]|_"
    return docs.str.replace(regex, replacement_char, regex=True)

@utils.data_agnostic
@utils.regroup_data_series
def remove_punct(docs: pd.Series, del_parenthesis: bool = True, replacement_char: str = ' ') -> pd.Series:
    '''Replaces all non alpha-numeric characters by spaces

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        del_parenthesis (bool): Whether parenthesis and slashes are removed (def= True)
        replacement_char (str): Replacement character (def= ' ')
    Returns:
        pd.Series: Modified documents
    '''
    return impl_remove_punct(docs, del_parenthesis, replacement_char)


def impl_trim_string(docs: pd.Series) -> pd.Series:
    '''Trims spaces: multiple spaces become one

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    # TODO: better way ?
    docs = docs.str.replace(r'[\t\f\v ]{2,}', ' ', regex=True)
    docs = impl_remove_leading_and_ending_spaces(docs)
    return docs


@utils.data_agnostic
@utils.regroup_data_series
def trim_string(docs: pd.Series) -> pd.Series:
    '''Trims spaces: multiple spaces become one

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.trim_string')
    return impl_trim_string(docs)

@utils.regroup_data_series
def impl_remove_leading_and_ending_spaces(docs: pd.Series) -> pd.Series:
    '''Removes leading and trailing spaces

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    return docs.str.replace(r'(^(\s)+)|((\s)+$)', '', regex=True)



@utils.data_agnostic
def remove_leading_and_ending_spaces(docs: pd.Series) -> pd.Series:
    '''Removes leading and trailing spaces

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.remove_leading_and_ending_spaces')
    return impl_remove_leading_and_ending_spaces(docs)

@utils.regroup_data_series
def impl_remove_numeric(docs: pd.Series, replacement_char: str = ' ') -> pd.Series:
    '''Replaces numeric strings by a space

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        replacement_char (str): Replacement character (def= ' ')
    Returns:
        pd.Series: Modified documents
    '''
    return docs.str.replace(r'([0-9]+)', replacement_char, regex=True)


@utils.data_agnostic
def remove_numeric(docs: pd.Series, replacement_char: str = ' ') -> pd.Series:
    '''Replaces numeric strings by a space

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        replacement_char (str): Replacement character (def= ' ')
    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.remove_numeric')
    return impl_remove_numeric(docs, replacement_char)

def impl_remove_stopwords(docs: pd.Series, opt: str = 'all', set_to_add: Union[list, None] = None,
                     set_to_remove: Union[list, None] = None) -> pd.Series:
    '''Removes stopwords

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        opt (str): Specifies which stopwords set are used, cf stopwords.py (def='all')
        set_to_add (list): List of words to append to the stopwords list
        set_to_remove (list): List of words to remove from the stopwords list
    Returns:
        pd.Series: Modified documents
    '''
    # stopwords.remove_stopwords use data_agnostic and regroup_data_series wrappers already
    return stopwords.remove_stopwords(docs, opt=opt, set_to_add=set_to_add, set_to_remove=set_to_remove)


# called function already with wrappers
def remove_stopwords(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame], opt: str = 'all', set_to_add: Union[list, None] = None,
                     set_to_remove: Union[list, None] = None) -> Union[str, list, np.ndarray, pd.Series, pd.DataFrame]:
    '''Removes stopwords

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        opt (str): Specifies which stopwords set are used, cf stopwords.py (def='all')
        set_to_add (list): List of words to append to the stopwords list
        set_to_remove (list): List of words to remove from the stopwords list
    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.remove_stopwords')
    return impl_remove_stopwords(docs, opt=opt, set_to_add=set_to_add, set_to_remove=set_to_remove)

@utils.regroup_data_series
def impl_remove_accents(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Removes all accents and special characters (ç..)

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        use_tqdm (bool): Whether tqdm should be used (default: True)

    Returns:
        pd.Series: Modified documents
    '''
    if use_tqdm:
        return docs.progress_apply(lambda x: ''.join((c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn')) if isinstance(x, str) else None)
    else:
        return docs.apply(lambda x: ''.join((c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn')) if isinstance(x, str) else None)


@utils.data_agnostic
def remove_accents(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Removes all accents and special characters (ç..)

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        use_tqdm (bool): Whether tqdm should be used (default: True)

    Returns:
        pd.Series: Modified documents
    '''
    return impl_remove_accents(docs, use_tqdm)

def impl_remove_gender_synonyms(docs: pd.Series) -> pd.Series:
    '''[French] Removes gendered synonyms
    # Find occurences such as "male version / female version" (eg: Coiffeur / Coiffeuse)
    # By convention, the male version is kept (in accordance with the lemmatizer)

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    # synonym_malefemale_replacement.remove_gender_synonyms uses data_agnostic and regroup_data_series wrappers already
    return synonym_malefemale_replacement.remove_gender_synonyms(docs)


# wrappers in the main function
def remove_gender_synonyms(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame]) -> Union[str, list, np.ndarray, pd.Series, pd.DataFrame]:
    '''[French] Removes gendered synonyms
    # Find occurences such as "male version / female version" (eg: Coiffeur / Coiffeuse)
    # By convention, the male version is kept (in accordance with the lemmatizer)

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.remove_gender_synonyms')
    return impl_remove_gender_synonyms(docs)

# lemmatizer.lemmatize has already wrappers
def impl_lemmatize(docs: pd.Series) -> pd.Series:
    '''Lemmatizes the documents
    Appel à une API externe

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.lemmatize')
    # Process
    return lemmatizer.lemmatize(docs)

# lemmatizer.lemmatize has already wrappers
def lemmatize(docs: Union[str, list, np.ndarray, pd.Series, pd.DataFrame]) -> Union[str, list, np.ndarray, pd.Series, pd.DataFrame]:
    '''Lemmatizes the documents
    Appel à une API externe

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.lemmatize')
    # Process
    return impl_lemmatize(docs)

@utils.regroup_data_series
def impl_stemmatize(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Stemmatizes words in the documents

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    stemmer = FrenchStemmer()
    if use_tqdm:
        return docs.progress_apply(lambda x: " ".join(stemmer.stem(x) for x in x.split(' ')) if isinstance(x, str) else None)
    else:
        return docs.apply(lambda x: " ".join(stemmer.stem(x) for x in x.split(' ')) if isinstance(x, str) else None)



@utils.data_agnostic
def stemmatize(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Stemmatizes words in the documents

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.stemmatize')
    logger.warning('Calling the FRENCH stemmer')
    return impl_stemmatize(docs)

@utils.regroup_data_series
def impl_add_point(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Adds a dot at the end of each line

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    if use_tqdm:
        return docs.progress_apply(lambda x: (x + '.' if not x.endswith('.') else x) if isinstance(x, str) else None)
    else:
        return docs.apply(lambda x: (x + '.' if not x.endswith('.') else x) if isinstance(x, str) else None)


@utils.data_agnostic
def add_point(docs: pd.Series, use_tqdm: bool = False) -> pd.Series:
    '''Adds a dot at the end of each line

    Args:
        docs (pd.Series): Documents to process

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.add_point')
    return impl_add_point(docs, use_tqdm)

@utils.regroup_data_series
def impl_deal_with_specific_characters(docs: pd.Series) -> pd.Series:
    '''Adds spaces before and after some punctuations (, : ; .)

    Args:
      docs (pd.Series): Documents to process
    Returns:
      pd.Series: Modified documents
    '''
    return docs.str.replace(r"(\s)?([',.;:])(\s)?", r' \2 ', regex=True)


@utils.data_agnostic
def deal_with_specific_characters(docs: pd.Series) -> pd.Series:
    '''Adds spaces before and after some punctuations (, : ; .)

    Args:
      docs (pd.Series): Documents to process
    Returns:
      pd.Series: Modified documents
    '''
    logger.debug('Calling basic.deal_with_specific_characters')
    return impl_deal_with_specific_characters(docs)

@utils.regroup_data_series
def impl_replace_urls(docs: pd.Series, replacement_char: str = ' ', replace_with_domain: bool = False) -> pd.Series:
    '''Replaces URLs by either a str or the url domain

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        replacement_char (str): Replacement character (def= ' ')
        replace_with_domain (bool): Replacement_char is overriden and the url is replaced by its domain (def= False)
    Returns:
        pd.Series: Modified documents
    '''
    # based on : https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
    regex = r'(?i)(?<!\w|/)(((http|ftp|https):\/\/)*(www\.|ftp\.)+|((http|ftp|https):\/\/)+(www\.|ftp\.)*)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])?'
    if not replace_with_domain:
        return docs.str.replace(regex, replacement_char, regex=True)
    else:
        return docs.str.replace(regex, r' \8 ', regex=True)


@utils.data_agnostic
def replace_urls(docs: pd.Series, replacement_char: str = ' ', replace_with_domain: bool = False) -> pd.Series:
    '''Replaces URLs by either a str or the url domain

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        replacement_char (str): Replacement character (def= ' ')
        replace_with_domain (bool): Replacement_char is overriden and the url is replaced by its domain (def= False)
    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.replace_urls')
    return impl_replace_urls(docs, replacement_char, replace_with_domain)
    
@utils.regroup_data_series
def impl_remove_words(docs: pd.Series, words_to_remove: List[str], case_insensitive=False) -> pd.Series:
    '''Function to remove words from a list

    Args:
        docs (pd.Series): Documents to process
        words_to_remove (list<str>): List of words to remove
    Kwargs:
        case_insensitive (bool): Whether the replacement is case sensitive (defaut : False)
    Returns:
        pd.Series: Modified documents
    '''
    regex = utils.get_regex_match_words(words_to_remove, case_insensitive=case_insensitive)
    return docs.str.replace(regex, '', regex=True)
    

@utils.data_agnostic
def remove_words(docs: pd.Series, words_to_remove: List[str], case_insensitive=False) -> pd.Series:
    '''Function to remove words from a list

    Args:
        docs (pd.Series): Documents to process
        words_to_remove (list<str>): List of words to remove
    Kwargs:
        case_insensitive (bool): Whether the replacement is case sensitive (defaut : False)
    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling utils.remove_words')
    return impl_remove_words(docs, words_to_remove, case_insensitive)

@utils.regroup_data_series
def impl_fix_text(docs: pd.Series, use_tqdm: bool = False,  **ftfy_kwargs) -> pd.Series:
    '''Fixes numerous inconsistencies within a text (via ftfy)
       By default:
        - Removes some Linux instructions
        - Fixes encoding
        - Fixes HTML entities
        - Fixes some quotes
        - Replaces tied letter (e.g. œ)
        - Replaces characters larger than normal
        - Fixes line breaks (LF)
        - Fixes UTF-16 "surrogates"
        - Removes "control characters"
        - Removes "Byte-Order Mark"
        - NFC Normalization (accents)

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        ftfy_kwargs (dict): Kwargs forwarded to ftfy

    Returns:
        pd.Series: Modified documents
    '''
    if use_tqdm:
        return docs.progress_apply(lambda x: ftfy.fix_text(x, **ftfy_kwargs) if isinstance(x, str) else None)
    else:
        return docs.apply(lambda x: ftfy.fix_text(x, **ftfy_kwargs) if isinstance(x, str) else None)

@utils.data_agnostic
def fix_text(docs: pd.Series, use_tqdm: bool = False, **ftfy_kwargs) -> pd.Series:
    '''Fixes numerous inconsistencies within a text (via ftfy)
       By default:
        - Removes some Linux instructions
        - Fixes encoding
        - Fixes HTML entities
        - Fixes some quotes
        - Replaces tied letter (e.g. œ)
        - Replaces characters larger than normal
        - Fixes line breaks (LF)
        - Fixes UTF-16 "surrogates"
        - Removes "control characters"
        - Removes "Byte-Order Mark"
        - NFC Normalization (accents)

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        ftfy_kwargs (dict): Kwargs forwarded to ftfy

    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling basic.fix_text')
    return impl_fix_text(docs, use_tqdm, **ftfy_kwargs)

if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
