#!/usr/bin/env python3

## Functions to split texts into sentences
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
# - split_sentences : Splits a text into sentences
# - split_sentences_df : Splits a set of texts into sentences


import re
import pandas as pd
from itertools import chain
from typing import Union, List

from words_n_fun import CustomTqdm as tqdm


tqdm.pandas()

# Get logger
import logging

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    '''Splits a text into sentences

    Args:
        text (str): Arbitrary text
    Returns:
        list<str> : List of sentences
    '''

    # We split around \n after singling them out and removing trailing spaces.
    text = re.compile("\s*\n+").sub('\n', text)
    text = re.compile("\s+$").sub('', text)
    text_list = text.split('\n')

    # Text between parenthesis are kept aside
    parentheses_matches = {}
    parentheses_regex = r"\(([^)]+)\)"
    k = 0  # Incrementation UID
    for i, sentence in enumerate(text_list):
        for match in re.compile(parentheses_regex).finditer(sentence):
            tmp_text = match.group()
            tmp_replace = f"UID_PE_PARENTHESES_{k}"
            parentheses_matches[tmp_replace] = tmp_text
            sentence = sentence.replace(tmp_text, tmp_replace, 1)
            text_list[i] = sentence
            k += 1

    # We split around ".", "!", "?" if they are followed by a whitespace or a newline
    # Special occurences ("i.e", "e.g", "M.") are skipped
    regex = "(?<!\s\w\.\w.\s)(?<!\sM\.\s)(?<=(?:\.|\?|\!)\s)"
    text_list = list(chain.from_iterable([re.compile(regex).split(_) for _ in text_list]))

    # Parenthesis texts are injected back in the list
    for i, elem in enumerate(text_list):
        for k in parentheses_matches.keys():
            elem = elem.replace(k, parentheses_matches[k])
        text_list[i] = elem

    return text_list


def split_sentences_df(df: pd.DataFrame, col: Union[str, int]) -> pd.DataFrame:
    '''Function to split several texts from a pandas DataFrame into sentences

    Args:
        df (pd.DataFrame): DataFrame containing the texts
        col (str ou int): Column name where the text is
    Returns:
        pd.DataFrame: New DataFrame with the text split into sentences
    '''
    df2 = df.copy()
    df2[col] = df2[col].progress_apply(split_sentences)
    return df2.explode(col).reset_index(drop=True)


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
