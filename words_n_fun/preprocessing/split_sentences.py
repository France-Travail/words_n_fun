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
import logging
import functools
import pandas as pd
from typing import Union, List, Tuple

from words_n_fun import CustomTqdm as tqdm

tqdm.pandas()

# Get logger
logger = logging.getLogger(__name__)

# REGEX USED
# Join several \n
re_multiline = re.compile(r"\s*\n+")
# Remove last spaces
re_lastspace = re.compile(r"\s+$")
# Detect parenthesis
re_parenthesis = re.compile(r"\([^())]+\)")
# Regex to split punctuation ignoring classicar acronims
re_specialsplit_v1 = re.compile("(?<!\s\w\.\w\.\s)(?<!\sM\.\s)(?<=(?:\.|\?|!)\s)")
re_specialsplit_v2 = re.compile(
    r"(?<!^\w.\s)(?<!\s\w.\s)(?<!\w\.\w\.\s)(?<!\bMme\.\s)(?<!\bDr\.\s)(?<=(?:\.|\?|\!)\s)"
)


def parenthesis_to_key(line: str) -> Tuple[str, List[Tuple]]:
    """Replace parenthesis content for UID_PE_PARENTHESES_{k}
    We do iteratively to avoid problems with imbricated parenthesis
    """
    k = 0
    matches = []
    while match := re_parenthesis.search(line):
        parenthesis = match.group()
        replacement = f"UID_PE_PARENTHESES_{k}"
        matches.append((replacement, parenthesis))
        line = line.replace(parenthesis, replacement, 1)
        k += 1
    return line, matches


def key_to_parenthesis(lines: List[str], key_parenthesis: List[Tuple]) -> List[str]:
    """apply parenthesis to a group of lines"""
    # verify there are parenthesis to replace
    if not key_parenthesis:
        return lines

    for i, line in enumerate(lines):
        # do replacement iteratively
        while "UID_PE_PARENTHESES" in line:
            # reversed because the last one should be replaced first
            for key, value in reversed(key_parenthesis):
                line = line.replace(key, value, 1)
        lines[i] = line
    return lines


def split_sentences(text: str, version: int = 1) -> List[str]:
    """Splits a text into sentences
        - Parenthesis are removed before split so parenthesis are not splitted
        - Split by \n and then punctuation: '. ', '? ','! '
        - Classic acronyms are avoided M., Mme.,Dr,  A.C.M.E or A. C. M. E.

    Args:
        text (str): Arbitrary text
        version (int): 1 = original version, 2 = extra rules are added
    Returns:
        list<str> : List of sentences
    """
    if version not in [1, 2]:
        raise ValueError(" split_sentences version must be 1 or 2")
    re_specialsplit = re_specialsplit_v2 if version == 2 else re_specialsplit_v1
    # We split around \n after singling them out and removing trailing spaces.
    text = re_multiline.sub("\n", text)
    text = re_lastspace.sub("", text)
    text_lines = text.split("\n")

    text_list = []

    for i, sentence in enumerate(text_lines):
        # replace parenthesis
        sentence, key_paren = parenthesis_to_key(sentence)

        # split sentences
        # We split around ".", "!", "?" if they are followed by a whitespace or a newline
        # Special occurences ("i.e", "e.g", "M.") are skipped
        splits = re_specialsplit.split(sentence)

        # reapply parenthesis
        sentences = key_to_parenthesis(splits, key_paren)

        # combine all sentences
        text_list.extend(sentences)

    return text_list


def split_sentences_df(
    df: pd.DataFrame, col: Union[str, int], use_tqdm: bool = False, version: int = 1
) -> pd.DataFrame:
    """Function to split several texts from a pandas DataFrame into sentences

    Args:
        df (pd.DataFrame): DataFrame containing the texts
        col (str ou int): Column name where the text is
    Returns:
        pd.DataFrame: New DataFrame with the text split into sentences
    """
    func = functools.partial(split_sentences, version=version)
    df2 = df.copy()
    if use_tqdm:
        df2[col] = df2[col].progress_apply(func)
    else:
        df2[col] = df2[col].apply(func)

    return df2.explode(col).reset_index(drop=True)


if __name__ == "__main__":
    logger.error(
        "This script is not stand alone but belongs to a package that has to be imported."
    )
