#!/usr/bin/env python3

## Functions to remove gendered synonyms: incredibly specific to the french language
# TODO : To remove?
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
# - remove_gender_synonyms -> Removes gendered synonyms
# - matching_words -> Male/Female token matching
# - update_synonyms_set -> Update the synonyms set


import re
import pandas as pd
from typing import Tuple

from words_n_fun import utils

# Get logger
import logging

logger = logging.getLogger(__name__)


SYNONYM_DICT = {
    'eur': ['euse', 'se', 'rice', 'e', 'eure'],
    "ien": ['ne', 'nne', 'e'],
    "ier": ['ere', 're', 'e'],
    "ial": ['e'],
    "l": ['le'],
    "nt": ['e'],
    "at": ['e'],
    "f": ['ve'],
    "o": ['a'],
    "er": ['ere', 'ère', 'euse'],
    "i": ['e'],
    "d": ['e'],
    "é": ['e'],
    "e": ['e'],
    "on": ['onne'],
    "is": ['ise'],
    "s": ['s']
}


@utils.data_agnostic
@utils.regroup_data_series
def remove_gender_synonyms(docs: pd.Series) -> pd.Series:
    '''Removes gendered synonyms

    Args:
        docs (pd.Series): Documents to process
    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling synonym_malefemale_replacement.getSynonyms')

    ## Preprocessing
    docs = docs.str.replace('(\s*)/(\s*)', '/')  # Removes whitespaces around "/"
    docs = docs.str.replace('(\s*)\((\s*)', '(')  # Removes potential whitespaces before "("
    docs = docs.str.replace('\)', ') ')  # Add a space after ")"

    ## Set match paterns
    parenthesis_pattern = r"([\w\-]+)\(([\w\-]+)\)()"  # Case :  serveur(se)
    slash_pattern = r"([\w\-]+)/([\w\-]+)(\([\w\-]+\))?"  # Case: serveur/serveuse and serveur/serveur(se)
    slash_pattern_BiWords = r"([\w\-]+\s[\w\-]+)/([\w\-]+\s[\w\-]+)()"  # Case: apprenti boucher/apprentie bouchere
    slash_pattern_TriWords = r"([\w\-]+\s[\w\-]+\s[\w\-]+)/([\w\-]+\s[\w\-]+\s[\w\-]+)()"  # Case:  aide apprenti boucher/aide apprentie bouchere
    match_pattern = [parenthesis_pattern, slash_pattern, slash_pattern_BiWords, slash_pattern_TriWords]

    ## Creating synonyms listings
    synonyms_set = {}
    for i in range(len(docs)):  # For every document
        for match in match_pattern:  # ... and for every match ...
            # Update the synonyms set
            text = docs.iloc[i]
            if isinstance(text, str):
                synonyms_set = update_synonyms_set(synonyms_set, re.findall(match, text), i)

    ## Update the  documents
    for i in range(len(docs)):  # For every document
        text = docs.iloc[i]

        if not isinstance(text, str):
            text = None
        else:
            # We process each pattern individually
            match_parenthesis_pattern = re.findall(parenthesis_pattern, text)
            match_slash_pattern = re.findall(slash_pattern, text)
            match_slash_pattern_BiWords = re.findall(slash_pattern_BiWords, text)
            match_slash_pattern_TriWords = re.findall(slash_pattern_TriWords, text)

            # Parenthesis
            if len(match_parenthesis_pattern) != 0:
                for (word1, word2, word3) in match_parenthesis_pattern:
                    if (word1, word2, word1) in synonyms_set:
                        text = text.replace(word1 + "(" + word2 + ")", word1)  # Case: serveur(se)
            # Slashes
            if len(match_slash_pattern) != 0:
                for (word1, word2, word3) in match_slash_pattern:
                    if word1 == word2:  # Case: serveur/serveur(se)
                        text = text.replace(word1 + "/" + word2 + "(" + word3 + ")", word1)
                        text = text.replace(word1 + "/" + word2, word1)
                    elif (word1, word2, word1) in synonyms_set:
                        text = text.replace(word1 + "/" + word2 + "(" + word3 + ")", word1)
                        text = text.replace(word1 + "/" + word2, word1)
            # Slashes BiWords
            if len(match_slash_pattern_BiWords) != 0:
                for (word1, word2, word3) in match_slash_pattern_BiWords:  # Case: apprenti boucher/apprentie bouchere
                    if (word1, word2, word1) in synonyms_set:
                        text = text.replace(word1 + "/" + word2, word1)
            # Slashes TriWords
            if len(match_slash_pattern_TriWords) != 0:
                for (word1, word2, word3) in match_slash_pattern_TriWords:
                    if (word1, word2, word1) in synonyms_set:
                        text = text.replace(word1 + "/" + word2, word1)
            docs.iloc[i] = text
    return docs


def matching_words(word1: str, word2: str) -> Tuple[str, str, str]:
    '''Male/Female token matching

    Args:
        word1 (str)
        word2 (str)

    Raises:
        TypeError : If word1 is empty
        TypeError : If word2 is empty

    Returns:
        tuple: (word1,word2,word1) if there is a match
        tuple: (word1,word2,"unknown") otherwise

    Rules :
        - If the end of word1 exists in the dictionnary and the end of word2 matches one of its values: eg serveur/se
        - If the end of word1 exists in the dictionnary and the end of word2 matches one of its values and the 4 first letters of both word1 and word 2 are identical: eg serveur/serveuse
        - If the end of word1 exists in the dictionnary and the end of word2 matches one of its values and the length of word2 is lesser or equal than the length of word1: eg conducteur(trice)
    '''

    if len(word1) == 0:
        raise TypeError('Word1 is empty.')
    if len(word2) == 0:
        raise TypeError('Word2 is empty.')

    combi = None
    for word1_end in SYNONYM_DICT.keys():
        for word2_end in SYNONYM_DICT[word1_end]:
            if combi is None:
                if word1[-len(word1_end) :] == word1_end:
                    if (
                        (word2 == word2_end)
                        or (word2[-len(word2_end) :] == word2_end and word1[:4] == word2[:4])
                        or (word2[-len(word2_end) :] == word2_end and len(word2) <= (len(word2_end) + 2))
                    ):
                        combi = (word1, word2, word1)
    if combi is None:
        combi = (word1, word2, "unknown")
    return combi


def update_synonyms_set(synonyms_set: dict, match: list, numligne: int) -> dict:
    '''Update the synonyms set

    Args:
        synonyms_set (dict)
        match (liste of tuples), eg : [('serveur', 'serveur', '(se)'), ('boucher', 'boucher', '(ere)')]
        numligne (int)
    Returns:
        dict : Synonyms set
    '''
    if len(match) == 0:
        pass
    else:
        combi = None
        for (word1, word2, word3) in match:
            # Example : Case serveur/serveur(se) #re.match("[^\d\s]+$", word1) : word1 is neither digit nor whitespace
            if word1 == word2 and re.match("^[\d\s]+$", word1) is None:
                combi = (word1, word2, word1)
            else:
                combi = matching_words(word1, word2)
            if combi is not None:
                if (list(combi)[2]) != "unknown" and combi not in synonyms_set.keys():
                    synonyms_set[combi] = numligne
    return synonyms_set


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
