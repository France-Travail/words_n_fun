#!/usr/bin/env python3

## Functions linked to the lemmatizer
#  It used to be a custom made french lemmatizer but has been depcrecated and replaced by spacy
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
# - lemmatize -> Lemmatizes text


import sys
import requests
import pandas as pd
import simplejson as json
from words_n_fun import utils

# Get logger
import logging
logger = logging.getLogger(__name__)


# Spacy has to be installed for the lemmatizer to work
# Since it is an optional dependency, a warning is raised if a call to the lemmatizer is done
if 'spacy' in sys.modules:
    LEMMATIZER_AVAILABLE = True
    import spacy
    try:
        if not spacy.util.is_package("fr_core_news_sm"):
            logger.info("Downloading fr_core_news_sm")
            spacy.cli.download('fr_core_news_sm')
        spacy_model = spacy.load('fr_core_news_sm')
    except:
        spacy_model = None
        logger.error("Unexpected error downloading or loading spacy model fr_core_news_sm")
else:
    LEMMATIZER_AVAILABLE = False
    logger.warning("Spacy has not been found, lemmatizer features are not available.")
    logger.warning("To use it, you must install spacy. For instance: pip install words-n-fun[lemmatizer]")

@utils.data_agnostic
@utils.regroup_data_series
def lemmatize(docs: pd.Series) -> pd.Series:
    '''Text lemmatizer - spacy
    #   This feature uses the fr_core_news_sm from Spacy to process the text

    Args:
        docs (pd.Series): Documents to process

    Raises:
        ImportError : If spacy is not found
        Exception : fr_core_news_sm model is not found
    Returns:
        pd.Series: Modified documents
    '''
    if not LEMMATIZER_AVAILABLE:
        logger.error("Spacy has not been found, lemmatizer features are not available.")
        logger.error("To use it, you must install spacy. For instance: pip install words-n-fun[lemmatizer]")
        raise ImportError("Spacy has not been found, lemmatizer features are not available.")
    if not spacy.util.is_package("fr_core_news_sm"):
        logger.error("Unable to call spacy lemmatizer withouth spacy fr_core_news_sm model")
        raise Exception("Unable to call spacy lemmatizer withouth spacy fr_core_news_sm model")
    docs = (
        pd.Series(docs)
        .str.lower()
        .str.replace('\W', ' ')
        .str.replace(r"([0-9]+(\.[0-9]+)?)", r" \1 ")
        .str.replace('\s+', ' ', regex=True)
        .str.strip()
    )
    docs = list(docs.values)
    if len(docs) == 0:
        return None
    lemmatized = []
    for doc in docs:
        if isinstance(doc, str):
            content = spacy_model(doc)
            lem = ' '.join([token.lemma_ for token in content])
            lemmatized.append(lem)
        else:
            lemmatized.append(None)

    return pd.Series(lemmatized)



if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
