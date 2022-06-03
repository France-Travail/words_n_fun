#!/usr/bin/env python3

## Vectorization and tokenization functions
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
# - split_text_into_tokens -> Splits an input text into seq_size tokens (word or char) with at most nbech tokens

from typing import Tuple
import pandas as pd
from words_n_fun import utils

# Get logger
import logging

logger = logging.getLogger(__name__)


@utils.data_agnostic
def split_text_into_tokens(docs: pd.Series, nbech: int = 10, seq_size: int = 3, step: int = 1,
                           granularity: str = "word") -> Tuple[pd.Series, pd.Series]:
    '''Split an input text into seq_size tokens (word or char) with at most nbech tokens

    Args:
      docs (pd.Series): Documents to modify
    Kwargs:
      nbech (int): Max number of  sequences (default=10)
      seq_size (int): Number of tokens per sequence (default=3)
      step (int): Overlap between sequences (default=1)
      granularity (str): Tokenization granularity ('word' or 'char')
    Raises:
      ValueError: If nbech is not > 0
      ValueError: If seq_size is not > 0
      ValueError: If step is not > 0
      ValueError: If granularity is neither word nor char
    Returns:
      pd.Series: List of sequences generated per document
      pd.Series: List of "next item" generated per document
    '''
    logger.debug('Calling fonction basic.split_text_into_tokens')
    if nbech <= 0:
        raise ValueError("nbech must be > 0")
    if seq_size <= 0:
        raise ValueError("seq_size must be > 0")
    if step <= 0:
        raise ValueError("Step must be > 0")
    if granularity not in ['word', 'char']:
        raise ValueError("granularity must either be word or char")

    sequences = pd.Series([None] * docs.shape[0])
    next_items = pd.Series([None] * docs.shape[0])
    for i in range(docs.shape[0]):
        text = docs.iloc[i]
        sequence = []
        next_item = []

        if not isinstance(text, str):
            continue

        if granularity == "char":
            for j in range(0, len(text) - seq_size, step):
                if "." not in text[j : j + seq_size]:
                    sequence.append(text[j : j + seq_size])
                    next_item.append(text[j + seq_size])
                if len(sequence) >= nbech:
                    break

        elif granularity == "word":
            words = text.split()
            for j in range(0, len(words) - seq_size, step):
                sequence.append(words[j : j + seq_size])
                next_item.append(words[j + seq_size])
                if len(sequence) >= nbech:
                    break
                if (len(words) - seq_size - 1) % step != 0:
                    sequence.append(words[len(words) - seq_size : len(words)])
                    next_item.append(words[len(words) - 1])
        sequences.iloc[i] = sequence
        next_items.iloc[i] = next_item
    return sequences, next_items


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
