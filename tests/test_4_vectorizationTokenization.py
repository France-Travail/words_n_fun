#!/usr/bin/env python3

## Tests - unit test of vectorization_tokenization functions
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

# Libs unittest
import unittest
from unittest.mock import Mock
from unittest.mock import patch

# Utils libs
import os
import importlib
import numpy as np
import pandas as pd
from words_n_fun import utils
from words_n_fun.preprocessing import vectorization_tokenization

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class VectorizationTokenizationTests(unittest.TestCase):
    '''Main class to test all functions in vectorization_tokenization.py.'''

    # Mock du decorateur DataAgnostic (on le bypass pour les tests)
    default_decorator = lambda f: f
    utils.data_agnostic = default_decorator
    utils.data_agnostic_input = default_decorator
    # Reload de la librairie vectorization_tokenization (pour application du decorateur par defaut)
    importlib.reload(vectorization_tokenization)


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_split_text_into_tokens(self):
        '''Testing function vectorization_tokenization.split_text_into_tokens'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        words_sequences = [[['Chauffeur(se)', 'accompagnateur(trice)', 'pers'], ['pers', 'à', 'mob'], ['mob', '-', '5']],
                            [['Je', 'maîtrise', '12'], ['le', 'Permis', 'B'], ['12', 'langages', 'informatiques']],
                            [['Coordinateur', "d'Equipe", "d'Action"]],
                            None,
                            None]
        words_next_items = [['à', '-', 'ans'], ['langages', 'B', 'dont'], ['Territoriale'], None, None]
        char_sequences = [['Ch', 'au', 'ff', 'eu'], ['Je', ' m', 'aî', 'tr'], ['Co', 'or', 'di', 'na'], None, None]
        char_next_items = [['a', 'f', 'e', 'r'], [' ', 'a', 't', 'i'], ['o', 'd', 'n', 't'], None, None]

        # Vérification du fonctionnement type
        sequences, next_items = vectorization_tokenization.split_text_into_tokens(pd.Series(docs), nbech=3, seq_size=3, step=2, granularity="word")
        self.assertEqual(list(sequences.replace({np.nan:None})), words_sequences)
        self.assertEqual(list(next_items.replace({np.nan:None})), words_next_items)
        sequences, next_items = vectorization_tokenization.split_text_into_tokens(pd.Series(docs), nbech=4, seq_size=2, step=2, granularity="char")
        self.assertEqual(list(sequences.replace({np.nan:None})), char_sequences)
        self.assertEqual(list(next_items.replace({np.nan:None})), char_next_items)

        with self.assertRaises(ValueError):
            vectorization_tokenization.split_text_into_tokens(pd.Series(docs), nbech=-1)
        with self.assertRaises(ValueError):
            vectorization_tokenization.split_text_into_tokens(pd.Series(docs), seq_size=-1)
        with self.assertRaises(ValueError):
            vectorization_tokenization.split_text_into_tokens(pd.Series(docs), step=-1)
        with self.assertRaises(ValueError):
            vectorization_tokenization.split_text_into_tokens(pd.Series(docs), granularity='test')


# Execution des tests
if __name__ == '__main__':
    unittest.main()
