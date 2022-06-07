#!/usr/bin/env python3

## Test - unit test of lemmatizer (deprecated) functions
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
try:
    import spacy
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping all lemmatizer tests as spacy can't be imported.")
import importlib
import numpy as np
import pandas as pd
from words_n_fun import utils
from words_n_fun.preprocessing import lemmatizer

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class LemmatizerTests(unittest.TestCase):
    '''Main class to test all functions in lemmatizer.py.'''

    # Mock du decorateur DataAgnostic (on le bypass pour les tests)
    default_decorator = lambda f: f
    utils.data_agnostic = default_decorator
    utils.data_agnostic_input = default_decorator
    # Reload de la librairie lemmatizer (pour application du decorateur par defaut)
    importlib.reload(lemmatizer)


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_lemmatize(self):
        '''Testing function lemmatizer.lemmatize'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "^Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_lemmatized = ['chauffeur se accompagnateur trice pers à mob 5 an de expérience', 'je maîtris 12 langage informatique dont le c j avoir le permis b', 'chauffeur se accompagnateur trice pers à mob 5 an de expérience', 'coordinateur d equipe d action territorial', None, None]

        if spacy.util.is_package("fr_core_news_sm"):
            # Vérification du fonctionnement type
            self.assertEqual(list(lemmatizer.lemmatize(pd.Series(docs)).replace({np.nan:None})), docs_lemmatized)
            self.assertEqual(list(lemmatizer.lemmatize(pd.Series(docs*100)).replace({np.nan:None})), docs_lemmatized*100)


# Execution des tests
if __name__ == '__main__':
    unittest.main()
