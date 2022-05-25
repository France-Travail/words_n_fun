#!/usr/bin/env python3

## Test - unit test of stopwords functions
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
from pe_semantic import utils
from pe_semantic.preprocessing import stopwords

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class StopwordsTests(unittest.TestCase):
    '''Main class to test all functions in stopwords.py.'''

    # Mock du decorateur DataAgnostic (on le bypass pour les tests)
    default_decorator = lambda f: f
    utils.data_agnostic = default_decorator
    utils.data_agnostic_input = default_decorator
    # Reload de la librairie stopwords (pour application du decorateur par defaut)
    importlib.reload(stopwords)


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_remove_stopwords(self):
        '''Test de la fonction stopwords.remove_stopwords'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_stopwords_removed = ["Chauffeur()  accompagnateur(trice) pers  mob - 5 ans  expérience.", "Je maîtrise 12 langages informatiques   C & '  Permis B", "Coordinateur 'Equipe 'Action Territoriale ", None, None]
        docs_stopwords_removed_default = ["Chauffeur()  accompagnateur(trice) pers  mob - 5 ans  expérience.", "Je maîtrise 12 langages informatiques   C & '  Permis B", "Coordinateur 'Equipe 'Action Territoriale ", None, None]
        docs_stopwords_removed_custom_add = ["Chauffeur()  accompagnateur(trice) pers  mob - 5 ans  .", "Je maîtrise 12 langages informatiques   C & '  Permis B", "Coordinateur 'Equipe 'Action Territoriale ", None, None]
        docs_stopwords_none = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None]
        docs_stopwords_none_custom_add = ["Chauffeur(se)  accompagnateur(trice) pers à  - 5 ans de expérience.", "Je maîtrise 12  informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d' Territoriale ", None, None]
        docs_stopwords_removed_custom_remove = ["Chauffeur()  accompagnateur(trice) pers à mob - 5 ans  expérience.", "Je maîtrise 12 langages informatiques dont  C & '  Permis B", "Coordinateur 'Equipe 'Action Territoriale ", None, None]
        docs_stopwords_removed_custom_add_remove = ["Chauffeur(se)  accompagnateur(trice) pers à  - 5 ans de expérience.", "Je maîtrise 12  informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None]
        docs_unchanged = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs)).replace({np.nan:None})), docs_stopwords_removed)
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), opt='').replace({np.nan:None})), docs_stopwords_removed_default)
        # Ajout custom set
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), opt='', set_to_add=stopwords.STOPWORDS_OFFRES_1 + stopwords.STOPWORDS_OFFRES_2).replace({np.nan: None})), docs_stopwords_removed_custom_add)
        # Opt 'none'
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), opt='none').replace({np.nan: None})), docs_stopwords_none)
        # Test none et custom
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), opt='none', set_to_add=['mob', 'langages', 'Action', 'permis']).replace({np.nan: None})), docs_stopwords_none_custom_add)
        # Test remove custom
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), set_to_remove=['à', 'dont']).replace({np.nan: None})), docs_stopwords_removed_custom_remove)
        # Test add custom & remove custom
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), opt='none', set_to_add=['mob', 'langages', 'pers'], set_to_remove=['pers']).replace({np.nan: None})), docs_stopwords_removed_custom_add_remove)
        # Test remove all
        self.assertEqual(list(stopwords.remove_stopwords(pd.Series(docs), opt='none', set_to_add=['mob', 'langages', 'Action', 'permis'], set_to_remove=['mob', 'langages', 'Action', 'permis', 'à', 'dont']).replace({np.nan: None})), docs_unchanged)


    def test_stopwords_ascii(self):
        '''Test de la fonction stopwords.stopwords_ascii'''
        # Vérification du fonctionnement type
        self.assertEqual(type(stopwords.stopwords_ascii()), list)


    def test_stopwords_nltk(self):
        '''Test de la fonction stopwords.stopwords_nltk'''
        # Vérification du fonctionnement type
        self.assertEqual(type(stopwords.stopwords_nltk()), list)
        self.assertEqual(type(stopwords.stopwords_nltk(try_update=True)), list)

    def test_stopwords_nltk_ascii(self):
        '''Test de la fonction stopwords.stopwords_nltk_ascii'''
        # Vérification du fonctionnement type
        self.assertEqual(type(stopwords.stopwords_nltk_ascii()), list)



# Execution des tests
if __name__ == '__main__':
    unittest.main()
