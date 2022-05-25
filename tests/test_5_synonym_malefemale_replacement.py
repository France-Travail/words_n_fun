#!/usr/bin/env python3

## Test - unit test of synonym_malefemale_replacement functions
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
from pe_semantic.preprocessing import synonym_malefemale_replacement

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class SynonymTests(unittest.TestCase):
    '''Main class to test all functions in synonym_malefemale_replacement.py'''

    # Mock du decorateur DataAgnostic (on le bypass pour les tests)
    default_decorator = lambda f: f
    utils.data_agnostic = default_decorator
    utils.data_agnostic_input = default_decorator
    # Reload de la librairie basic (pour application du decorateur par defaut)
    importlib.reload(synonym_malefemale_replacement)


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_remove_gender_synonyms(self):
        '''Test de la fonction synonym_malefemale_replacement.remove_gender_synonyms'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None, "serveur/serveur(se), agriculteur (trice) blabla ouvrier/ ouvrière blabla aide apprenti boucher /aide apprentie bouchere"]
        docs_gender_syn_removed = ['Chauffeur   accompagnateur  pers à mob - 5 ans de expérience.', "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None, 'serveur , agriculteur  blabla ouvrier blabla aide apprenti boucher']

        # Vérification du fonctionnement type
        self.assertEqual(list(synonym_malefemale_replacement.remove_gender_synonyms(pd.Series(docs)).replace({np.nan:None})), docs_gender_syn_removed)


    def test_matching_words(self):
        '''Test de la fonction synonym_malefemale_replacement.matching_words'''
        word1 = 'serveur'
        word2 = 'serveuse'
        result =  ('serveur', 'serveuse', 'serveur')

        # Vérification du fonctionnement type
        self.assertEqual(synonym_malefemale_replacement.matching_words(word1,word2), result)


    def test_update_synonyms_set(self):
        '''Test de la fonction synonym_malefemale_replacement.update_synonyms_set'''
        synonyms_set = {}
        match = [('serveur', 'serveuse', ''),('boucher','ere',''),('boucher','bouchere','')]
        numligne = 1
        result =  {('serveur', 'serveuse', 'serveur'): 1, ('boucher', 'ere', 'boucher'): 1,('boucher', 'bouchere', 'boucher'): 1}

        # Vérification du fonctionnement type
        self.assertEqual(synonym_malefemale_replacement.update_synonyms_set(synonyms_set, match, numligne), result)




# Execution des tests
if __name__ == '__main__':
    unittest.main()
