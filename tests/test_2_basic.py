#!/usr/bin/env python3

## Tests - unit test of basic functions
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
import sys
try:
    import spacy
except ModuleNotFoundError:
    pass
import importlib
import numpy as np
import pandas as pd
from words_n_fun import utils
from words_n_fun.preprocessing import basic
from words_n_fun.preprocessing import stopwords

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class BasicTests(unittest.TestCase):
    '''Main class to test all functions in basic.py.
    These functions are basics funtions to process string data.
    There are no 'complex' functions such as a lemmatizer.
    '''

    # Récupération du comportement nominal de basic.remove_accents (utile pour stopwords)
    nominal_remove_accents = basic.remove_accents

    # Mock du decorateur DataAgnostic (on le bypass pour les tests)
    default_decorator = lambda f: f
    utils.data_agnostic = default_decorator
    utils.data_agnostic_input = default_decorator

    # Reload de la librairie basic (pour application du decorateur par defaut)
    importlib.reload(basic)


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_notnull(self):
        '''Testing function basic.notnull'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_series = pd.Series(docs)
        docs_series_copy = pd.Series(docs)
        docs_processed = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, '']

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.notnull(pd.Series(docs_series))), docs_processed)
        # Vérification non modification input
        _ = basic.notnull(docs_series)
        pd.testing.assert_series_equal(docs_series, docs_series_copy)


    def test_remove_non_string(self):
        '''Testing function basic.remove_non_string'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_processed = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", '', '']

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_non_string(pd.Series(docs))), docs_processed)


    def test_get_true_spaces(self):
        '''Testing function basic.get_true_spaces'''
        docs = ["Ceci est un test", "Ceci est	un autre test 	", "Ceci est encore un	autre test.\n Avec un retour à la ligne :)", 5, None]
        docs_processed = ["Ceci est un test", "Ceci est un autre test  ", "Ceci est encore un autre test.  Avec un retour à la ligne :)", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.get_true_spaces(pd.Series(docs)).replace({np.nan: None})), docs_processed)



    def test_to_lower(self):
        '''Testing function basic.to_lower'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_lowered = ["chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "je maîtrise 12 langages informatiques dont le c & j'ai le permis b", "coordinateur d'equipe d'action territoriale ", None, None]
        docs_lowered_except_singleLetters = ["chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "je maîtrise 12 langages informatiques dont le C & j'ai le permis B", "coordinateur d'equipe d'action territoriale ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.to_lower(pd.Series(docs)).replace({np.nan: None})), docs_lowered)
        self.assertEqual(list(basic.to_lower(pd.Series(docs), threshold_nb_chars=2).replace({np.nan: None})), docs_lowered_except_singleLetters)



    def test_pe_matching(self):
        '''Testing function basic.pe_matching'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_series = pd.Series(docs)
        docs_series_copy = pd.Series(docs)
        docs_matched = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le PermisB", "Coordinateur d'Equipe d'Action Territoriale ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.pe_matching(pd.Series(docs)).replace({np.nan: None})), docs_matched)
        # Vérification non modification input
        _ = basic.notnull(docs_series)
        pd.testing.assert_series_equal(docs_series, docs_series_copy)



    def test_remove_punct(self):
        '''Testing function basic.remove_punct'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d_Equipe d'Action Territoriale ", 5, None]
        docs_punct_replaced = ["Chauffeur se   accompagnateur trice  pers à mob   5 ans de expérience ", "Je maîtrise 12 langages informatiques dont le C   j ai le Permis B", "Coordinateur d Equipe d Action Territoriale ", None, None]
        docs_punct_replaced_except_parentesis = ["Chauffeur(se)  accompagnateur(trice) pers à mob   5 ans de expérience ", "Je maîtrise 12 langages informatiques dont le C   j ai le Permis B", "Coordinateur d Equipe d Action Territoriale ", None, None]
        docs_punct_removed = ["Chauffeurse  accompagnateurtrice pers à mob  5 ans de expérience", "Je maîtrise 12 langages informatiques dont le C  jai le Permis B", "Coordinateur dEquipe dAction Territoriale ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_punct(pd.Series(docs)).replace({np.nan: None})), docs_punct_replaced)
        self.assertEqual(list(basic.remove_punct(pd.Series(docs), del_parenthesis=False).replace({np.nan: None})), docs_punct_replaced_except_parentesis)
        self.assertEqual(list(basic.remove_punct(pd.Series(docs), replacement_char='').replace({np.nan: None})), docs_punct_removed)



    def test_trim_string(self):
        '''Testing function basic.trim_string'''
        docs = ["    Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques \n\n dont le C & j'ai le Permis B\n", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_processed = ["Chauffeur(se) accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques \n\n dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.trim_string(pd.Series(docs)).replace({np.nan: None})), docs_processed)



    def test_remove_leading_and_ending_spaces(self):
        '''Testing function basic.remove_leading_and_ending_spaces'''
        docs = [" )test test toto(", "             \t  test \t", 'test avec nouvelle ligne\n', 5, None]
        docs_processed = [")test test toto(", "test", 'test avec nouvelle ligne', None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_leading_and_ending_spaces(pd.Series(docs)).replace({np.nan: None})), docs_processed)



    def test_remove_numeric(self):
        '''Testing function basic.remove_numeric'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_numeric_replaced = ["Chauffeur(se)  accompagnateur(trice) pers à mob -   ans de expérience.", "Je maîtrise   langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None]
        docs_numeric_removed = ["Chauffeur(se)  accompagnateur(trice) pers à mob -  ans de expérience.", "Je maîtrise  langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_numeric(pd.Series(docs)).replace({np.nan: None})), docs_numeric_replaced)
        self.assertEqual(list(basic.remove_numeric(pd.Series(docs), replacement_char='').replace({np.nan: None})), docs_numeric_removed)



    def test_remove_accents(self):
        '''Testing function basic.remove_accents'''
        docs = ["tést têst tèst", 5, None]
        docs_accents_removed = ["test test test", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_accents(pd.Series(docs)).replace({np.nan: None})), docs_accents_removed)
        self.assertEqual(list(basic.remove_accents(pd.Series(docs), use_tqdm=False).replace({np.nan: None})), docs_accents_removed)


    @patch('words_n_fun.preprocessing.basic.remove_accents', side_effect=nominal_remove_accents)
    def test_remove_stopwords(self, RemoveAccentMock):
        '''Testing function basic.remove_stopwords'''
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
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs)).replace({np.nan: None})), docs_stopwords_removed)
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), opt='').replace({np.nan: None})), docs_stopwords_removed_default)
        # Ajout custom set
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), opt='', set_to_add=stopwords.STOPWORDS_OFFRES_1 + stopwords.STOPWORDS_OFFRES_2).replace({np.nan: None})), docs_stopwords_removed_custom_add)
        # Opt 'none'
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), opt='none').replace({np.nan: None})), docs_stopwords_none)
        # Test none et custom
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), opt='none', set_to_add=['mob', 'langages', 'Action', 'permis']).replace({np.nan: None})), docs_stopwords_none_custom_add)
        # Test remove custom
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), set_to_remove=['à', 'dont']).replace({np.nan: None})), docs_stopwords_removed_custom_remove)
        # Test add custom & remove custom
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), opt='none', set_to_add=['mob', 'langages', 'pers'], set_to_remove=['pers']).replace({np.nan: None})), docs_stopwords_removed_custom_add_remove)
        # Test remove all
        self.assertEqual(list(basic.remove_stopwords(pd.Series(docs), opt='none', set_to_add=['mob', 'langages', 'Action', 'permis'], set_to_remove=['mob', 'langages', 'Action', 'permis', 'à', 'dont']).replace({np.nan: None})), docs_unchanged)



    def test_remove_gender_synonyms(self):
        '''Testing function basic.remove_gender_synonyms'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None, "serveur/serveur(se), agriculteur (trice) blabla ouvrier/ ouvrière blabla aide apprenti boucher /aide apprentie bouchere"]
        docs_gender_syn_removed = ['Chauffeur   accompagnateur  pers à mob - 5 ans de expérience.', "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", None, None, 'serveur , agriculteur  blabla ouvrier blabla aide apprenti boucher']

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_gender_synonyms(pd.Series(docs)).replace({np.nan: None})), docs_gender_syn_removed)


    @unittest.skipIf('spacy' not in sys.modules, "Skipping all lemmatizer tests as spacy can't be imported.")
    def test_lemmatize(self):
        '''Testing function basic.lemmatize'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "^Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_lemmatized = ['chauffeur se accompagnateur trice pers à mob 5 an de expérience', 'je maîtrise 12 langage informatique dont le c j avoir le permettre b', 'chauffeur se accompagnateur trice pers à mob 5 an de expérience', 'coordinateur d equipe d action territorial', None, None]

        # On check seulement si model lemmatizer installé
        if spacy.util.is_package("fr_core_news_sm"):
            # Vérification du fonctionnement type
            self.assertEqual(list(basic.lemmatizer.lemmatize(pd.Series(docs)).replace({np.nan:None})), docs_lemmatized)
            self.assertEqual(list(basic.lemmatizer.lemmatize(pd.Series(docs*100)).replace({np.nan:None})), docs_lemmatized*100)


    def test_stemmatize(self):
        '''Testing function basic.stemmatize'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_stemmatized = ['chauffeur(se)  accompagnateur(trice) per à mob - 5 an de expérience.', "je maîtris 12 langag informat dont le c & j'ai le perm b", "coordin d'equip d'action territorial ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.stemmatize(pd.Series(docs)).replace({np.nan: None})), docs_stemmatized)



    def test_add_point(self):
        '''Testing function basic.add_point'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_processed = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B.", "Coordinateur d'Equipe d'Action Territoriale .", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.add_point(pd.Series(docs)).replace({np.nan: None})), docs_processed)



    def test_deal_with_specific_characters(self):
        '''Testing function basic.deal_with_specific_characters'''
        docs = [".Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", " .Je maîtrise,12 langages, informatiques , dont le C & j'ai le Permis B. ", " . Coordinateur d'Equipe d'Action Territoriale . ", 5, None]
        docs_processed = [" . Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience . ", " . Je maîtrise , 12 langages , informatiques , dont le C & j ' ai le Permis B . ", " . Coordinateur d ' Equipe d ' Action Territoriale . ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.deal_with_specific_characters(pd.Series(docs)).replace({np.nan: None})), docs_processed)



    def test_replace_urls(self):
        '''Testing function basic.replace_urls'''
        docs = ["ceci est un text avec une URL:http://www.test.com/zaeaze?pd==sqd", "ahttp://ftp.test.com", "HTTP://toto.fr/dfsdsfo?pfp=zaeazpo",
                "www.test", "www.test.com/toto", "toto.titi@gmail.com", 5, None]
        docs_url_replaced = ["ceci est un text avec une URL: ", "ahttp://ftp.test.com", " ",
                             "www.test", " ", "toto.titi@gmail.com", None, None]
        docs_url_replaced_2 = ["ceci est un text avec une URL:URL", "ahttp://ftp.test.com", "URL",
                               "www.test", "URL", "toto.titi@gmail.com", None, None]
        docs_url_replaced_3 = ["ceci est un text avec une URL: test.com ", "ahttp://ftp.test.com", " toto.fr ",
                               "www.test", " test.com ", "toto.titi@gmail.com", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.replace_urls(pd.Series(docs)).replace({np.nan: None})), docs_url_replaced)
        self.assertEqual(list(basic.replace_urls(pd.Series(docs), replacement_char='URL').replace({np.nan: None})), docs_url_replaced_2)
        self.assertEqual(list(basic.replace_urls(pd.Series(docs), replacement_char="azsedazzad", replace_with_domain=True).replace({np.nan: None})), docs_url_replaced_3)



    def test_remove_words(self):
        '''Testing function basic.remove_words'''
        docs = ["ceci est un test de la fonction remove_words OK", "il s agit d une fonction qui compte les mots dans une serie pandas", "test compte j'ok test", 5, None]
        words_to_remove = ['test', 'toto', 'fonction', 'ok']
        docs_processed = ["ceci est un  de la  remove_words OK", "il s agit d une  qui compte les mots dans une serie pandas", " compte j' ", None, None]
        docs_processed_case = ["ceci est un  de la  remove_words ", "il s agit d une  qui compte les mots dans une serie pandas", " compte j' ", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.remove_words(pd.Series(docs), words_to_remove).replace({np.nan: None})), docs_processed)
        self.assertEqual(list(basic.remove_words(pd.Series(docs), words_to_remove, case_insensitive=True).replace({np.nan: None})), docs_processed_case)



    def test_fix_text(self):
        '''Testing function basic.fix_text'''
        docs = ["Ãºnico", "là entités HTML &lt;3", "ＬＯＵＤ　ＮＯＩＳＥＳ", 5, None]
        results = ["único", "là entités HTML <3", "LOUD NOISES", None, None]
        results_2 = ["Ãºnico", "là entités HTML <3", "LOUD NOISES", None, None]

        # Vérification du fonctionnement type
        self.assertEqual(list(basic.fix_text(pd.Series(docs)).replace({np.nan: None})), results)
        self.assertEqual(list(basic.fix_text(pd.Series(docs), fix_encoding=False).replace({np.nan: None})), results_2)



# Execution des tests
if __name__ == '__main__':
    unittest.main()
