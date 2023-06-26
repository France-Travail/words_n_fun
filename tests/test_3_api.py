#!/usr/bin/env python3
# coding=utf-8

## Tests - unit test of api functions
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
from unittest.mock import patch

# Utils libs
import os
import functools
import importlib
import pandas as pd
from words_n_fun import utils
from words_n_fun.preprocessing import api
from words_n_fun.preprocessing import basic

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class ApiTests(unittest.TestCase):
    '''Main class to test all functions in api.py.'''

    #Fix pour nose, la redéfinition du décorateur dans le test_basic, se propageait ici
    importlib.reload(utils)
    importlib.reload(basic)
    importlib.reload(api)


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_get_preprocessor(self):
        '''Testing function api.get_preprocessor'''
        # Vérification du fonctionnement type
        self.assertEqual(api.get_preprocessor().__class__, api.PreProcessor)
        self.assertEqual(api.get_preprocessor(pipeline=api.DEFAULT_PIPELINE, prefered_column='docs', modify_data=True, chunksize=0, first_row='header', columns=['docs', 'tags'], sep=',', nrows=0).__class__, api.PreProcessor)

        with self.assertRaises(ValueError):
            api.get_preprocessor(chunksize=-3)
        with self.assertRaises(ValueError):
            api.get_preprocessor(first_row='bad_value')
        with self.assertRaises(ValueError):
            api.get_preprocessor(nrows=-3)


    def test_preprocess_pipeline(self):
        '''Testing function api.preprocess_pipeline'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_def_pipeline = ['chauffeur accompagn per mob an experient', 'maitris langag informat c permisb', 'coordin equip action territorial', '', '']
        docs_series = pd.Series(["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None])
        docs_series_copy = pd.Series(["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None])
        docs_dataframe = pd.DataFrame([["test", "Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience."],
                                       ["test", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B"],
                                       ["test", "Coordinateur d'Equipe d'Action Territoriale "],
                                       ["test", 5],
                                       ["test", None]], columns=['test', 'docs'])
        docs_dataframe_copy = pd.DataFrame([["test", "Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience."],
                                            ["test", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B"],
                                            ["test", "Coordinateur d'Equipe d'Action Territoriale "],
                                            ["test", 5],
                                            ["test", None]], columns=['test', 'docs'])
        docs_test_2 = ['chauffeur(se)  (trice) pers à mob - 5 ans de expérience.', "je maîtrise 12 langages informatiques dont le c & j'ai le  b", "coordinateur d'equipe d' territoriale ", None, None]
        test_file = 'testing_file5.csv'
        result_file = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                    ['maitris langag informat c permisb', 'col 2 line 2'],
                                    ['coordin equip action territorial', 'col 2 line 3']
                                   ], columns=['col 1', 'col 2'])
        result_file_col_2 = pd.DataFrame([["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", 'col lin'],
                                          ["Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", 'col lin'],
                                          ["Coordinateur d'Equipe d'Action Territoriale ", 'col lin']
                                         ], columns=['col 1', 'col 2'])
        result_file_not_modified = pd.DataFrame([["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", 'col 2 line 1', 'chauffeur accompagn per mob an experient'],
                                                 ["Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", 'col 2 line 2', 'maitris langag informat c permisb'],
                                                 ["Coordinateur d'Equipe d'Action Territoriale ", 'col 2 line 3', 'coordin equip action territorial']
                                                ], columns=['col 1', 'col 2', 'col 1_processed'])
        result_file_data = pd.DataFrame([['col', 'col 2'],
                                         ['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                         ['maitris langag informat c permisb', 'col 2 line 2'],
                                         ['coordin equip action territorial', 'col 2 line 3']
                                        ], columns=['docs', 'tags'])
        result_file_skip = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                         ['maitris langag informat c permisb', 'col 2 line 2'],
                                         ['coordin equip action territorial', 'col 2 line 3']
                                        ], columns=['docs', 'tags'])
        result_file_skip_2 = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                           ['maitris langag informat c permisb', 'col 2 line 2'],
                                           ['coordin equip action territorial', 'col 2 line 3']
                                          ], columns=['test', 'test2'])
        result_file_sep = pd.DataFrame([['chauffeur accompagn per mob an experient col lin'],
                                        ['maitris langag informat c permisb col lin'],
                                        ['coordin equip action territorial col lin']
                                       ], columns=['col 1,col 2'])
        result_file_nrows = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                          ['maitris langag informat c permisb', 'col 2 line 2']
                                         ], columns=['col 1', 'col 2'])
        result_file_nrows_data = pd.DataFrame([['col', 'col 2'],
                                               ['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                              ], columns=['docs', 'tags'])
        result_file_nrows_skip = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                               ['maitris langag informat c permisb', 'col 2 line 2']
                                              ], columns=['docs', 'tags'])

        # Vérification du fonctionnement type
        self.assertEqual(api.preprocess_pipeline(docs), docs_def_pipeline)
        # Vérification non modification input
        _ = api.preprocess_pipeline(docs_series)
        pd.testing.assert_series_equal(docs_series, docs_series_copy)
        _ = api.preprocess_pipeline(docs_dataframe, prefered_column='docs')
        pd.testing.assert_frame_equal(docs_dataframe, docs_dataframe_copy)
        # Vérification envoie fonction dans pipeline
        self.assertEqual(api.preprocess_pipeline(docs, pipeline=['to_lower', functools.partial(basic.remove_words, words_to_remove=['accompagnateur', 'permis', 'action'])]), docs_test_2)
        # Verification fonctionnement fichier .csv
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file)), result_file)
        # Verification fonctionnement prefered_column
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, prefered_column='toto')), result_file)
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, prefered_column='col 1')), result_file)
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, prefered_column='col 2')), result_file_col_2)
        # Verification fonctionnement modify_data
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, modify_data=False)), result_file_not_modified)
        # Verification fonctionnement chunksize
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, chunksize=1)), result_file)
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, chunksize=100000)), result_file)
        # Verification fonctionnement first_row
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, first_row='data')), result_file_data)
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, first_row='skip')), result_file_skip)
        # Verification fonctionnement columns
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, first_row='skip', columns=['test', 'test2'])), result_file_skip_2)
        # Verification fonctionnement sep
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, sep=';'), sep=';'), result_file_sep)
        # Verification fonctionnement nrows
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, nrows=2)), result_file_nrows)
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, nrows=2, first_row='data')), result_file_nrows_data)
        pd.testing.assert_frame_equal(pd.read_csv(api.preprocess_pipeline(test_file, nrows=2, first_row='skip')), result_file_nrows_skip)

        with self.assertRaises(ValueError):
            api.preprocess_pipeline(docs, chunksize=-3)
        with self.assertRaises(ValueError):
            api.preprocess_pipeline(docs, first_row='bad_value')
        with self.assertRaises(ValueError):
            api.preprocess_pipeline(docs, nrows=-3)

        # Nettoyage fichiers
        dir = os.path.abspath(os.getcwd())
        list_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.startswith('testing_file5_')]
        for f in list_files:
            os.remove(f)


    def test_PreProcessor(self):
        '''Test de la classe api.PreProcessor'''
        docs = ["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", "Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", "Coordinateur d'Equipe d'Action Territoriale ", 5, None]
        docs_def_pipeline = ['chauffeur accompagn per mob an experient', 'maitris langag informat c permisb', 'coordin equip action territorial', '', '']
        docs_test_2 = ['chauffeur(se)  (trice) pers à mob - 5 ans de expérience.', "je maîtrise 12 langages informatiques dont le c & j'ai le  b", "coordinateur d'equipe d' territoriale ", None, None]
        test_file = 'testing_file5.csv'
        result_file = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                    ['maitris langag informat c permisb', 'col 2 line 2'],
                                    ['coordin equip action territorial', 'col 2 line 3']
                                   ], columns=['col 1', 'col 2'])
        result_file_col_2 = pd.DataFrame([["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.", 'col lin'],
                                          ["Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B", 'col lin'],
                                          ["Coordinateur d'Equipe d'Action Territoriale ", 'col lin']
                                         ], columns=['col 1', 'col 2'])
        result_file_not_modified = pd.DataFrame([["Chauffeur(se)  accompagnateur(trice) pers à mob - 5 ans de expérience.",'col 2 line 1', 'chauffeur accompagn per mob an experient'],
                                                 ["Je maîtrise 12 langages informatiques dont le C & j'ai le Permis B",'col 2 line 2', 'maitris langag informat c permisb'],
                                                 ["Coordinateur d'Equipe d'Action Territoriale ",'col 2 line 3', 'coordin equip action territorial']
                                                ], columns=['col 1', 'col 2', 'col 1_processed'])
        result_file_data = pd.DataFrame([['col', 'col 2'],
                                         ['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                         ['maitris langag informat c permisb', 'col 2 line 2'],
                                         ['coordin equip action territorial', 'col 2 line 3']
                                        ], columns=['docs', 'tags'])
        result_file_skip = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                         ['maitris langag informat c permisb', 'col 2 line 2'],
                                         ['coordin equip action territorial', 'col 2 line 3']
                                        ], columns=['docs', 'tags'])
        result_file_skip_2 = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                           ['maitris langag informat c permisb', 'col 2 line 2'],
                                           ['coordin equip action territorial', 'col 2 line 3']
                                          ], columns=['test', 'test2'])
        result_file_sep = pd.DataFrame([['chauffeur accompagn per mob an experient col lin'],
                                        ['maitris langag informat c permisb col lin'],
                                        ['coordin equip action territorial col lin']
                                       ], columns=['col 1,col 2'])
        result_file_nrows = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                          ['maitris langag informat c permisb', 'col 2 line 2']
                                         ], columns=['col 1', 'col 2'])
        result_file_nrows_data = pd.DataFrame([['col', 'col 2'],
                                               ['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                              ], columns=['docs', 'tags'])
        result_file_nrows_skip = pd.DataFrame([['chauffeur accompagn per mob an experient', 'col 2 line 1'],
                                               ['maitris langag informat c permisb', 'col 2 line 2']
                                              ], columns=['docs', 'tags'])

        # Vérification du fonctionnement type
        self.assertEqual(api.PreProcessor().transform(docs), docs_def_pipeline)
        # Vérification envoie fonction dans pipeline
        self.assertEqual(api.PreProcessor(pipeline=['to_lower', functools.partial(basic.remove_words, words_to_remove=['accompagnateur', 'permis', 'action'])]).transform(docs), docs_test_2)
        # Verification fonctionnement fichier .csv
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor().transform(test_file)), result_file)
        # Verification fonctionnement prefered_column
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(prefered_column='toto').transform(test_file)), result_file)
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(prefered_column='col 1').transform(test_file)), result_file)
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(prefered_column='col 2').transform(test_file)), result_file_col_2)
        # Verification fonctionnement modify_data
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(modify_data=False).transform(test_file)), result_file_not_modified)
        # Verification fonctionnement chunksize
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(chunksize=1).transform(test_file)), result_file)
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(chunksize=100000).transform(test_file)), result_file)
        # Verification fonctionnement first_row
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(first_row='data').transform(test_file)), result_file_data)
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(first_row='skip').transform(test_file)), result_file_skip)
        # Verification fonctionnement columns
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(first_row='skip', columns=['test', 'test2']).transform(test_file)), result_file_skip_2)
        # Verification fonctionnement sep
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(sep=';').transform(test_file), sep=';'), result_file_sep)
        # Verification fonctionnement nrows
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(nrows=2).transform(test_file)), result_file_nrows)
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(nrows=2, first_row='data').transform(test_file)), result_file_nrows_data)
        pd.testing.assert_frame_equal(pd.read_csv(api.PreProcessor(nrows=2, first_row='skip').transform(test_file)), result_file_nrows_skip)

        with self.assertRaises(ValueError):
            api.PreProcessor(chunksize=-3).transform(docs)
        with self.assertRaises(ValueError):
            api.PreProcessor(first_row='bad_value').transform(docs)
        with self.assertRaises(ValueError):
            api.PreProcessor(nrows=-3).transform(docs)

        # Nettoyage fichiers
        dir = os.path.abspath(os.getcwd())
        list_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.startswith('testing_file5_')]
        for f in list_files:
            os.remove(f)


    @patch('logging.Logger._log')
    def test_check_pipeline_order(self, PrintMockLog):
        '''Testing function api.check_pipeline_order'''
        def test(docs):
            return docs

        # On reenable le logger
        logging.disable(logging.NOTSET)
        # Assert pas de return
        self.assertEqual(None, api.check_pipeline_order(['test']))
        # Assertions sur print
        self.assertEqual(len(PrintMockLog.mock_calls), 1)  # warning sur fonction n'existe pas
        api.check_pipeline_order(['notnull', 'remove_non_string'])
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 1)  # warning ordre utilisation
        api.check_pipeline_order(['notnull', 'remove_non_string', 'notnull'])
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 1 + 2)  # warning ordre utilisation (x2)
        api.check_pipeline_order(['notnull', test])
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 1 + 2 + 1)  # Fonction custom
        api.check_pipeline_order(['notnull', test, 'toto'])
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 1 + 2 + 1 + 2)  # Fonction custom & fonction n'existe pas
        api.check_pipeline_order(['notnull', test, 'toto', test])
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 1 + 2 + 1 + 2 + 3)  # Fonction custom (x2) & fonction n'existe pas
        api.check_pipeline_order(['notnull', test, 'toto', 'toto'])
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 1 + 2 + 1 + 2 + 3 + 3)  # Fonction custom & fonction n'existe pas (x2)

        # RESET DEFAULT
        logging.disable(logging.CRITICAL)


    def test_listing_count_words(self):
        '''Testing function api.listing_count_words'''
        docs = ["ceci est un test de la fonction listing_count_words", "il s agit d une fonction qui compte les mots dans une serie pandas", "test compte ok test"]
        wanted_result = pd.DataFrame([['agit', 1], ['ceci', 1], ['compte', 2], ['d', 1], ['dans', 1], ['de', 1], ['est', 1],
                                      ['fonction', 2], ['il', 1], ['la', 1], ['les', 1], ['listing_count_words', 1], ['mots', 1],
                                      ['ok', 1], ['pandas', 1], ['qui', 1], ['s', 1], ['serie', 1], ['test', 3], ['un', 1],
                                      ['une', 2]], columns=['word', 'count'])

        # Vérification du fonctionnement type
        pd.testing.assert_frame_equal(api.listing_count_words(pd.Series(docs)), wanted_result)



    def test_list_one_appearance_word(self):
        '''Testing function api.list_one_appearance_word'''
        docs = ["ceci est un test de la fonction list_one_appearance_word", "il s agit d une fonction qui compte les mots dans une serie pandas", "test compte ok test"]
        wanted_result = pd.Series(['agit', 'ceci', 'd', 'dans', 'de', 'est', 'il', 'la', 'les', 'list_one_appearance_word', 'mots',
                                   'ok', 'pandas', 'qui', 's', 'serie', 'un'], name='word')

        # Vérification du fonctionnement type
        pd.testing.assert_series_equal(api.list_one_appearance_word(pd.Series(docs)), wanted_result)



# Execution des tests
if __name__ == '__main__':
    unittest.main()
