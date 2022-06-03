#!/usr/bin/env python3

## Tests - unit test of utils functions
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
import re
import ntpath
import numpy as np
import pandas as pd

import words_n_fun as wnf
from words_n_fun import utils

# Pour ces tests, on garde le logger pour certains tests
# Du coup, default to critical
import logging
logging.disable(logging.CRITICAL)
logger = wnf.logger


class UtilsTests(unittest.TestCase):
    '''Main class to test all functions in utils.py'''

    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    # On test le logger en même temps que la fonction timer()
    # TODO : dissocier les deux ?
    @patch('logging.Logger._log')
    def test_timer(self, PrintMockLog):
        '''Testing function utils.timer'''
        # Definition d'une fonction à tester
        def test(x):
            return x**2

        # On reenable le logger
        logging.disable(logging.NOTSET)
        # Assert bon return
        self.assertEqual(utils.timer(test)(5), 25)
        # Assert print called once
        self.assertEqual(len(PrintMockLog.mock_calls), 1)

        # LOG level 'DEBUG'
        logger.setLevel(logging.DEBUG)
        # Assert print debug & info both called once
        utils.timer(test)(5)
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 2)

        # LOG level 'INFO'
        logger.setLevel(logging.INFO)
        # Assert print called one more time for INFO
        utils.timer(test)(5)
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 2 + 1)

        # LOG level 'ERROR'
        logger.setLevel(logging.ERROR)
        # Assert print not called
        utils.timer(test)(5)
        self.assertEqual(len(PrintMockLog.mock_calls), 1 + 2 + 1 + 0)

        # RESET DEFAULT
        logger.setLevel(logging.INFO)
        logging.disable(logging.CRITICAL)


    def test_data_agnostic(self):
        '''Testing function utils.data_agnostic'''
        # Definition d'une fonction à décorer
        def test_function(docs):
            if type(docs) != pd.Series: raise TypeError('')
            return docs.apply(lambda x: 'test')
        # Vals à tester
        test_str = "ceci est un test"
        result_str = "test"
        test_list = ["ceci est un test", 5, None]
        result_list = ["test", "test", "test"]
        test_np_array = np.array(["ceci est un test", 5, None])
        result_np_array = np.array(["test", "test", "test"])
        test_series = pd.Series(["ceci est un test", 5, None])
        test_series_copy = pd.Series(["ceci est un test", 5, None])
        result_series = pd.Series(["test", "test", "test"])
        test_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'],
                                       ['a', 'b'], ['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        test_dataframe_copy = pd.DataFrame([['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'],
                                            ['a', 'b'], ['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        result_dataframe_col_1 = pd.DataFrame([['test', 'b'], ['test', 'b'], ['test', 'b'], ['test', 'b'], ['test', 'b'], ['test', 'b'],
                                               ['test', 'b'], ['test', 'b'], ['test', 'b']], columns=['col 1', 'col 2'])
        result_dataframe_col_2 = pd.DataFrame([['a', 'test'], ['a', 'test'], ['a', 'test'], ['a', 'test'], ['a', 'test'], ['a', 'test'],
                                               ['a', 'test'], ['a', 'test'], ['a', 'test']], columns=['col 1', 'col 2'])
        test_file = "./testing_file.csv"
        result_file_col_1 = pd.DataFrame([['test', 'col 2 line 1'], ['test', 'col 2 line 2'], ['test', 'col 2 line 3'],
                                          ['test', 'col 2 line 4'], ['test', 'col 2 line 5'], ['test', 'col 2 line 6'],
                                          ['test', 'col 2 line 7'], ['test', 'col 2 line 8'], ['test', 'col 2 line 9']], columns=['col 1', 'col 2'])
        result_file_col_2 = pd.DataFrame([['col 1 line 1', 'test'], ['col 1 line 2', 'test'], ['col 1 line 3', 'test'],
                                          ['col 1 line 4', 'test'], ['col 1 line 5', 'test'], ['col 1 line 6', 'test'],
                                          ['col 1 line 7', 'test'], ['col 1 line 8', 'test'], ['col 1 line 9', 'test']], columns=['col 1', 'col 2'])
        test_file_2 = "./testing_file4.csv"
        result_file_def_sep = pd.DataFrame([['test'], ['test'], ['test'],
                                            ['test'], ['test'], ['test'],
                                            ['test'], ['test'], ['test']], columns=['col 1;col 2'])
        result_file_semi_col = pd.DataFrame([['test', 'col 2 line 1'], ['test', 'col 2 line 2'], ['test', 'col 2 line 3'],
                                             ['test', 'col 2 line 4'], ['test', 'col 2 line 5'], ['test', 'col 2 line 6'],
                                             ['test', 'col 2 line 7'], ['test', 'col 2 line 8'], ['test', 'col 2 line 9']], columns=['col 1', 'col 2'])
        empty_file = "./testing_file3.csv"

        # Vérification du fonctionnement type
        self.assertEqual(utils.data_agnostic(test_function)(test_str), result_str)
        self.assertEqual(utils.data_agnostic(test_function)(test_list), result_list)
        np.testing.assert_array_equal(utils.data_agnostic(test_function)(test_np_array), result_np_array)
        pd.testing.assert_series_equal(utils.data_agnostic(test_function)(test_series), result_series)
        pd.testing.assert_frame_equal(utils.data_agnostic(test_function)(test_dataframe), result_dataframe_col_1)
        pd.testing.assert_frame_equal(utils.data_agnostic(test_function, prefered_column="col 2")(test_dataframe), result_dataframe_col_2)
        pd.testing.assert_frame_equal(pd.read_csv(utils.data_agnostic(test_function)(test_file)), result_file_col_1)
        pd.testing.assert_frame_equal(pd.read_csv(utils.data_agnostic(test_function, prefered_column="col 2")(test_file)), result_file_col_2)
        pd.testing.assert_frame_equal(pd.read_csv(utils.data_agnostic(test_function)(test_file_2)), result_file_def_sep)
        pd.testing.assert_frame_equal(pd.read_csv(utils.data_agnostic(test_function, sep=';')(test_file_2), sep=';'), result_file_semi_col)
        # Vérification non modification input
        _ = utils.data_agnostic(test_function)(test_series)
        pd.testing.assert_series_equal(test_series, test_series_copy)
        _ = utils.data_agnostic(test_function, prefered_column="col 1")(test_dataframe)
        pd.testing.assert_frame_equal(test_dataframe, test_dataframe_copy)

        # Nettoyage fichiers
        dir = os.path.abspath(os.getcwd())
        list_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.startswith('testing_file_')] \
                     + [os.path.join(dir, f) for f in os.listdir(dir) if f.startswith('testing_file4_')]
        for f in list_files:
            os.remove(f)


    def test_data_agnostic_input(self):
        '''Testing function utils.data_agnostic_input'''
        # Definition d'une fonction à décorer
        def test_function(docs):
            if type(docs) != pd.Series: raise TypeError('')
            return docs.apply(lambda x: 'test')
        # Vals à tester
        test_str = "ceci est un test"
        result_str = pd.Series("test")
        test_list = ["ceci est un test", 5, None]
        result_list = pd.Series(["test", "test", "test"])
        test_np_array = np.array(["ceci est un test", 5, None])
        result_np_array = pd.Series(["test", "test", "test"])
        test_series = pd.Series(["ceci est un test", 5, None])
        result_series = pd.Series(["test", "test", "test"])
        test_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'],
                                       ['a', 'b'], ['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        result_dataframe_col_1 = pd.Series(['test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test'], name="col 1")
        result_dataframe_col_2 = pd.Series(['test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test'], name="col 2")
        test_file =  "./testing_file.csv"
        result_file_col_1 = pd.Series(['test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test'], name="col 1")
        result_file_col_2 = pd.Series(['test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test'], name="col 2")
        test_file_2 = "./testing_file4.csv"
        result_file_def_sep = pd.Series(['test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test'], name='col 1;col 2')
        result_file_semi_col = pd.Series(['test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test'], name="col 1")
        empty_file = "./testing_file3.csv"

        # Vérification du fonctionnement type
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_str), result_str)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_list), result_list)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_np_array), result_np_array)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_series), result_series)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_dataframe), result_dataframe_col_1)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function, prefered_column="col 2")(test_dataframe), result_dataframe_col_2)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_file), result_file_col_1)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function, prefered_column="col 2")(test_file), result_file_col_2)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function)(test_file_2), result_file_def_sep)
        pd.testing.assert_series_equal(utils.data_agnostic_input(test_function, sep=";")(test_file_2), result_file_semi_col)


    def test_get_docs_type(self):
        '''Testing function utils.get_docs_type'''
        test_str = "ceci est un test"
        test_list = ["ceci est un test", 5, None]
        test_np_array = np.array(["ceci est un test", 5, None])
        test_series = pd.Series(["ceci est un test", 5, None])
        test_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'],
                                       ['a', 'b'], ['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        test_file =  "./testing_file.csv"

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_docs_type(test_str), 'str')
        self.assertEqual(utils.get_docs_type(test_list), 'list')
        self.assertEqual(utils.get_docs_type(test_np_array), 'np.ndarray')
        self.assertEqual(utils.get_docs_type(test_series), 'pd.Series')
        self.assertEqual(utils.get_docs_type(test_dataframe), 'pd.DataFrame')
        self.assertEqual(utils.get_docs_type(test_file), 'file_path')

        #Vérification du type du/des input(s)
        with self.assertRaises(TypeError):
            utils.get_docs_type(5)


    def test_get_docs_length(self):
        '''Testing function utils.get_docs_length'''
        test_str = "ceci est un test"
        test_list = ["ceci est un test", 5, None]
        test_np_array = np.array(["ceci est un test", 5, None])
        test_series = pd.Series(["ceci est un test", 5, None])
        test_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'], ['a', 'b'],
                                       ['a', 'b'], ['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        test_file_comma =  "./testing_file.csv"
        test_file_colon =  "./testing_file4.csv"

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_docs_length(test_str), 1)
        self.assertEqual(utils.get_docs_length(test_list), 3)
        self.assertEqual(utils.get_docs_length(test_np_array), 3)
        self.assertEqual(utils.get_docs_length(test_series), 3)
        self.assertEqual(utils.get_docs_length(test_dataframe), 9)
        self.assertEqual(utils.get_docs_length(test_file_comma, sep=','), 9)
        self.assertEqual(utils.get_docs_length(test_file_colon, sep=';'), 9)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='data', sep=','), 10)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='skip', sep=','), 9)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='header', sep=',', nrows=5), 5)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='data', sep=',', nrows=5), 5)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='skip', sep=',', nrows=5), 5)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='header', sep=',', nrows=30), 9)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='data', sep=',', nrows=30), 10)
        self.assertEqual(utils.get_docs_length(test_file_comma, first_row='skip', sep=',', nrows=30), 9)

    def test_get_file_length(self):
        '''Testing function utils.file_length'''
        input_file_1 = "./testing_file.csv"
        expected_result_1 = 10
        input_file_2 = "./testing_file6.csv"
        expected_result_2 = 200000
        expected_result_3 = 200000 * 2
        # Vérification du fonctionnement type
        self.assertEqual(utils.get_file_length(input_file_1), expected_result_1)
        self.assertEqual(utils.get_file_length(input_file_2), expected_result_2)
        self.assertEqual(utils.get_file_length(input_file_2, sep=';'), expected_result_3)

        with self.assertRaises(FileNotFoundError):
            utils.get_file_length("imaginary_file.txt")


    def test_get_new_csv_name(self):
        '''Testing function utils.get_new_csv_name'''
        input_file = "./testing_file.csv"

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_new_csv_name(input_file).endswith('.csv'), True)
        self.assertEqual(ntpath.basename(utils.get_new_csv_name(input_file)).startswith('testing_file_'), True)
        self.assertEqual(os.path.isfile(utils.get_new_csv_name(input_file)), False)


    def test_get_generator(self):
        '''Testing function utils.get_generator'''
        test_str = 'test'
        result_str = 'test'
        test_list = ['test', 'test 2']
        result_list = ['test', 'test 2']
        test_np_array = np.array(["ceci est un test", 5, None])
        result_np_array = np.array(["ceci est un test", 5, None])
        test_series = pd.Series(["ceci est un test", 5, None])
        result_series = pd.Series(["ceci est un test", 5, None])
        test_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        result_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        test_file = "./testing_file.csv"
        result_file = pd.DataFrame([['col 1 line 1', 'col 2 line 1'], ['col 1 line 2', 'col 2 line 2'], ['col 1 line 3', 'col 2 line 3'],
                                    ['col 1 line 4', 'col 2 line 4'], ['col 1 line 5', 'col 2 line 5'], ['col 1 line 6', 'col 2 line 6'],
                                    ['col 1 line 7', 'col 2 line 7'], ['col 1 line 8', 'col 2 line 8'], ['col 1 line 9', 'col 2 line 9']], columns=['col 1', 'col 2'])
        result_file_chunk_1 = pd.DataFrame([['col 1 line 1', 'col 2 line 1']], columns=['col 1', 'col 2'])
        result_file_n_rows_5 = pd.DataFrame([['col 1 line 1', 'col 2 line 1'], ['col 1 line 2', 'col 2 line 2'], ['col 1 line 3', 'col 2 line 3'],
                                             ['col 1 line 4', 'col 2 line 4'], ['col 1 line 5', 'col 2 line 5']], columns=['col 1', 'col 2'])
        result_file_cols = pd.DataFrame([['col 1', 'col 2'], ['col 1 line 1', 'col 2 line 1'], ['col 1 line 2', 'col 2 line 2'], ['col 1 line 3', 'col 2 line 3'],
                                         ['col 1 line 4', 'col 2 line 4'], ['col 1 line 5', 'col 2 line 5'], ['col 1 line 6', 'col 2 line 6'],
                                         ['col 1 line 7', 'col 2 line 7'], ['col 1 line 8', 'col 2 line 8'], ['col 1 line 9', 'col 2 line 9']], columns=['test', 'test 2'])

        # Vérification du fonctionnement type
        self.assertEqual(next(utils.get_generator(test_str)), result_str)
        self.assertEqual(next(utils.get_generator(test_list)), result_list)
        np.testing.assert_array_equal(next(utils.get_generator(test_np_array)), result_np_array)
        pd.testing.assert_series_equal(next(utils.get_generator(test_series)), result_series)
        pd.testing.assert_frame_equal(next(utils.get_generator(test_dataframe)), result_dataframe)
        pd.testing.assert_frame_equal(next(utils.get_generator(test_file)), result_file)
        pd.testing.assert_frame_equal(next(utils.get_generator(test_file, chunksize=1)), result_file_chunk_1)
        pd.testing.assert_frame_equal(next(utils.get_generator(test_file, chunksize=5)), result_file_n_rows_5)
        pd.testing.assert_frame_equal(next(utils.get_generator(test_file, first_row='data', columns=['test', 'test 2'])), result_file_cols)

        with self.assertRaises(ValueError):
            next(utils.get_generator(test_file, chunksize=-3))
        with self.assertRaises(ValueError):
            next(utils.get_generator(test_file, first_row='bad_value'))
        with self.assertRaises(ValueError):
            next(utils.get_generator(test_file, nrows=-3))
        with self.assertRaises(FileNotFoundError):
            next(utils.get_generator("imaginary_file.csv"))


    def test_get_df_generator_from_csv(self):
        '''Testing function utils.get_df_generator_from_csv'''
        input_file =  "./testing_file.csv"
        expected_result = [pd.DataFrame( [["col 1 line 1", "col 2 line 1"]], columns=['col 1', 'col 2'], index=[0]),
                           pd.DataFrame( [["col 1 line 2", "col 2 line 2"]], columns=['col 1', 'col 2'], index=[1]),
                           pd.DataFrame( [["col 1 line 3", "col 2 line 3"]], columns=['col 1', 'col 2'], index=[2]),
                           pd.DataFrame( [["col 1 line 4", "col 2 line 4"]], columns=['col 1', 'col 2'], index=[3]),
                           pd.DataFrame( [["col 1 line 5", "col 2 line 5"]], columns=['col 1', 'col 2'], index=[4]),
                           pd.DataFrame( [["col 1 line 6", "col 2 line 6"]], columns=['col 1', 'col 2'], index=[5]),
                           pd.DataFrame( [["col 1 line 7", "col 2 line 7"]], columns=['col 1', 'col 2'], index=[6]),
                           pd.DataFrame( [["col 1 line 8", "col 2 line 8"]], columns=['col 1', 'col 2'], index=[7]),
                           pd.DataFrame( [["col 1 line 9", "col 2 line 9"]], columns=['col 1', 'col 2'], index=[8])
                           ]

        # Vérification du fonctionnement type
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='header', sep=',', chunksize=1)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))

        # Chargement en 1 passe
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='header', sep=',', chunksize=0)
        result = next(generator_to_test)
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))

        # Test skip header & columns
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', sep=',', columns=['col 1', 'col 2'], chunksize=1)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))
        # 1 colonne en trop
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', sep=',', columns=['col 1', 'col 2', 'one_useless_column_name'], chunksize=1)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))
        # 1 colonne en moins
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', sep=',', columns=['col 1'], chunksize=1)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result).rename(columns={'col 2': '0'}))

        # Test avec chunksize  = 2
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='header', sep=',', chunksize=2)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))

        # Test avec chunksize > taille fichier
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='header', sep=',', chunksize=1000)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))

        # Test first_row = data
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='data', columns=['col 1', 'col 2'], sep=',')
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat([pd.DataFrame([['col 1', 'col 2']], columns=['col 1', 'col 2'])] + expected_result).reset_index(drop=True))

        # Test first_row = skip
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', columns=['col 1 test', 'col 2'], sep=',')
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result).rename(columns={'col 1': 'col 1 test'}))

        # Test nrows != 0
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='header', sep=',', nrows=5)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result[:5]))
        #
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', columns=['col 1', 'col 2'], sep=',', nrows=5)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result[:5]).reset_index(drop=True))
        # nrows > file_length
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='header', sep=',', nrows=500)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))

        # Load empty df
        generator_to_test = utils.get_df_generator_from_csv("./testing_file2.csv", first_row='header', sep=',', nrows=5)
        result = pd.concat([_ for _ in generator_to_test])
        pd.testing.assert_frame_equal(result, pd.DataFrame([], columns=['col 1', 'col 2']))

        # Vérification premeir nom de col '0' (utile pour la fonction get_column_to_be_processed)
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', columns=[], sep=',', chunksize=1)
        result = pd.concat([_ for _ in generator_to_test])
        self.assertEqual(result.columns[0], '0')


        # Autres tests (mélange certains arguments)
        #
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', sep=',', columns=['col 1', 'col 2'], chunksize=0)
        result = next(generator_to_test)
        pd.testing.assert_frame_equal(result, pd.concat(expected_result))
        #
        generator_to_test = utils.get_df_generator_from_csv(input_file, first_row='skip', sep=',', columns=['col 1', 'col 2'], chunksize=0, nrows=5)
        result = next(generator_to_test)
        pd.testing.assert_frame_equal(result, pd.concat(expected_result[:5]))

        with self.assertRaises(ValueError):
            gen = utils.get_df_generator_from_csv("./testing_file3.csv")
            next(gen)
        with self.assertRaises(ValueError):
            gen = utils.get_df_generator_from_csv(input_file, chunksize=-3)
            next(gen)
        with self.assertRaises(ValueError):
            gen = utils.get_df_generator_from_csv(input_file, nrows=-3)
            next(gen)
        # Vérification de l'existence du fichier :
        with self.assertRaises(FileNotFoundError):
            gen = utils.get_df_generator_from_csv("imaginary_file.txt")
            next(gen)


    def test_get_columns_to_use(self):
        '''Testing function utils.get_columns_to_use'''
        test_file =  "./testing_file.csv"
        test_file_one_row =  "./testing_file2.csv"
        empty_file = "./testing_file3.csv"
        test_file_semi_col =  "./testing_file4.csv"

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_columns_to_use(test_file), ['col 1', 'col 2'])
        self.assertEqual(utils.get_columns_to_use(test_file, file_length=10), ['col 1', 'col 2'])
        self.assertEqual(utils.get_columns_to_use(test_file, first_row='skip'), ['docs', 'tags'])
        self.assertEqual(utils.get_columns_to_use(test_file, first_row='skip', columns=['test', 'test2']), ['test', 'test2'])
        self.assertEqual(utils.get_columns_to_use(test_file, first_row='data'), ['docs', 'tags'])
        self.assertEqual(utils.get_columns_to_use(test_file_one_row), ['col 1', 'col 2'])
        self.assertEqual(utils.get_columns_to_use(test_file_one_row, first_row='data', columns=['test', 'test2']), ['test', 'test2'])
        self.assertEqual(utils.get_columns_to_use(test_file_semi_col), ['col 1;col 2'])
        self.assertEqual(utils.get_columns_to_use(test_file_semi_col, sep=';'), ['col 1', 'col 2'])

        with self.assertRaises(ValueError):
            utils.get_columns_to_use(test_file, first_row='bad_value')
        with self.assertRaises(ValueError):
            utils.get_columns_to_use(test_file, file_length=-1)
        with self.assertRaises(ValueError):
            utils.get_columns_to_use(empty_file)
        with self.assertRaises(FileNotFoundError):
            utils.get_columns_to_use("imaginary_file.txt")


    def test_get_new_column_name(self):
        '''Testing function utils.get_new_column_name'''
        docs_column = ['test', 'test2']
        processed_column = 'test2'
        new_column = 'test2_processed'
        docs_column_2 = [0, 1, '1_processed']
        processed_column_2 = 1
        new_column_2 = '1_processed_2'

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_new_column_name(docs_column, processed_column), new_column)
        self.assertEqual(utils.get_new_column_name(docs_column_2, processed_column_2), new_column_2)


    def test_get_column_to_be_processed(self):
        '''Testing function utils.get_column_to_be_processed'''
        test_str = 'test'
        result_str = 'docs'
        test_list = ['test', 'test 2']
        result_list = 'docs'
        test_np_array = np.array(["ceci est un test", 5, None])
        result_np_array = 'docs'
        test_series = pd.Series(["ceci est un test", 5, None])
        result_series = 'docs'
        test_dataframe = pd.DataFrame([['a', 'b'], ['a', 'b']], columns=['col 1', 'col 2'])
        result_dataframe_col_1 = 'col 1'
        result_dataframe_col_2 = 'col 2'
        test_file = "./testing_file.csv"
        result_file_col_1 = 'col 1'
        result_file_col_2 = 'col 2'

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_column_to_be_processed(test_str), result_str)
        self.assertEqual(utils.get_column_to_be_processed(test_list), result_list)
        self.assertEqual(utils.get_column_to_be_processed(test_np_array), result_np_array)
        self.assertEqual(utils.get_column_to_be_processed(test_series), result_series)
        self.assertEqual(utils.get_column_to_be_processed(test_dataframe), result_dataframe_col_1)
        self.assertEqual(utils.get_column_to_be_processed(test_dataframe, prefered_column='toto'), result_dataframe_col_1)
        self.assertEqual(utils.get_column_to_be_processed(test_dataframe, prefered_column='col 2'), result_dataframe_col_2)
        self.assertEqual(utils.get_column_to_be_processed(test_file), result_file_col_1)
        self.assertEqual(utils.get_column_to_be_processed(test_file, prefered_column='toto'), result_file_col_1)
        self.assertEqual(utils.get_column_to_be_processed(test_file, prefered_column='col 2'), result_file_col_2)

        with self.assertRaises(ValueError):
            utils.get_column_to_be_processed(test_file, first_row='bad_value')
        with self.assertRaises(FileNotFoundError):
            utils.get_column_to_be_processed("imaginary_file.csv")


    def test_regroup_data_series(self):
        '''Testing function utils.regroup_data_series'''
        # Definition d'une fonction à décorer
        def test_function(docs):
            if type(docs) != pd.Series: raise TypeError('')
            return docs.apply(lambda x: x if x in ['avant', 'milieu', 'après'] else 'test')
        # Vals à tester
        docs_test = pd.Series(['avant'] + ["ceci est un test"] * 5000 + ['milieu'] + ["ceci est un test"] * 5000 + ['après'], name='test')
        docs_test_copy = pd.Series(['avant'] + ["ceci est un test"] * 5000 + ['milieu'] + ["ceci est un test"] * 5000 + ['après'], name='test')
        docs_results =  pd.Series(['avant'] + ["test"] * 5000 + ['milieu'] + ["test"] * 5000 + ['après'], name='test')
        data_no_duplicates = pd.Series(['avant'] + ["ceci est un test"] + ['milieu'] + ['après'], name='test')
        data_no_duplicates_results = pd.Series(['avant'] + ["test"] + ['milieu'] + ['après'], name='test')


        # Vérification du fonctionnement type
        pd.testing.assert_series_equal(utils.regroup_data_series(test_function)(docs_test), docs_results)
        pd.testing.assert_series_equal(utils.regroup_data_series(test_function, prefix_text='blabla')(docs_test), docs_results)
        # Vérification non modification input
        _ = utils.regroup_data_series(test_function)(docs_test)
        pd.testing.assert_series_equal(docs_test, docs_test_copy)
        # Vérification fonctionnement quand pas de doublons
        pd.testing.assert_series_equal(utils.regroup_data_series(test_function, min_nb_data=1)(data_no_duplicates), data_no_duplicates_results)


    def test_regroup_data_df(self):
        '''Testing function utils.regroup_data_df'''
        # Definition d'une fonction à wrapper
        def test_function_1(df):
            if type(df) != pd.DataFrame: raise TypeError('')
            df['test1'] = df['test1'].str.replace('toto', 'titi')
            return df
        def test_function_2(df):
            if type(df) != pd.DataFrame: raise TypeError('')
            df['test3'] = df['test2'].str.replace('toto', 'tata')
            return df
        # Vals à tester
        df_test = pd.DataFrame([['toto', 'titi', 'tata'], ['tata', 'toto', 'titi'], ['titi', 'tata', 'toto']]*50000,
                               index=[_ for _ in range(50000*3)], columns=['test1', 'test2', 'test3'])
        df_results_1 =  pd.DataFrame([['titi', 'titi', 'tata'], ['tata', 'toto', 'titi'], ['titi', 'tata', 'toto']]*50000,
                                     index=[_ for _ in range(50000*3)], columns=['test1', 'test2', 'test3'])
        df_results_2 =  pd.DataFrame([['toto', 'titi', 'titi'], ['tata', 'toto', 'tata'], ['titi', 'tata', 'tata']]*50000,
                                     index=[_ for _ in range(50000*3)], columns=['test1', 'test2', 'test3'])


        # Vérification du fonctionnement type
        pd.testing.assert_frame_equal(utils.regroup_data_df(test_function_1)(df_test), df_results_1)
        pd.testing.assert_frame_equal(utils.regroup_data_df(test_function_1, columns_to_be_processed=['test1'])(df_test), df_results_1)
        pd.testing.assert_frame_equal(utils.regroup_data_df(test_function_1, prefix_text='hello')(df_test), df_results_1)
        pd.testing.assert_frame_equal(utils.regroup_data_df(test_function_2)(df_test), df_results_2)
        pd.testing.assert_frame_equal(utils.regroup_data_df(test_function_2, columns_to_be_processed=['test2'])(df_test), df_results_2)

        with self.assertRaises(KeyError):
            utils.regroup_data_df(test_function_2, columns_to_be_processed=['test1'])(df_test)


    def test_get_regex_match_words(self):
        '''Testing function utils.get_regex_match_words'''
        words = ['test?', 'toto']
        expected_result = '(?:^|(?<=[\\.\\?!,;:\\(\\)"\'/<>=\\[\\]\\{\\}\\~\\*\\s]))(test\\?|toto)(?=[\\.\\?!,;:\\(\\)"\'/<>=\\[\\]\\{\\}\\~\\*\\s]|$)'
        expected_result_2 = '(?i)(?:^|(?<=[\\.\\?!,;:\\(\\)"\'/<>=\\[\\]\\{\\}\\~\\*\\s]))(test\\?|toto)(?=[\\.\\?!,;:\\(\\)"\'/<>=\\[\\]\\{\\}\\~\\*\\s]|$)'
        expected_result_3 = '(?:^|(?<=[\\.\\s]))(test\\?|toto)(?=[\\?\\s]|$)'
        expected_result_4 = '(?:^|(?<=[\\.\\?!,;:\\(\\)"\'/<>=\\[\\]\\{\\}\\~\\*\\s]))(test?|toto)(?=[\\.\\?!,;:\\(\\)"\'/<>=\\[\\]\\{\\}\\~\\*\\s]|$)'

        # Vérification du fonctionnement type
        self.assertEqual(utils.get_regex_match_words(words), expected_result)
        # Vérification des kwargs
        self.assertEqual(utils.get_regex_match_words(words, case_insensitive=True), expected_result_2)
        self.assertEqual(utils.get_regex_match_words(words, accepted_char_ahead='.', accepted_char_behind='?'), expected_result_3)
        self.assertEqual(utils.get_regex_match_words(words, words_as_regex=True), expected_result_4)

        # Vérification utilisation as regex
        words_sentence = ['!?+?=test|', 'toto', 'test tata']
        sentence = 'ceci est une totophrase. avec !?+?=test|, ** et toto et aussi test tata!'
        expected_result_sentence = 'ceci est une totophrase. avec , ** et  et aussi !'

        regex = utils.get_regex_match_words(words_sentence)
        result = re.sub(regex, '', sentence)
        self.assertEqual(result, expected_result_sentence)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
