## Setup - Procédure d'installation pip
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
import os
from setuptools import setup

# Get package version
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.txt'), 'r') as version_file:
    version = version_file.read().strip()

version = os.getenv('VERSION') or version+'-local'

# Setup
setup(
    name='words_n_fun',
    version=version,
    packages=['pe_semantic', 'pe_semantic.preprocessing'],
    license='Aucune',
    author='Agence Data Services PE Nantes',
    description="Semantic library of the Data Services agency",
    platforms=['windows', 'linux'],
    package_data={
        'pe_semantic': ['configs/*.json', 'nltk_data/corpora/stopwords/french']
    },
    include_package_data=True,
    install_requires=[
        'pandas==1.3.5',
        'numpy==1.19.5',
        'nltk>=3.4.5,<3.6',
        'tqdm==4.62.2',  #https://github.com/tqdm/tqdm/issues/780
        'simplejson>=3.17.0,<3.17.3',
        'requests>=2.23.0,<2.25.1',
        'ftfy>=5.8,<6.0',
    ],
    extras_require={
        "lemmatizer": ["spacy==3.2.4", "markupsafe==2.0.1", "Cython==0.29.24", "fr-core-news-sm==3.2.0"]
    }
)
