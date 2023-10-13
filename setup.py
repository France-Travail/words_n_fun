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

import os
from setuptools import setup

# Get package directory
package_directory = os.path.dirname(os.path.abspath(__file__))

# Get package version (env variable or version file + -local)
version_path = os.path.join(package_directory, 'version.txt')
with open(version_path, 'r') as version_file:
    version = version_file.read().strip()
version = os.getenv('VERSION') or f"{version}+local"

# Get package description
readme_path = os.path.join(package_directory, 'README.md')
with open(readme_path, 'r') as readme_file:
    long_description = readme_file.read()

# Setup
setup(
    name='words_n_fun',
    version=version,
    packages=['words_n_fun', 'words_n_fun.preprocessing'],
    license='AGPL-3.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Agence Data Services PE Nantes",
    author_email="contactadsaiframeworks.00619@pole-emploi.fr",
    description="Semantic library of the Data Services agency",
    url="https://github.com/OSS-Pole-Emploi/words_n_fun",
    platforms=['windows', 'linux'],
    python_requires='>=3.8',
    package_data={
        'words_n_fun': ['configs/*.json', 'nltk_data/corpora/stopwords/french']
    },
    include_package_data=True,
    install_requires=[
        'pandas>=1.3,<1.5; python_version < "3.10"',
        'pandas>=1.3; python_version >= "3.10"',
        'numpy>=1.19,<1.24; python_version < "3.10"',
        'numpy>=1.19; python_version >= "3.10"',
        'nltk>=3.4',
        'ftfy>=5.8',
        'tqdm>=4.40',
        'simplejson>=3.17', 
        'requests>=2.23',
    ],
    extras_require={
        "lemmatizer": ["spacy>=3.7.1", "markupsafe>=2.1.3", "Cython>=3.0.3"]
    }
    # pip install words_n_fun || pip install words_n_fun[lemmatizer]
)
