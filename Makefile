## Makefile - Outils facilitant la gestion de l'environnement
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
####################################################
# Initialisation de l'environnement de dev en local
####################################################

NAME_VENV=venv_wnf

create-virtualenv: ## Création d'un environnement virtuel
	pip install virtualenv;\
	python -m venv $(NAME_VENV)

init-local-env: ## Initialisation de l'environnement de dev en local
ifndef VIRTUAL_ENV
	$(error Environnement virtuel python non positionné !)
endif
	pip install -r requirements.txt;\
	python setup.py develop

####################################################
# Tests
####################################################

test:
	nosetests -c nose_setup_coverage.cfg tests


####################################################
# Code quality
####################################################

quality: black isort flake8 ## Lance les outils d'aide à la qualité du code

black: ## Formatter
ifndef VIRTUAL_ENV
	$(error Environnement virtuel python non positionné !)
endif
	pip install black;\
	black -l 140 -t py37 -S .

isort: ## Formatter pour les imports
ifndef VIRTUAL_ENV
	$(error Environnement virtuel python non positionné !)
endif
	pip install isort;\
	isort --skip venv_words_n_fun -rc .

flake8: ## Mesure la qualité du code
ifndef VIRTUAL_ENV
	$(error Environnement virtuel python non positionné !)
endif
	pip install flake8;\
	flake8 --exclude=venv_words_n_fun .

setup_git_pre_commit_hook: ## Initialise hook precommit quality
ifndef VIRTUAL_ENV
	$(error Environnement virtuel python non positionné !)
endif
	pip install pre-commit==2.3.0
	pre-commit install
	pip install seed-isort-config==2.1.1
	-seed-isort-config
