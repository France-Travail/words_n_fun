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

create-virtualenv: ## Creation of a virtual environment
	pip install virtualenv;\
	python -m venv $(NAME_VENV)

init-local-env: ## Initialization of the local dev environment
ifndef VIRTUAL_ENV
	$(error Environnement virtuel python non positionné !)
endif
	pip install -r requirements.txt;\
	python setup.py develop

####################################################
# Tests
####################################################

test: ## Launch python tests
ifndef VIRTUAL_ENV
	$(error Python virtual environment not activated !)
endif
	nosetests -c nose_setup_coverage.cfg tests --exe # https://stackoverflow.com/questions/1457104/nose-unable-to-find-tests-in-ubuntu


####################################################
# Code quality
####################################################

quality: black isort flake8 ## Launch the code quality tools

black: ## Code formatter
ifndef VIRTUAL_ENV
	$(error Python virtual environment not activated !)
endif
	pip install black;\
	black -l 140 -t py38 -S .

isort: ## Utility to automatically sort imports
ifndef VIRTUAL_ENV
	$(error Python virtual environment not activated !)
endif
	pip install isort;\
	isort --skip venv_{{package_name}} -rc .

flake8: ## Guide Enforcement tool
ifndef VIRTUAL_ENV
	$(error Python virtual environment not activated !)
endif
	pip install flake8;\
	flake8 --exclude=venv_{{package_name}} . # add "|| exit 0" to avoid error
