#!/usr/bin/env python3

## Stopwords management functions
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
#
# Functions :
# - remove_stopwords
# - stopwords_ascii
# - stopwords_nltk
# - stopwords_nltk_ascii


import os
import nltk
import pandas as pd
from typing import Union

from words_n_fun import utils
from words_n_fun.preprocessing import basic

# Get logger
import logging

logger = logging.getLogger(__name__)

# Setting the path where the nltk stopwords data is located
stopwords_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'nltk_data')
nltk.data.path.append(stopwords_dir_path)


# From R stopwords package (wrapper around https://github.com/stopwords-iso/stopwords-iso)
STOPWORDS = ["a", "abord", "absolument", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait",
             "allaient", "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après",
             "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui", "aupres", "auquel",
             "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront",
             "aussi", "autre", "autrefois", "autrement", "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient",
             "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant", "ayez", "ayons",
             "b", "bah", "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr",
             "c", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là",
             "celui", "celui-ci", "celui-là", "celà", "cent", "cependant", "certain", "certaine", "certaines",
             "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque",
             "cher", "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante",
             "cinquantième", "cinquième", "clac", "clic", "combien", "comme", "comment", "comparable", "comparables",
             "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de", "debout", "dedans", "dehors",
             "deja", "delà", "depuis", "dernier", "derniere", "derriere", "derrière", "des", "desormais", "desquelles",
             "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra",
             "devrait", "different", "differentes", "differents", "différent", "différente", "différentes",
             "différents", "dire", "directe", "directement", "dit", "dite", "dits", "divers", "diverse", "diverses",
             "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc", "dont", "dos", "douze",
             "douzième", "dring", "droite", "du", "duquel", "durant", "dès", "début", "désormais", "e", "effet",
             "egale", "egalement", "egales", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin",
             "entre", "envers", "environ", "es", "essai", "est", "et", "etant", "etc", "etre", "eu", "eue", "eues",
             "euh", "eurent", "eus", "eusse", "eussent", "eusses",  "eussiez", "eussions", "eut", "eux", "eux-mêmes",
             "exactement", "excepté", "extenso", "exterieur", "eûmes", "eût", "eûtes", "f", "fais", "faisaient",
             "faisant", "fait", "faites", "façon", "feront", "fi", "flac", "floc",  "fois", "font", "force", "furent",
             "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gens",
             "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp", "hue",
             "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i", "ici", "il", "ils", "importe", "j", "je",
             "jusqu", "jusque", "juste", "k", "l", "la", "laisser", "laquelle", "las", "le", "lequel", "les",
             "lesquelles", "lesquels", "leur", "leurs", "longtemps", "lors", "lorsque", "lui", "lui-meme", "lui-même",
             "là", "lès", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "maximale", "me", "meme",
             "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "mine", "minimale", "moi",
             "moi-meme", "moi-même", "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même",
             "mêmes", "n", "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire",
             "necessairement", "neuf", "neuvième", "ni", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment",
             "notre", "nous", "nous-mêmes", "nouveau", "nouveaux", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh",
             "ohé", "ollé", "olé", "on", "ont", "onze", "onzième", "ore", "ou", "ouf", "ouias", "oust", "ouste",
             "outre", "ouvert", "ouverte", "ouverts", "o", "où", "p", "paf", "pan", "par", "parce", "parfois", "parle",
             "parlent", "parler", "parmi", "parole", "parseme", "partant", "particulier", "particulière",
             "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne", "personnes", "peu", "peut",
             "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "pièce", "plein", "plouf", "plupart", "plus",
             "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi",
             "pourrais", "pourrait", "pouvait", "prealable", "precisement", "premier", "première", "premièrement",
             "pres", "probable", "probante", "procedant", "proche", "près", "psitt", "pu", "puis", "puisque", "pur",
             "pure", "q", "qu", "quand", "quant", "quant-à-soi", "quanta", "quarante", "quatorze", "quatre",
             "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles",
             "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare",
             "rarement", "rares", "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste",
             "restent", "restrictif", "retour", "revoici", "revoilà", "rien", "s", "sa", "sacrebleu", "sait", "sans",
             "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient", "semble", "semblent",
             "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez",
             "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si", "sien", "sienne", "siennes",
             "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient", "sois", "soit", "soixante", "sommes",
             "son", "sont", "sous", "souvent", "soyez", "soyons", "specifique", "specifiques", "speculatif", "stop",
             "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit", "suivant", "suivante",
             "suivantes", "suivants", "suivre", "sujet", "superpose", "sur", "surtout", "t", "ta", "tac", "tandis",
             "tant", "tardive", "te", "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente",
             "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant",
             "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres", "trois",
             "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes",
             "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "valeur", "vas", "vers", "via", "vif",
             "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà", "vont", "vos",
             "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z", "zut", "à", "â", "ça",
             "ès", "étaient", "étais", "était", "étant", "état", "étiez", "étions", "été", "étée", "étées", "étés",
             "êtes", "être", "ô"]

# Specific Pôle Emploi stopwords #1
STOPWORDS_OFFRES_1 = ["recherche", "recherchons", "mission", "missions", "poste", "recrute", "recrutons"]

# Specific Pôle Emploi stopwords #1
STOPWORDS_OFFRES_2 = ["expérience", "assurer", "assurez", "travaux"]


@utils.data_agnostic
@utils.regroup_data_series
def remove_stopwords(docs: pd.Series, opt: str = 'all', set_to_add: Union[list, None] = None,
                     set_to_remove: Union[list, None] = None) -> pd.Series:
    '''Stopwords removal

    Args:
        docs (pd.Series): Documents to process
    Kwargs:
        opt (str): Specifies which stopwords set to use (def='all')
        set_to_add (list): Additionnal stopwords to look for and remove
        set_to_remove (list): Words existing in the stopwords set that should not be removed
    Returns:
        pd.Series: Modified documents
    '''
    logger.debug('Calling stopwords.remove_stopwords')
    if set_to_add is None:
        set_to_add = []
    if set_to_remove is None:
        set_to_remove = []
    # Check if everything is in lowercase (NaNs are replaced, letters are kept)
    if docs.fillna('').str.replace(r"[^A-Za-z]", '').replace('', 'placeholder').str.islower().sum() != docs.shape[0]:
        logger.warning(docs)
        logger.warning('Some characters appear to be in uppercase, stopwords are in lowercase only.')
    # Common soptwords lists
    usage = {
        'none': set(),
        'iso': set().union(STOPWORDS, stopwords_ascii()),
        'nltk': set().union(stopwords_nltk(), stopwords_nltk_ascii()),
        'offres_pe': set().union(STOPWORDS_OFFRES_1, STOPWORDS_OFFRES_2),
        'all': set().union(STOPWORDS, stopwords_ascii(), stopwords_nltk(), stopwords_nltk_ascii()),
    }

    if opt in usage.keys():
        stopwords_list = list(usage.get(opt))
    else:
        logger.warning(
            f"Option {opt} does not exist " +
            f"Existing options are : {', '.join(usage.keys())}. " +
            "By default, all the stopwords are used."
        )
        stopwords_list = STOPWORDS
    # Add custom set
    if len(set_to_add) != 0:
        stopwords_list = list(set(stopwords_list + set_to_add))
    # Remove unwanted words
    if len(set_to_remove) != 0:
        stopwords_list = list(set(stopwords_list) - set(set_to_remove))
    # Empty list case
    if len(stopwords_list) == 0:
        logger.warning("Stopwords_list is empty.")
        logger.warning("Non strings entries are still replaced by None.")
        return docs.apply(lambda x: x if isinstance(x, str) else None)

    regex = utils.get_regex_match_words(stopwords_list)
    return docs.str.replace(regex, '')


def stopwords_ascii() -> list:
    ''' Returns stopwords list in ASCII format (without any special character nor accents)

    Returns:
        list: Stopwords in ASCII format
    '''
    logger.debug('Calling stopwords.stopwords_ascii')
    # Process
    return basic.remove_accents(STOPWORDS, use_tqdm=False)


def stopwords_nltk(try_update: bool = False) -> list:
    ''' Returns the list of FRENCH stopwords from the NLTK package.

    Kwargs:
        try_update (bool): Controls whether we shoud try to update (download required) the stopwords list.
    Returns:
        list: FRENCH stopwords from the NLTK package
    '''
    logger.debug('Calling stopwords.stopwords_nltk')
    if try_update:
        logger.debug("Trying to download an up to date list from NLTK.")
        nltk.download('stopwords', quiet=True)
    return nltk.corpus.stopwords.words('french')


def stopwords_nltk_ascii() -> list:
    ''' Returns the list of FRENCH stopwords from the NLTK package in ASCII format.

    Kwargs:
        try_update (bool): Controls whether we shoud try to update (download required) the stopwords list.
    Returns:
        list: FRENCH stopwords from the NLTK package in ASCII format
    '''
    logger.debug('Calling stopwords.stopwords_nltk_ascii')
    # Process
    return basic.remove_accents(stopwords_nltk(), use_tqdm=False)


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
