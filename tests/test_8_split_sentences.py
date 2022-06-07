#!/usr/bin/env python3

## Test - unit test of split sentences functions
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

# utils libs
import os
import numpy as np
import pandas as pd
from words_n_fun.preprocessing import split_sentences

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class SplitSentencesTests(unittest.TestCase):
    '''Main class to test functions in split_sentences'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test_split_sentences_0(self):
        '''On check juste les formats d'entrées de split_sentences'''
        self.assertEqual(split_sentences.split_sentences('test'), ['test'])


    def test_split_sentences_1(self):
        # Vals à tester
        text = "Sous la Direction de M. Untel, vous prenez en charge l'animation des formations en habilitations électriques pour personnel électricien.\nVous participez également à l'adaptation des formations aux évolutions réglementaires et de marché.\nFormations sur sites : déplacements fréquents à prévoir.\n\nDétail de l'Offre :\n- Type de contrat : CDD à la mission ou contrat de prestation\n- Lieu : Région Midi-Pyrénées \n- Rémunération : Selon statut\n\nProfil :\n- BTS, DUT en génie Electrique ou Electrotechnique \n- Expérience avérée dans le domaine de l'électricité\n- Vous avez une expérience de 2 ans minimum dans le domaine de la formation\n- Vous aimez transmettre et veillez à la satisfaction client.\n- Qualité relationnelles et rédactionnelles / Organisation / Rigueur\n- Permis B et véhicule indispensables"
        result = ["Sous la Direction de M. Untel, vous prenez en charge l'animation des formations en habilitations électriques pour personnel électricien.",
                 "Vous participez également à l'adaptation des formations aux évolutions réglementaires et de marché.",
                 'Formations sur sites : déplacements fréquents à prévoir.',
                 "Détail de l'Offre :",
                 '- Type de contrat : CDD à la mission ou contrat de prestation',
                 '- Lieu : Région Midi-Pyrénées',
                 '- Rémunération : Selon statut',
                 'Profil :',
                 '- BTS, DUT en génie Electrique ou Electrotechnique',
                 "- Expérience avérée dans le domaine de l'électricité",
                 '- Vous avez une expérience de 2 ans minimum dans le domaine de la formation',
                 '- Vous aimez transmettre et veillez à la satisfaction client.',
                 '- Qualité relationnelles et rédactionnelles / Organisation / Rigueur',
                 '- Permis B et véhicule indispensables']
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_2(self):
        # Vals à tester
        text = "Malgré la crise sanitaire du COVID19, l'offre d'emploi est maintenue. Le recrutement peut être reporté à une date ultérieure. N'hésitez pas à postuler. \n\nNous recherchons un(e)  audioprothésiste confirmé(e) ou jeune diplômé(e) pour compléter notre équipe. Vous travaillerez  au sein de nos laboratoires de Nort sur Erdre, Savenay et St Nazaire (planning à définir selon l'activité).\n\nTemps partiel possible.\n \nPrésent depuis 2007 en Loire-Atlantique nous vous proposons de rejoindre notre équipe dédiée 100% à l'audition (nous disposons de l'agrément Lyric).\n\nVous aurez la garantie de travailler dans les meilleures conditions possibles avec comme seul objectif la satisfaction du patient.\n\n"
        result = ["Malgré la crise sanitaire du COVID19, l'offre d'emploi est maintenue. ",
                  'Le recrutement peut être reporté à une date ultérieure. ',
                  "N'hésitez pas à postuler.",
                  'Nous recherchons un(e)  audioprothésiste confirmé(e) ou jeune diplômé(e) pour compléter notre équipe. ',
                  "Vous travaillerez  au sein de nos laboratoires de Nort sur Erdre, Savenay et St Nazaire (planning à définir selon l'activité).",
                  'Temps partiel possible.',
                  "Présent depuis 2007 en Loire-Atlantique nous vous proposons de rejoindre notre équipe dédiée 100% à l'audition (nous disposons de l'agrément Lyric).",
                  'Vous aurez la garantie de travailler dans les meilleures conditions possibles avec comme seul objectif la satisfaction du patient.']
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_3(self):
        # Vals à tester
        text = "CC3F. Vos missions : en capacité, en cuisine, de préparer les ingrédients des burritos que nous servons, accueillir les clients, être à l'ouverture ou à la fermeture du restaurant. Vous encaisserez les clients. Vous serez également en capacité de suivre les stocks et de réceptionner les marchandises. Vous respecterez les procédures de caisse, de coffre et de dépôts et alerterez la hiérarchie si anomalie. Vous respecterez les règles d'hygiène et de sécurité. Vous contribuerez à la gestion des déchets. Vous superviserez l'équipe et formerez les nouveaux employés. vous suivrez une formation adaptée à nos besoins, avant embauche. Horaires du restaurant, i.e. 8h30/23h (voire plus), du lundi au samedi. Vous travaillerez avec des horaires en coupure. Contrats entre 15h et 24h/s. Etre véhiculé(e) est un plus. Prise de poste courant avril, après processus de recrutement et formation."
        result = ['CC3F. ',
                  "Vos missions : en capacité, en cuisine, de préparer les ingrédients des burritos que nous servons, accueillir les clients, être à l'ouverture ou à la fermeture du restaurant. ",
                  'Vous encaisserez les clients. ',
                  'Vous serez également en capacité de suivre les stocks et de réceptionner les marchandises. ',
                  'Vous respecterez les procédures de caisse, de coffre et de dépôts et alerterez la hiérarchie si anomalie. ',
                  "Vous respecterez les règles d'hygiène et de sécurité. ",
                  'Vous contribuerez à la gestion des déchets. ',
                  "Vous superviserez l'équipe et formerez les nouveaux employés. ",
                  'vous suivrez une formation adaptée à nos besoins, avant embauche. ',
                  'Horaires du restaurant, i.e. 8h30/23h (voire plus), du lundi au samedi. ',
                  'Vous travaillerez avec des horaires en coupure. ',
                  'Contrats entre 15h et 24h/s. ',
                  'Etre véhiculé(e) est un plus. ',
                  'Prise de poste courant avril, après processus de recrutement et formation.']
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_4(self):
        # Vals à tester
        text = "Laurent, le Responsable et son équipe de 6 personnes vous attendent pour participer au bon fonctionnement des réseaux d'alimentation en eau potable et assainissement sur le périmètre de La Roche sur Yon. \n\nVous réalisez l'entretien et la maintenance des réseaux d'eau et des équipements hydrauliques.\nVous effectuez les travaux de terrassement pour la pose et le renouvellement des canalisations sur les chantiers\nVous contribuez à la réparation des fuites d'eau \nVous réalisez la maintenance des équipements sur le réseau : Poteaux incendie, châteaux d'eau, purges, ventouse, stabilisateurs de pression, débitmètres....\n\nLe respect des règles de sécurité est fondamental sur le poste.\n\nL'astreinte vous amènera à intervenir en cas d'urgence pour garantir la continuité du service (permis B + véhicule de service).\n\nMalgré la crise sanitaire du COVID19, le process de recrutement est maintenu dans le respect des règles sanitaires en vigueur. N'hésitez pas à postuler."
        result = ["Laurent, le Responsable et son équipe de 6 personnes vous attendent pour participer au bon fonctionnement des réseaux d'alimentation en eau potable et assainissement sur le périmètre de La Roche sur Yon.",
                  "Vous réalisez l'entretien et la maintenance des réseaux d'eau et des équipements hydrauliques.",
                  'Vous effectuez les travaux de terrassement pour la pose et le renouvellement des canalisations sur les chantiers',
                  "Vous contribuez à la réparation des fuites d'eau",
                  "Vous réalisez la maintenance des équipements sur le réseau : Poteaux incendie, châteaux d'eau, purges, ventouse, stabilisateurs de pression, débitmètres....",
                  'Le respect des règles de sécurité est fondamental sur le poste.',
                  "L'astreinte vous amènera à intervenir en cas d'urgence pour garantir la continuité du service (permis B + véhicule de service).",
                  'Malgré la crise sanitaire du COVID19, le process de recrutement est maintenu dans le respect des règles sanitaires en vigueur. ',
                  "N'hésitez pas à postuler."]
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_5(self):
        # Vals à tester
        text = "Vous effectuez les travaux en vert de la vigne :  \nÉbourgeonnage\nEpillonage\nRelevage\nBinage\nContrat saisonnier, prise de poste fin avril ou début mai selon météo. Vous avez impérativement votre moyen de locomotion pour accéder seul aux parcelles de l'exploitation. ** Aucun logement possible sur place ** \nL'employeur déclare mettre en œuvre les mesures sanitaires pour préserver la santé et la sécurité de ses salariés.\n"
        result = ['Vous effectuez les travaux en vert de la vigne :',
                  'Ébourgeonnage',
                  'Epillonage',
                  'Relevage',
                  'Binage',
                  'Contrat saisonnier, prise de poste fin avril ou début mai selon météo. ',
                  "Vous avez impérativement votre moyen de locomotion pour accéder seul aux parcelles de l'exploitation. ",
                  '** Aucun logement possible sur place **',
                  "L'employeur déclare mettre en œuvre les mesures sanitaires pour préserver la santé et la sécurité de ses salariés."]
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_6(self):
        # Vals à tester
        text = "L'agence d'emploi (CDI, intérim et formation) Targett Interim (62970) recherche pour un de ses clients un « Poseur d'enseignes lumineuses » H/F \n\nAu sein d'une entreprise spécialisée dans la réalisation et la pose d'enseignes lumineuses, vous aurez pour mission: \n- Pose d'enseignes ou d'adhésifs sur façade de bâtiment et magasin.\n- Travail en équipe de 2 avec un chef d'équipe\n- Utilisation de petits outillages\n- Poste parfois en hauteur\n- Contact clientèle\n- Respect des règles de sécurité\n\nPoste à pourvoir dès que possible pour une période de 6 mois renouvelable. \n\nVous souhaitez intégrer une agence d'intérim qui est à votre écoute, réactive dès que vous avez besoin d'un document administratif, qui fait tout pour vous trouver le poste de vos rêves ? Vous en avez marre d'être considéré comme un numéro et vous voulez avoir un sentiment d'appartenance ? Alors rejoignez-nous ! \nTargett Interim \ns.vaillant@outlook.fr"
        result = ["L'agence d'emploi (CDI, intérim et formation) Targett Interim (62970) recherche pour un de ses clients un « Poseur d'enseignes lumineuses » H/F",
                  "Au sein d'une entreprise spécialisée dans la réalisation et la pose d'enseignes lumineuses, vous aurez pour mission:",
                  "- Pose d'enseignes ou d'adhésifs sur façade de bâtiment et magasin.",
                  "- Travail en équipe de 2 avec un chef d'équipe",
                  '- Utilisation de petits outillages',
                  '- Poste parfois en hauteur',
                  '- Contact clientèle',
                  '- Respect des règles de sécurité',
                  'Poste à pourvoir dès que possible pour une période de 6 mois renouvelable.',
                  "Vous souhaitez intégrer une agence d'intérim qui est à votre écoute, réactive dès que vous avez besoin d'un document administratif, qui fait tout pour vous trouver le poste de vos rêves ? ",
                  "Vous en avez marre d'être considéré comme un numéro et vous voulez avoir un sentiment d'appartenance ? ",
                  'Alors rejoignez-nous !',
                  'Targett Interim',
                  's.vaillant@outlook.fr']
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_7(self):
        # Vals à tester
        text = "Communication: accueil, écoute, transmission d'informations aux partenaires responsables, dialogue, sécurisation de l'enfant.\n\nEducation: apprentissage des règles d'hygiène corporelle de vie en collectivité, aide à l'acquisition des fonctions sensorielles et motrices de l'autonomie, aide au développement affectif et individuel.\n\nExécution: soins d'hygiène corporelle, selon le milieu collectif : préparation, distribution et aide à la prise des repas, entretien courant des locaux et des équipements, fabrication d'éléments simples et aménagement des espaces.\n\nOrganisation: des activités en fonction des besoins des enfants et de la collectivité, du poste de travail, gestion des stocks et des matériaux\n\nVous êtes titulaire du diplôme d'état de puériculture. "
        result = ["Communication: accueil, écoute, transmission d'informations aux partenaires responsables, dialogue, sécurisation de l'enfant.",
                  "Education: apprentissage des règles d'hygiène corporelle de vie en collectivité, aide à l'acquisition des fonctions sensorielles et motrices de l'autonomie, aide au développement affectif et individuel.",
                  "Exécution: soins d'hygiène corporelle, selon le milieu collectif : préparation, distribution et aide à la prise des repas, entretien courant des locaux et des équipements, fabrication d'éléments simples et aménagement des espaces.",
                  'Organisation: des activités en fonction des besoins des enfants et de la collectivité, du poste de travail, gestion des stocks et des matériaux',
                  "Vous êtes titulaire du diplôme d'état de puériculture."]
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_8(self):
        # Vals à tester
        text = "POSTE  URGENT\nPoste (pour la durée de la crise) en remplacement (pour la durée de la crise) sanitaire.\nmissions recentrées prioritairement autour de l'entretien des locaux (avec renforcement des procédures d'hygiène en lien avec la situation sanitaire! , service des repas, vaisselle... ).\n"
        result = ['POSTE  URGENT',
                  'Poste (pour la durée de la crise) en remplacement (pour la durée de la crise) sanitaire.',
                  "missions recentrées prioritairement autour de l'entretien des locaux (avec renforcement des procédures d'hygiène en lien avec la situation sanitaire! , service des repas, vaisselle... )."]
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_9(self):
        # Vals à tester
        text = "* Poste réservé aux personnes titulaires d'une Reconnaissance en Qualité Travailleur Handicapé (RQTH) en cours de validité délivrée par la MDPH\n***\nAu sein d'une usine de fabrication de maroquinerie en cuir vous réalisez : \n- de la coupe et emboutissage de pièces\n- de la mise en épaisseur des matières\n- du thermocollage, encollage\n- de la teinte sur certaines zones du produit\n- de la préparation de lots de pièces complets avant l'assemblage définitif\n\nProfil recherché : Vous avez idéalement un diplôme en couture/maroquinerie ou une expérience industrielle dans le secteur de la maroquinerie. Vous faites preuve de rigueur, de minutie et de dextérité. \nPassionné(e) par le cuir et s'inscrivant dans une démarche Qualité et d'Amélioration Continue permanente et individuelle, vous souhaitez participer à un véritable projet d'entreprise.\nHoraires : de 15h00 à 23h00 du Lundi au Jeudi et de 11h00 à 17h00 le Vendredi.\n"
        result = ["* Poste réservé aux personnes titulaires d'une Reconnaissance en Qualité Travailleur Handicapé (RQTH) en cours de validité délivrée par la MDPH",
                  '***',
                  "Au sein d'une usine de fabrication de maroquinerie en cuir vous réalisez :",
                  '- de la coupe et emboutissage de pièces',
                  '- de la mise en épaisseur des matières',
                  '- du thermocollage, encollage',
                  '- de la teinte sur certaines zones du produit',
                  "- de la préparation de lots de pièces complets avant l'assemblage définitif",
                  'Profil recherché : Vous avez idéalement un diplôme en couture/maroquinerie ou une expérience industrielle dans le secteur de la maroquinerie. ',
                  'Vous faites preuve de rigueur, de minutie et de dextérité.',
                  "Passionné(e) par le cuir et s'inscrivant dans une démarche Qualité et d'Amélioration Continue permanente et individuelle, vous souhaitez participer à un véritable projet d'entreprise.",
                  'Horaires : de 15h00 à 23h00 du Lundi au Jeudi et de 11h00 à 17h00 le Vendredi.']
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_10(self):
        # Vals à tester
        text = "-  Proposer une solution efficace aux demandes de vos clients, dans le respect des procédures (informations, administratif, suivi contractuel, facturation\n-  Traiter leurs réclamation\n-  Garantir la satisfaction et la fidélisation des clients au quotidien\nNotre client vous garantit :\n-  Une entreprise à l'écoute, où tout le monde a son rôle à jouer et peut proposer ses idées.\n-  Une proximité avec les managers dans une démarche d'amélioration continue, avec une vraie culture de bonnes pratiques et de partage.\n-  Des positions de travail étudiées pour vous assurer le meilleur confort\nPoste à temps complet, 35h/semaine\nAmplitude horaires : du lundi au vendredi de 8h30 à 19h et le samedi de 8h30 à 18h\nAccessible en Transports en Communs \nVotre profil :\n- Vous avez le sourire,\n-  Vous êtes ponctuel,\n-  Vous avez l'envie et la volonté,\n-  Vos capacités d'écoute et votre sens du service sont des atouts que l'on vous reconnaît,\n-  Vous êtes rigoureux et organisé,\n-  Vous êtes à l'aise avec la gestion des oup informatiques,\n-  Vous vous exprimez parfaitement à l'écrit comme à l'oral\n-  Vous êtes disponible et motivé\nExpérience:\n- conseiller client h/f ou similaire: 1 an, débutant ayant affinité avec la relation client accepté\nAccompagné tout au long de votre parcours d'intégration, vous maitriserez rapidement la relation client et les différents produits. Le suivi personnalisé par une équipe de formateurs confirmés permettra d'optimiser votre montée en compétence.\nLe savoir être est un élément essentiel du recrutement pour ce poste \nRémunération et avantages :\n-  Taux horaire fixe + 10% de fin de mission + 10% de congés payés\n-  Primes collective et/ou individuelle + participation aux bénéfices + CET 5%\n-  Acompte de paye à la semaine si besoin,\n-  Possibilité d'intégration rapide, de formation et d'évolution,\n-  Bénéficiez d'aides et de services dédiés (mutuelle, logement, garde enfant, déplacement )."
        result = ['-  Proposer une solution efficace aux demandes de vos clients, dans le respect des procédures (informations, administratif, suivi contractuel, facturation',
                  '-  Traiter leurs réclamation',
                  '-  Garantir la satisfaction et la fidélisation des clients au quotidien',
                  'Notre client vous garantit :',
                  "-  Une entreprise à l'écoute, où tout le monde a son rôle à jouer et peut proposer ses idées.",
                  "-  Une proximité avec les managers dans une démarche d'amélioration continue, avec une vraie culture de bonnes pratiques et de partage.",
                  '-  Des positions de travail étudiées pour vous assurer le meilleur confort',
                  'Poste à temps complet, 35h/semaine',
                  'Amplitude horaires : du lundi au vendredi de 8h30 à 19h et le samedi de 8h30 à 18h',
                  'Accessible en Transports en Communs',
                  'Votre profil :',
                  '- Vous avez le sourire,',
                  '-  Vous êtes ponctuel,',
                  "-  Vous avez l'envie et la volonté,",
                  "-  Vos capacités d'écoute et votre sens du service sont des atouts que l'on vous reconnaît,",
                  '-  Vous êtes rigoureux et organisé,',
                  "-  Vous êtes à l'aise avec la gestion des oup informatiques,",
                  "-  Vous vous exprimez parfaitement à l'écrit comme à l'oral",
                  '-  Vous êtes disponible et motivé',
                  'Expérience:',
                  '- conseiller client h/f ou similaire: 1 an, débutant ayant affinité avec la relation client accepté',
                  "Accompagné tout au long de votre parcours d'intégration, vous maitriserez rapidement la relation client et les différents produits. ",
                  "Le suivi personnalisé par une équipe de formateurs confirmés permettra d'optimiser votre montée en compétence.",
                  'Le savoir être est un élément essentiel du recrutement pour ce poste',
                  'Rémunération et avantages :',
                  '-  Taux horaire fixe + 10% de fin de mission + 10% de congés payés',
                  '-  Primes collective et/ou individuelle + participation aux bénéfices + CET 5%',
                  '-  Acompte de paye à la semaine si besoin,',
                  "-  Possibilité d'intégration rapide, de formation et d'évolution,",
                  "-  Bénéficiez d'aides et de services dédiés (mutuelle, logement, garde enfant, déplacement )."]
        self.assertEqual(split_sentences.split_sentences(text), result)


    def test_split_sentences_df(self):
        '''Fonction pour vérifier le fonctionnement de la fonction split_sentences_df'''

        # Get df to test & result
        df = pd.DataFrame({
            'OFF_CLE' : ['c1', 'c2'],
            'OFF_DESCRIPTION' : ['t1\nt2', 't3\nt4. t5'],
        })
        df2 = df.copy()
        df_result = pd.DataFrame({
            'OFF_CLE' : ['c1', 'c1', 'c2', 'c2', 'c2'],
            'OFF_DESCRIPTION' : ['t1', 't2', 't3', 't4. ', 't5'],
        })  # Offres PE en premier

        # split_sentences_df(df, col)
        df_processed = split_sentences.split_sentences_df(df, 'OFF_DESCRIPTION')
        pd.testing.assert_frame_equal(df_processed, df_result)
        pd.testing.assert_frame_equal(df, df2)  # On check si pas de modif. sur df original



# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
