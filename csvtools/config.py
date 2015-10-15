#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This file contains the shared ressources for all csvtools.

Basically the data here are mappings from the internal format of the evaluation framework to human readable latex
sources. It's important that sometimes data is stored at duplicated places because the usage of the data within the
toolkit itself differs.

"""
from __future__ import print_function

from collections import defaultdict

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


# Mapping between Filename and Language.
mapping = {
    'oana-en.xml': 'English',
    'oana-fa.xml': 'Farsi',
    'oana-ro.xml': 'Romanian',
    'oana-sl.xml': 'Slovenian',
    'oana-cs.xml': 'Czech',
    'oana-et.xml': 'Estonian',
    'oana-hu.xml': 'Hungarian',
    'oana-pl.xml': 'Polish',
    'oana-sk.xml': 'Slovak',
    'oana-sr.xml': 'Serbian'
}

# List of Languages used in the evaluation
langs = ['oana-cs.xml', 'oana-en.xml', 'oana-et.xml', 'oana-fa.xml', 'oana-hu.xml', 'oana-pl.xml', 'oana-ro.xml',
         'oana-sr.xml', 'oana-sk.xml', 'oana-sl.xml']

# Conversion from Tagset in Evaluation to human readable Tagset.
tagsets = {
    'MTE': 'MSD',
    'UNIVERSAL': 'Universal'
}

# List of Taggers used in Evaluation
taggers = ['nltkBrill', 'skMultinomialNB', 'skPerceptron', 'skLinearSVC']

# Dicts for storing data for ranked results
rank_mte = {
    'nltkBrill': defaultdict(list),
    'skMultinomialNB': defaultdict(list),
    'skPerceptron': defaultdict(list),
    'skLinearSVC': defaultdict(list)
}
rank_univ = {
    'nltkBrill': defaultdict(list),
    'skMultinomialNB': defaultdict(list),
    'skPerceptron': defaultdict(list),
    'skLinearSVC': defaultdict(list)
}

# Mapping from Tagger-Name to LaTeX-Document String
map_taggers = {
    'nltkBrill': "\\nltk{}-Brill",
    'skMultinomialNB': "\\scikit{}-MultinomialNB",
    'skPerceptron': "\\scikit{}-Perceptron",
    'skLinearSVC': "\\scikit{}-LinearSVC"
}


# Mapping from Language-Filename to Color
colors = {
    'oana-en.xml': 'red',
    'oana-fa.xml': 'green',
    'oana-ro.xml': 'magenta',
    'oana-sl.xml': 'lightgray',
    'oana-cs.xml': 'brown',
    'oana-et.xml': 'olive',
    'oana-hu.xml': 'purple',
    'oana-pl.xml': 'teal',
    'oana-sk.xml': 'lightgray',
    'oana-sr.xml': 'cyan',
    'baseline': 'black',
}


# Mapping from Configuration Option to human readable String
copts2str = {
    'baseline': 'base',
    'brill24': 'br24',
    'fntbl37': 'ftbl37',
    'nltkdemo18': 'nl18',
    'nltkdemo18plus': 'nl18+',
    '[-3, -2, -1, 0, 1, 2, 3]': 'arnd3',
    '[-2, -1, 0, 1, 2]': 'arnd2',
    '[-1, 0, 1]': 'arnd1',
    '[0]': 'base',
    '[-1, 0]': 'left1',
    '[-2, -1, 0]': 'left2',
    '[-3, -2, -1, 0]': 'left3',
    '[0, 1]': 'right1',
    '[0, 1, 2]': 'right2',
    '[0, 1, 2, 3]': 'right3',
}

# List of Configuration Options for Brill
configs_brill = ['fntbl37', 'brill24', 'nltkdemo18', 'nltkdemo18plus']

# List of Configuration Options for ScikitLearn

configs_scikit = ['[-3, -2, -1, 0, 1, 2, 3]',
                  '[-2, -1, 0, 1, 2]',
                  '[-1, 0, 1]',
                  '[0]',
                  '[-1, 0]',
                  '[-2, -1, 0]',
                  '[-3, -2, -1, 0]',
                  '[0, 1]',
                  '[0, 1, 2]',
                  '[0, 1, 2, 3]']

# Path to the Results-File
RESULTS_CSV = '../evaluation_results.csv'
