#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX Tables with the Spearman Correlation Coefficient between all Configurations, Taggers and Tagsets for each Language..
"""
from __future__ import print_function

from collections import defaultdict
import csv

from scipy.stats import spearmanr
from config import mapping, langs, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"

metric = dict()
rank = defaultdict(list)

with open(RESULTS_CSV, 'rb') as csvfile:
    result = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in result:

        # exclude out of vocabulary lines
        if row['tagger'] == 'tmpOOV':
            continue

        # save all things from different lines into metric, such that
        # we know when we have gathered all lines that belong together
        metric['corpus'] = row['corpus']
        metric['tagger'] = row['tagger']
        metric['tagset'] = row['tagset']
        metric['config options'] = row['config options']
        metric[row['metric']] = float(row['mean'])

        # make sure that we have all lines that belong together
        if all(test in metric for test in
               ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):
            rank[row['corpus']].append(dict(metric))
            metric = dict()

# sort first language according to f1-score
rank[langs[0]].sort(key=lambda a: 1 - a['f1'])

# sort other configs in the same order as the languages are in the first config
for language in langs:
    # exclude first config from sorting
    if language == langs[0]:
        continue
    # reorder the current config
    tmpList = []

    for metric in rank['oana-cs.xml']:
        tmpList.append(filter(lambda a: a['tagger'] == metric['tagger']
                                        and a['tagset'] == metric['tagset']
                                        and a['config options'] == metric['config options'], rank[language])[0])
    rank[language] = tmpList


def header(cfgs):
    print(
    """\\begin{table}
       \\begin{tabular}{@{} cl*{%s}c @{}}\\\\""" % len(cfgs))


def footer(cfgs):
    lastline = "        "
    for i in range(0, len(cfgs) - 1):
        lastline = lastline + " & \\rot{\\textbf{%s}}" % (mapping[cfgs[i]])
    print(lastline)
    print(
    """   \end{tabular}
       \caption{Spearman Correlation across all Configurations, Taggers and Tagsets for all Languages}
       \label{table:spearmanConfigLanguages}
   \end{table}""")


def line(cfg, values, lnumber):
    if lnumber > 0:
        ln = "      \\textbf{%s}" % mapping[cfg]
        for i in range(lnumber):
            ln = ln + " & {:1.2f}".format(values[i])
        for i in range(lnumber, len(values) - 1):
            ln = ln + " &"
        print(ln + "\\\\")

# create list of lists
spear_input = []
for language in langs:
    spear_input.append(map(lambda a: a['f1'], rank[language]))
# transpose lists for having the correct parameter for the spearman method
spear_input = map(list, zip(*spear_input))

# actual printing is done here
# mte
header(langs)
sp = spearmanr(spear_input)[0]
i = 0
for (lng, row) in zip(langs, sp):
    line(lng, row, i)
    i = i + 1
footer(langs)
