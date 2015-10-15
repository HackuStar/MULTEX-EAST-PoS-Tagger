#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX Tables with the Spearman Correlation Coefficient between all Configurations for a Tagger and a Tagset.
"""
from __future__ import print_function

from collections import defaultdict
import csv

from scipy.stats import spearmanr
from config import taggers,configs_brill,configs_scikit, rank_mte, rank_univ, copts2str, map_taggers, RESULTS_CSV

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
        if all(test in metric for test in ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):
            # separate MSD tags from universal, so that we can compute spearman for both
            # and separate the taggers, too
            curr_rank = None
            if row['tagset'] == "mte":
                curr_rank = rank_mte[row['tagger']]
            else:
                curr_rank = rank_univ[row['tagger']]

            curr_rank[row['config options']].append(dict(metric))
            last = row['tagger']
            last_tag = row['tagset']
            metric = dict()

# sort first config of each tagger according to f1-score
for tagger in taggers:
    configs = None
    if tagger == taggers[0]:
        configs = configs_brill
    else:
        configs = configs_scikit
    rank_mte[tagger][configs[0]].sort(key=lambda a: 1-a['f1'])
    rank_univ[tagger][configs[0]].sort(key=lambda a: 1-a['f1'])

# sort other configs in the same order as the languages are in the first config
for tagger in taggers:
    if tagger == taggers[0]:
        configs = configs_brill
    else:
        configs = configs_scikit

    for config in configs:
        # exclude first config from sorting
        if config == configs[0]:
            continue
        # reorder the current config
        tmpList_mte = []
        tmpList_univ = []
        
        for lang in rank_mte[tagger][configs[0]]:
            tmpList_mte.append(filter(lambda a: a['corpus'] == lang['corpus'], rank_mte[tagger][config])[0])
            tmpList_univ.append(filter(lambda a: a['corpus'] == lang['corpus'], rank_univ[tagger][config])[0])

        rank_mte[tagger][config] = tmpList_mte
        rank_univ[tagger][config] = tmpList_univ

def header(cfgs):
    hdr = """\\begin{table}
    \\begin{tabular}{@{} cl*{%s}c @{}}
        """ % len(cfgs)
    for i in range(len(cfgs)):
        hdr = hdr + " & \\rot{\\textbf{%s}}" % copts2str[cfgs[i]]

    print(hdr + "\\\\")

def footer(tagger, isMsdTagset):
    print("""   \end{tabular}
    \caption{Spearman Correlation across all Configurations for the %s Tagger and the %s Tagset}
    \label{table:spearmanConfigTagger_%s}
\end{table}""" % (map_taggers[tagger], "MSD" if isMsdTagset else "Universal", tagger))

def line(cfg, values):
    ln = "      \\textbf{%s}" % copts2str[cfg]
    for i in range(len(values)):
        ln = ln + " & {:1.2f}".format(values[i])
    print(ln + "\\\\")


# create tables
for tagger in taggers:
    if tagger == taggers[0]:
        configs = configs_brill
    else:
        configs = configs_scikit

    # create list of lists
    spear_input_mte = []
    spear_input_univ = []
    for config in configs:
        spear_input_mte.append(map(lambda a: a['f1'], rank_mte[tagger][config]))
        spear_input_univ.append(map(lambda a: a['f1'], rank_univ[tagger][config]))
    # transpose lists for having the correct parameter for the spearman method
    spear_input_mte = map(list, zip(*spear_input_mte))
    spear_input_univ = map(list, zip(*spear_input_univ))
    
    # actual printing is done here
    #mte
    header(configs)
    sp_mte = spearmanr(spear_input_mte)[0]
    for (cfg, row) in zip(configs, sp_mte):
        line(cfg, row)
    footer(tagger, True)
    #univ
    header(configs)
    sp_univ = spearmanr(spear_input_univ)[0]
    for (cfg, row) in zip(configs, sp_univ):
        line(cfg, row)
    footer(tagger, False)


