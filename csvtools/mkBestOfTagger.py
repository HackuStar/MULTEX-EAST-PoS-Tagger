#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX Table with all Information from the Results-File.
"""
from __future__ import print_function

from collections import defaultdict
import csv

from config import tagsets, mapping, map_taggers, taggers, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"

oov = defaultdict()


def header(tagger):
    print
    """
   \\subsubsection{%s}
   \\begin{table}
   \\resizebox{\\textwidth}{!}{
       \\begin{tabular}{l|l|l|c|c|d{0.4}|d{0.4}|c}
           \\textbf{Language} & \\textbf{Configuration} & \\textbf{Tagset} & \multicolumn{2}{c|}{\\textbf{Time (seconds)}} & \multicolumn{1}{c|}{\\textbf{Accuracy}} &  \multicolumn{1}{c|}{\\textbf{F1-Score}} &  \\textbf{Out of}\\\\
           & &  & \\textbf{Training} & \\textbf{Tagging} & \multicolumn{2}{c|}{(second line: standard deviation)} &  \\textbf{Vocabulary}\\\\
               """ % map_taggers[tagger]


def footer(lang_code, tagset):
    print
    """   \end{tabular}}
       \caption{Best Results per Language for the %s Tagger}
       \label{table:taggersBestOfTagger_%s}
   \end{table}
   \clearpage""" % (map_taggers[lang_code], lang_code)


def mkFirstLine(metric, std, row):
    print(
        "        \multirow{{4}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} &  \multirow{{2}}{{*}}{{{:3.1f}}} & \multirow{{2}}{{*}}{{{:2.2f}}} & {:0.4f} & {:0.4f} & \multirow{{4}}{{*}}{{{:0.4f}}}\\\\".format(
            mapping[row['corpus']],
            row['config options'],
            tagsets[metric['tagset'].upper()],
            metric['training_time'],
            metric['prediction_time'],
            metric['accuracy'],
            metric['f1'],
            oov[row['corpus']]))

    print("        &  &   && & {:0.4f}  & {:0.4f} &\\\\".format(std['accuracy'], std['f1']))


def mkLine(metric, std, row):
    print(
        "         & \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} &  \multirow{{2}}{{*}}{{{:3.1f}}} & \multirow{{2}}{{*}}{{{:2.2f}}} & {:0.4f} & {:0.4f} & \\\\".format(
            row['config options'], tagsets[metric['tagset'].upper()], metric['training_time'],
            metric['prediction_time'],
            metric['accuracy'], metric['f1']))

    print("""        &  &   && & {:0.4f}  & {:0.4f} &\\\\""".format(std['accuracy'], std['f1']))


metric = dict()
std = dict()
last = ""
last_tag = "mte"
last_corpus = ""
bestof_univ = defaultdict(list)
bestof_msd = defaultdict(list)

with open(RESULTS_CSV, 'rb') as csvfile:
    result = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in result:
        last_corpus = row['corpus']

        if row['tagger'] == 'tmpOOV':
            if row['metric'] == 'oov':
                oov[row['corpus']] = float(row['mean']) / 100
            continue

        metric['corpus'] = row['corpus']
        metric['tagger'] = row['tagger']
        metric['tagset'] = row['tagset']

        metric['config options'] = row['config options']

        metric[row['metric']] = float(row['mean'])
        std[row['metric']] = float(row['std'])

        if all(test in metric for test in
               ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):

            if row['tagset'] == 'universal':
                cur = filter(lambda (a, b): row['corpus'] == a['corpus'], bestof_univ[row["tagger"]])
                if not cur:
                    bestof_univ[row['tagger']].append((dict(metric), dict(std)))
                else:
                    fstSize = len(bestof_univ[row['tagger']])
                    bestof_univ[row['tagger']] = filter(lambda (a, b): (not row['corpus'] == a['corpus']) or
                                                                       (row['corpus'] == a['corpus']
                                                                        and metric['f1'] < a['f1']),
                                                        bestof_univ[row['tagger']])
                    if fstSize > len(bestof_univ[row['tagger']]):
                        bestof_univ[row['tagger']].append((dict(metric), dict(std)))
            else:
                cur = filter(lambda (a, b): row['corpus'] == a['corpus'], bestof_msd[row["tagger"]])
                if not cur:
                    bestof_msd[row['tagger']].append((dict(metric), dict(std)))
                else:
                    fstSize = len(bestof_msd[row['tagger']])
                    bestof_msd[row['tagger']] = filter(lambda (a, b): (not row['corpus'] == a['corpus']) or
                                                                      (row['corpus'] == a['corpus']
                                                                       and metric['f1'] < a['f1']),
                                                       bestof_msd[row['tagger']])
                    if fstSize > len(bestof_msd[row['tagger']]):
                        bestof_msd[row['tagger']].append((dict(metric), dict(std)))

            last = row['tagger']
            last_tag = row['tagset']

            metric = dict()
            std = dict()

for tagger in taggers:
    header(tagger)
    bestof_univ[tagger].sort(key=lambda (a, b): a['corpus'])
    bestof_msd[tagger].sort(key=lambda (a, b): a['corpus'])

    for metricuniv, metricmsd in zip(bestof_univ[tagger], bestof_msd[tagger]):  #
        print
        "        \\hline"
        mkFirstLine(metricuniv[0], metricuniv[1], metricuniv[0])
        mkLine(metricmsd[0], metricmsd[1], metricmsd[0])

    footer(tagger, '')
