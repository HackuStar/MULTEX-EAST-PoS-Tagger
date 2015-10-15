#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX Table with all Information from the Results-File.
"""
from __future__ import print_function
from collections import defaultdict
import csv

from config import tagsets, mapping, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"

oov = defaultdict()


def header():
    header = """

    \\begin{table}
    \\resizebox{\\textwidth}{!}{
        \\begin{tabular}{l|l|l|d{0.4}|d{0.4}|c}
            \\textbf{Tagger} & \\textbf{Configuration} & \\textbf{Language}  & \multicolumn{1}{c|}{\\textbf{Accuracy}} & \multicolumn{1}{c|}{\\textbf{F1-Score}} & \\textbf{Out of}\\\\
            & &  & \multicolumn{2}{c|}{(second line: standard deviation)} & \\textbf{Vocabulary}\\\\
            """

    print(header)


def footer(lang_code, tagset):
    footer = """
    \end{tabular}}
    \caption{Best Results Overview with the %s Tagset}
    \label{table:taggersBestOf_%s}
    \end{table}""" % (tagsets[tagset.upper()], tagset)
    print(footer)


def mkLine(metric, std, row):
    print("""\\hline
                \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} &  {:0.4f} & {:0.4f} & \multirow{{2}}{{*}}{{{:0.4f}}} \\\\""".format(
        row['tagger'], row['config options'], mapping[row['corpus']], metric['accuracy'], metric['f1'],
        oov[row['corpus']]))

    print("""   && & {:0.4f} & {:0.4f} &\\\\""".format(std['accuracy'], std['f1']))


metric = dict()
std = dict()
last = ""
last_tag = "mte"
last_corpus = ""
bestof = {"mte": [], "universal": []}

with open(RESULTS_CSV, 'rb') as csvfile:
    result = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in result:
        last_corpus = row['corpus']

        if row['tagger'] == 'tmpOOV' and row['metric'] == 'oov':
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

            cur = filter(lambda (a, b): row['corpus'] == a['corpus'], bestof[row["tagset"]])
            if not cur:
                bestof[row['tagset']].append((dict(metric), dict(std)))
            else:
                fstSize = len(bestof[row['tagset']])
                bestof[row['tagset']] = filter(lambda (a, b): (not row['corpus'] == a['corpus']) or
                                                              (row['corpus'] == a['corpus'] and metric['f1'] < a['f1']),
                                               bestof[row['tagset']])
                if fstSize > len(bestof[row['tagset']]):
                    bestof[row['tagset']].append((dict(metric), dict(std)))

            last = row['tagger']
            last_tag = row['tagset']

            metric = dict()
            std = dict()

for tagset in bestof:
    header()
    bestof[tagset].sort(key=lambda (a, b): a['f1'], reverse=True)

    for metric, std in bestof[tagset]:
        mkLine(metric, std, metric)
    footer(last_corpus, tagset)
