#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX Table with all Information from the Results-File.
"""
from __future__ import print_function

import csv

from config import tagsets, mapping, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


def header():
    header = """
    \\begin{table}
    \\resizebox{.9\\textwidth}{!}{
        \\begin{tabular}{l|l|d{3.1}|d{2.2}|d{0.4}|d{0.4}|d{0.4}|d{0.4}}
            \\textbf{Tagger} & \\textbf{Configuration} & \multicolumn{2}{c|}{\\textbf{Time (seconds)}} & \multicolumn{1}{c|}{\\textbf{Accuracy}} & \multicolumn{1}{c|}{\\textbf{Recall}} & \multicolumn{1}{c|}{\\textbf{Precision}} & \multicolumn{1}{c}{\\textbf{F1-Score}} \\\\
            &   & \multicolumn{1}{c|}{\\textbf{Training}} & \multicolumn{1}{c|}{\\textbf{Tagging}} & \multicolumn{4}{c}{(second line: standard deviation)} \\\\
            """

    print(header)


def footer(lang_code, tagset):
    footer = """
    \end{tabular}}
    \scriptsize
    \caption{Results of the different Part of Speech-Taggers on the Language %s for the %s Tagset}
    \label{table:lang%s%s}
    \end{table}\clearpage""" % (mapping[lang_code], tagsets[tagset.upper()], lang_code.upper(), tagset)
    print(footer)


def mkLine(metric, std, row):
    print("""\\hline
                {} & {} & {:3.1f} & {:2.2f} & {:0.4f} & {:0.4f} & {:0.4f} & {:0.4f} \\\\""".format(
        row['tagger'], row['config options'], metric['training_time'], metric['prediction_time'],
        metric['accuracy'], metric['recall'], metric['precision'], metric['f1']))

    print("""&  & &   & {:0.4f} & {:0.4f} & {:0.4f} & {:0.4f} \\\\""".format(
        std['accuracy'], std['recall'], std['precision'], std['f1']))


metric = dict()
std = dict()
last = ""
last_tag = "mte"
last_corpus = "oana-en.xml"

header()
with open(RESULTS_CSV, 'rb') as csvfile:
    result = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in result:

        if row['tagger'] == 'tmpOOV':
            continue

        metric['corpus'] = row['corpus']
        metric['tagger'] = row['tagger']
        metric['tagset'] = row['tagset']

        metric['config_option'] = row['config options']

        metric[row['metric']] = float(row['mean'])
        std[row['metric']] = float(row['std'])

        if all(test in metric for test in
               ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):

            if not last == row['tagger']:
                print
                "\\hline"

            if not last_tag == row['tagset'] or \
                    not last_corpus == row['corpus']:
                footer(last_corpus, last_tag)
                header()

            last = row['tagger']
            last_tag = row['tagset']
            last_corpus = row['corpus']

            mkLine(metric, std, row)

            metric = dict()
            std = dict()

footer(last_corpus, last_tag)
