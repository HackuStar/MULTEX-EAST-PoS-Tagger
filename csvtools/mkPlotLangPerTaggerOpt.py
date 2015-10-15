#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX / TIKZ Plots with PGFPlots, displaying the Performance of all Taggers an Configurations.
"""
from __future__ import print_function

from collections import defaultdict
import csv

from config import tagsets, copts2str, mapping, colors, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"

oov = defaultdict()


def header(labels, tagger):
    header = """
\\begin{tikzpicture}
\\begin{axis}[
title={%s},
ylabel=F1-Score,
x tick label style={rotate=45,anchor=east},
xtick=data,
legend style={
at={(0,0)},
anchor=north east},
symbolic x  coords={%s},
]""" % (tagger, labels)

    print(header)


def footer(lang_code, tagset):
    footer = """\\legend{}\\end{axis}\\end{tikzpicture}"""
    print(footer)


def mkLine(metric, std, row):
    print("""\\addplot coordinates {{({},{:0.4f})}};""".format(
        metric['config_option'], metric['f1']))


metric = dict()
std = dict()
last = ""
last_tag = "mte"
last_corpus = ""
tagger = defaultdict(lambda: defaultdict(list))

with open(RESULTS_CSV, 'rb') as csvfile:
    result = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in result:

        if row['tagger'] == 'tmpOOV':
            continue

        metric['corpus'] = row['corpus']
        metric['tagger'] = row['tagger']
        metric['tagset'] = row['tagset']

        metric['config_option'] = row['config options']

        if metric['config_option'] == "[0]":
            metric['config_option'] = "baseline"

        metric[row['metric']] = float(row['mean'])
        std[row['metric']] = float(row['std'])

        if all(test in metric for test in
               ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):

            if tagger[row['tagger']][row['corpus']] is None:
                tagger[row['tagger']][row['corpus']] = []

            tagger[row['tagger']][row['corpus']].append((metric['config_option'], metric['f1'], row['tagset']))

            last = row['tagger']
            last_tag = row['tagset']
            last_corpus = row['corpus']

            metric = dict()
            std = dict()

for workset in ['universal', 'mte']:
    print (
    """
       \\begin{table}
       \\resizebox{\\textwidth}{!}{
       \\begin{tabular}{rl}""")

    plotnr = 0
    for taggerlist in ['nltkBrill', 'skLinearSVC', 'skMultinomialNB', 'skPerceptron']:
        labels = list()
        lastset = ""
        first = True
        tblcontent = ""

        lastlang = ""
        for language in tagger[taggerlist]:
            for copt, f1, tagset in tagger[taggerlist][language]:
                if copt == "baseline":
                    continue

                copt = copts2str[copt]

                if not tagset == workset:
                    continue

                if not lastlang == language:
                    tblcontent += "\\addplot[color=%s, mark=+] coordinates {" % colors[language]

                lastlang = language

                tblcontent += "({},{:0.4f})".format(copt, f1)

                if copt not in labels:
                    labels.append(copt)

            tblcontent += "};\\addlegendentry{%s}\n" % mapping[language]

        header(",".join(labels), taggerlist)
        print(        tblcontent)
        footer(taggerlist, workset)

        plotnr += 1

        if plotnr < 2:
            print (
            "&")
        else:
            print(
            "\\\\")
            plotnr = 0

    print (
    """
       \\multicolumn{2}{c}{
       \\begin{tikzpicture}
       \\begin{axis}[%
       hide axis,
       xmin=10,
       xmax=50,
       ymin=0,
       ymax=0.4,
       legend style={draw=white!15!black,legend cell align=left},
       legend columns=5,
       ]""")
    for langflag in colors:
        if langflag == "baseline":
            continue
        print(
        """\\addlegendimage{color=%s, mark=+}\\addlegendentry{%s};""" % (colors[langflag], mapping[langflag]))

    print(
    """
       \\end{axis}
       \\end{tikzpicture}
           }""")
    print(
    """
       \end{tabular}}

       \caption{Chart of Results for the %s Tagset over all Taggers and Configurations}
       \label{graphic:comparisonTagger_Language_%s}
       \end{table}
       """ % (tagsets[workset.upper()], workset))
