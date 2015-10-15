#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX / TIKZ Plots with PGFPlots, displaying the Performance of all Taggers an Configurations.
"""
from __future__ import print_function

from collections import defaultdict
import csv

import numpy as np

from config import mapping, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"

oov = defaultdict()


def footer(lang_code, tagset):
    footer = """\\end{axis}\\end{tikzpicture}"""
    print(footer)


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

mean_per_tagger_lang = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list())))

for tagger_data in tagger:
    for language in tagger[tagger_data]:
        mean = defaultdict(lambda: list())
        for copt, f1, tagset in tagger[tagger_data][language]:
            if copt == "baseline":
                continue
            mean_per_tagger_lang[tagger_data][tagset][language] += [f1]


def header(labels, tagger):
    header = """
\\begin{tikzpicture}
\\begin{axis}[
title={%s},
ylabel=F1-Score,
x tick label style={rotate=45,anchor=east},
xtick={%s},
legend style={draw=white!15!black,legend cell align=left},
symbolic x  coords={%s},
]""" % (tagger, ",".join(labels), ",".join(labels))

    return (header)


def mkLine(lang, f1, mark):
    return ("""\\addplot[mark={}] coordinates {{({},{:0.4f})}};\n""".format(
        mark, mapping[lang], f1))


print("""
\\begin{table}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{cc}""")

i = 1

for tagger in mean_per_tagger_lang:
    lines = ""
    out = ""
    labels = list()
    tagsetcords = defaultdict(lambda: list())

    for tagset in mean_per_tagger_lang[tagger]:
        for language in mean_per_tagger_lang[tagger][tagset]:
            out += mkLine(language, np.array(mean_per_tagger_lang[tagger][tagset][language]).mean(),
                          "x" if tagset == "universal" else "+")

            tagsetcords[tagset] += ["""({},{:0.4f})""".format(mapping[language], np.array(
                mean_per_tagger_lang[tagger][tagset][language]).mean())]
            if mapping[language] not in labels:
                labels.append(mapping[language])

    print(header(labels, tagger))
    print(out)
    for tagset in tagsetcords:
        print(
            """\\addplot[%s] coordinates {%s};""" % (
                "red" if tagset == "universal" else "blue", "".join(tagsetcords[tagset])))
    footer("", "")

    if not i % 2 == 0:
        print("\n&\n")

    else:

        print("\\\\")

    i += 1

print("""\\multicolumn{2}{c}{""")

print
"""\\begin{tikzpicture}
\\begin{axis}[%
hide axis,
xmin=10,
xmax=50,
ymin=0,
ymax=0.4,
legend style={draw=white!15!black,legend cell align=left},
legend columns=1,]
\\addlegendimage{red}\\addlegendentry{Universal Tagset};
\\addlegendimage{blue}\\addlegendentry{MSD Tagset};
\\end{axis}
\\end{tikzpicture}"""

print("""
        }
        \\end{tabular}
        }

        \\caption{Comparision of performance between MSD and Universal Tagset}
        \\label{graphic:comp_tagset_performance}
        \\end{table}""")
