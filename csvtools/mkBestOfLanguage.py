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


def header(lang_code):
    header = """
\\subsubsection{%s}
\\begin{table}
    \\resizebox{\\textwidth}{!}{
    \\begin{tabular}{l|l|l|c|c|d{0.4}|d{0.4}|c}
            \\textbf{Tagger} & \\textbf{Configuration} & \\textbf{Tagset} & \multicolumn{2}{c|}{\\textbf{Time (seconds)}} & \multicolumn{1}{c|}{\\textbf{Accuracy}} & \multicolumn{1}{c|}{\\textbf{F1-Score}} & \\textbf{Out of}\\\\
            & &  & \\textbf{Training} & \\textbf{Tagging} & \multicolumn{2}{c|}{(second line: standard deviation)} & \\textbf{Vocabulary}\\\\
            """ % mapping[lang_code]

    print(header)


def footer(lang_code, tagset):
    footer = """
    \end{tabular}}
    \caption{Best Results per Tagger for the %s Language}
    \label{table:taggersBestOfLng_%s}
\end{table}

\paragraph{Conclusion}
\clearpage""" % (mapping[lang_code], lang_code)
    print(footer)


def mkLine(metric, std, row):
    print("""            \\cmidrule{{1-7}}
            \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} &  \multirow{{2}}{{*}}{{{:3.1f}}} & \multirow{{2}}{{*}}{{{:2.2f}}} & {:0.4f} & {:0.4f} & \\\\""".format(
        row['tagger'], row['config options'], tagsets[metric['tagset'].upper()], metric['training_time'],
        metric['prediction_time'],
        metric['accuracy'], metric['f1']))

    print("""            &  &   && & {:0.4f} & {:0.4f} &\\\\""".format(
        std['accuracy'], std['f1']))


def mkFirstLine(metric, std, row):
    print("""            \\hline
            \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} & \multirow{{2}}{{*}}{{{}}} &  \multirow{{2}}{{*}}{{{:3.1f}}} & \multirow{{2}}{{*}}{{{:2.2f}}} & {:0.4f} & {:0.4f} & \multirow{{19}}{{*}}{{{:0.4f}}}\\\\""".format(
        row['tagger'], row['config options'], tagsets[metric['tagset'].upper()], metric['training_time'],
        metric['prediction_time'],
        metric['accuracy'], metric['f1'], oov[row['corpus']]))

    print("""            &  &   && & {:0.4f} & {:0.4f} &\\\\""".format(
        std['accuracy'], std['f1']))


metric = dict()
std = dict()
last = ""
last_tag = "mte"
last_corpus = ""
bestof_mte = defaultdict(list)
bestof_univ = defaultdict(list)

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

        if row['tagset'] == "mte":
            if all(test in metric for test in
                   ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):

                cur = filter(lambda (a, b): row['tagset'] == a['tagset'] and row['tagger'] == a['tagger'],
                             bestof_mte[row["corpus"]])
                if not cur:
                    bestof_mte[row['corpus']].append((dict(metric), dict(std)))
                else:
                    fstSize = len(bestof_mte[row['corpus']])
                    bestof_mte[row['corpus']] = filter(
                        lambda (a, b): (not row['tagset'] == a['tagset']) or (not row['tagger'] == a['tagger']) or
                                       (row['tagset'] == a['tagset'] and row['tagger'] == a['tagger']
                                        and metric['f1'] < a['f1']), bestof_mte[row['corpus']])
                    if fstSize > len(bestof_mte[row['corpus']]):
                        bestof_mte[row['corpus']].append((dict(metric), dict(std)))

                last = row['tagger']
                last_tag = row['tagset']

                metric = dict()
                std = dict()
        else:
            if all(test in metric for test in
                   ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):

                cur = filter(lambda (a, b): row['tagset'] == a['tagset'] and row['tagger'] == a['tagger'],
                             bestof_univ[row["corpus"]])
                if not cur:
                    bestof_univ[row['corpus']].append((dict(metric), dict(std)))
                else:
                    fstSize = len(bestof_univ[row['corpus']])
                    bestof_univ[row['corpus']] = filter(
                        lambda (a, b): (not row['tagset'] == a['tagset']) or (not row['tagger'] == a['tagger']) or
                                       (row['tagset'] == a['tagset'] and row['tagger'] == a['tagger']
                                        and metric['f1'] < a['f1']), bestof_univ[row['corpus']])
                    if fstSize > len(bestof_univ[row['corpus']]):
                        bestof_univ[row['corpus']].append((dict(metric), dict(std)))

                last = row['tagger']
                last_tag = row['tagset']

                metric = dict()
                std = dict()

for language in ['oana-cs.xml', 'oana-en.xml', 'oana-et.xml', 'oana-fa.xml', 'oana-hu.xml', 'oana-pl.xml',
                 'oana-ro.xml', 'oana-sr.xml', 'oana-sk.xml', 'oana-sl.xml']:
    header(language)
    bestof_mte[language].sort(key=lambda (a, b): a['f1'], reverse=True)
    bestof_univ[language].sort(key=lambda (a, b): a['f1'], reverse=True)
    isFirstLine = True

    for metric, std in bestof_univ[language]:
        if isFirstLine:
            mkFirstLine(metric, std, metric)
            isFirstLine = False
        else:
            mkLine(metric, std, metric)
    print("            \\cmidrule{1-7}\morecmidrules\cmidrule{1-7}")
    for metric, std in bestof_mte[language]:
        mkLine(metric, std, metric)

    footer(language, '')
