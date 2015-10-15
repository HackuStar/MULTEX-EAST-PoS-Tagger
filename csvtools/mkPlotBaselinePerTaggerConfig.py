#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This Script generates LaTeX / TIKZ Plots with PGFPlots, comparing each Tagger to its Baseline.
"""
from __future__ import print_function

from collections import defaultdict
import csv
import itertools
import collections

from config import tagsets, copts2str, RESULTS_CSV

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"



oov = defaultdict()
taggerconf = defaultdict(list)

metric = dict()
std = dict()
last = ""
last_tag = "mte"
last_corpus = ""
tagger = defaultdict(lambda: defaultdict(list))
baselines = defaultdict(lambda: defaultdict(list))

with open(RESULTS_CSV, 'rb') as csvfile:
    result = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in result:

        if row['tagger'] == 'tmpOOV':
            continue

        metric['corpus'] = row['corpus']
        metric['tagger'] = row['tagger']
        metric['tagset'] = row['tagset']

        metric['config_option'] = copts2str[row['config options']]

        if metric['config_option'] == "[0]":
            metric['config_option'] = "baseline"

        metric[row['metric']] = float(row['mean'])
        std[row['metric']] = float(row['std'])

        if all(test in metric for test in
               ["accuracy", "recall", "precision", "f1", "training_time", "prediction_time"]):

            if tagger[row['tagger']][metric['config_option']] is None:
                tagger[row['tagger']][metric['config_option']] = []
                baselines[row['tagger']][metric['config_option']] = []

            if metric['config_option'] == "base":
                baselines[row['tagger']][metric['config_option']].append((row['corpus'], metric['f1'], row['tagset']))
            else:
                tagger[row['tagger']][metric['config_option']].append((row['corpus'], metric['f1'], row['tagset']))
                if metric['config_option'] not in taggerconf[metric['tagger']]:
                    taggerconf[metric['tagger']].append(metric['config_option'])

            last = row['tagger']
            last_tag = row['tagset']
            last_corpus = row['corpus']

            metric = dict()
            std = dict()

tagger = collections.OrderedDict(sorted(tagger.items()))
taggerconf = collections.OrderedDict(sorted(taggerconf.items()))

for curTagger in tagger:
    tagger[curTagger] = collections.OrderedDict(sorted(tagger[curTagger].items()))

for curTaggerConf in taggerconf:
    taggerconf[curTaggerConf].sort()

def header(labels, tagger):
    return """
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
]\n""" % (tagger, labels)


lang = set()
plots = defaultdict(lambda: defaultdict(list))
for tagset in ['universal', 'mte']:
    for curTagger in ['nltkBrill', 'skLinearSVC', 'skMultinomialNB', 'skPerceptron']:
        for option in taggerconf[curTagger]:
            plotpoints_better = """\\addplot+[color=green, forget plot,only marks, error bars/.cd,y dir=plus,y explicit] coordinates { \n"""
            plotpoints_worse = """\\addplot+[color=red, forget plot,only marks, error bars/.cd,y dir=minus,y explicit] coordinates { \n"""

            for workingOnTagger, workingOnBaseline in itertools.izip(tagger[curTagger][option],
                                                                     baselines[curTagger]['base']):
                tagger_lang, tagger_f1, tagger_tagset = workingOnTagger
                tagger_lang = tagger_lang.replace("oana-", "").replace(".xml", "")

                baseline_lang, baseline_f1, baseline_tagset = workingOnBaseline
                baseline_lang = baseline_lang.replace("oana-", "").replace(".xml", "")

                assert baseline_lang == tagger_lang or baseline_tagset == tagger_tagset, \
                    "csv order is crucial. Baseline must within or after tagger output"

                if not tagger_tagset == tagset:
                    continue

                lang.add(baseline_lang)

                if baseline_f1 > tagger_f1:
                    plotpoints_worse += """({},{:0.4f}) +- (0,{:0.4f})\n""".format(tagger_lang, baseline_f1, abs(baseline_f1 - tagger_f1))
                else:
                    plotpoints_better += """({},{:0.4f}) +- (0,{:0.4f})\n""".format(tagger_lang, baseline_f1, abs(tagger_f1-baseline_f1))

            plotpoints_worse += """};\n"""
            plotpoints_better += """};\n"""

            plot = header(",".join(lang), option)
            plot += plotpoints_better
            plot += plotpoints_worse
            plot += """\\legend{}\\end{axis}\\end{tikzpicture}"""

            plots[tagset][curTagger].append(plot)

legend = """
\\begin{tikzpicture}
\\begin{axis}[%
hide axis,
xmin=10,
xmax=50,
ymin=0,
ymax=0.4,
legend style={draw=white!15!black,legend cell align=left},
legend columns=1,]
\\addlegendimage{color=green, mark=-}\\addlegendentry{better than baseline};
\\addlegendimage{color=blue, mark=-}\\addlegendentry{baseline};
\\addlegendimage{color=red, mark=-}\\addlegendentry{worse than baseline};
\\end{axis}
\\end{tikzpicture}
"""

for tagset in plots:
    for tagger in plots[tagset]:
        if len(plots[tagset][tagger]) % 3 == 0:
            print("""
\\begin{table}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{ccc}""")
            curPlots = iter(plots[tagset][tagger])
            for plot in curPlots:
                print("%s\n&\n%s&\n%s\n\\\\" % (plot, next(curPlots), next(curPlots)))

            print("""
\\multicolumn{3}{c}{
%s
""" % legend)
        elif len(plots[tagset][tagger]) % 2 == 0:
            print("""
\\begin{table}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lr}""")

            curPlots = iter(plots[tagset][tagger])
            for plot in curPlots:
                print ("%s\n&\n%s\n\\\\" % (plot,next(curPlots)))

            print("""
\\multicolumn{2}{c}{
%s
""" % legend)

        print("""
        }
        \\end{tabular}
        }

        \\caption{Comparison over all Languages and Configuration Options for the %s Tagger within the %s Tagset}
        \\label{graphic:all_lang_copts_%s_%s}
        \\end{table}""" % (tagger, tagsets[tagset.upper()], tagger, tagset))
