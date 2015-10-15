#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""This file represents the Evaluationrunner itself. For configuration please scroll down for the appropriate section.
"""
import csv
from concurrent.futures import ThreadPoolExecutor

import collections
import numpy as np

from mte import MTECorpusReader
from mte import MTETagConverter

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"

if __name__ == "__main__":
    # configuration of the evaluation

    # file to store the results at
    result_file = 'evaluation.csv'

    # corpora to evaluate
    languages = ['oana-en.xml', 'oana-fa.xml', 'oana-ro.xml', 'oana-sl.xml', 'oana-cs.xml', 'oana-et.xml',
                 'oana-hu.xml', 'oana-pl.xml', 'oana-sk.xml', 'oana-sr.xml']

    # path to the multex-east corpus:
    corpus_root = ''

    # tagsets to evaluate
    tagsets = ['mte', 'universal']

    # postaggers to evaluate
    taggers = ['skMultinomialNB', 'skLinearSVC', 'skPerceptron', 'nltkBrill', 'tmpOOV']

    # metrics to evaluate
    metrics = ['accuracy', 'recall', 'precision', 'f1', 'oov', 'training_time', 'prediction_time']

    # number of folds
    n_folds = 10

    # number of workers (for thread pool executor)
    n_workers = 5

    executor = ThreadPoolExecutor(max_workers=n_workers)

    result_writer = open(result_file, 'w')
    fieldnames = ['tagger', 'corpus', 'tagset', 'config options', 'metric', 'min', 'max', 'mean', 'std']
    writer = csv.DictWriter(result_writer, fieldnames=fieldnames)
    writer.writeheader()

    for language in languages:
        loaded_corpus = MTECorpusReader(root=corpus_root, fileids=language)
        tagged_sents = loaded_corpus.tagged_sents()

        for tagset in tagsets:
            if tagset == "universal":
                working_sents = [[(word, MTETagConverter.msd_to_universal(tag)) for (word, tag) in sent] for sent in
                                 tagged_sents]

            elif tagset == 'mte':
                working_sents = tagged_sents
            else:
                raise Exception("unknown tagset %s" % (tagset,))

            for tagger in taggers:
                tagger_impl = getattr(__import__(tagger), tagger)()
                for config_coption in tagger_impl.config_options:

                    seq = range(0, n_folds)
                    assert isinstance(seq, collections.Iterable)

                    w_corpus = list(working_sents)

                    # devide the corpus in n similar sized slices
                    chunks = [[] for i in seq]
                    while len(w_corpus) != 0:
                        for i in seq:
                            if len(w_corpus) == 0:
                                break
                            else:
                                chunks[i].append(w_corpus.pop())

                    runs = []
                    for i in seq:
                        train = []
                        for j in seq:
                            if j != i:
                                train = train + chunks[j]

                        test = chunks[i]

                        run_impl = getattr(__import__(tagger), tagger)()
                        runs.append(executor.submit(run_impl.evaluate, train, test, config_coption))

                    for metric in metrics:
                        results = [run.result().get_result(metric) for run in runs]

                        results = np.array(results)

                        print("%s evaluated %s in %s: min: %s | max: %s | mean: %s | std:%s" % (tagger, metric,
                                                                                                language,
                                                                                                results.min(),
                                                                                                results.max(),
                                                                                                results.mean(),
                                                                                                results.std()))

                        writer.writerow({'tagger': tagger, 'corpus': language, 'tagset': tagset,
                                         'config options': config_coption, 'metric': metric,
                                         'min': results.min(), 'max': results.max(),
                                         'mean': results.mean(), 'std': results.std()
                                         })

                        result_writer.flush()
