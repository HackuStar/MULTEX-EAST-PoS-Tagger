# -*- coding: UTF-8 -*-
"""This File contains the Implementation for the Evaluation of the TextMiningProject (TMP) OutOfVocabulary (tmpOOV) helper
"""
from time import time

from collections import defaultdict
from functools import reduce

from MTEPosTaggerEval.AbstractPoSTaggerImpl import AbstractPoSTaggerImpl

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class tmpOOV(AbstractPoSTaggerImpl):
    """
    this is a helper implementation to calculate the out of vocabulary words only once.
    """

    config_options = ['mkOOV']
    oov = 0

    def evaluate(self, sents_train, sents_test, config_option):
        t = time()
        d = defaultdict(lambda: False)
        for sent in sents_train:
            for (word, tag) in sent:
                d[word] = True

        tokens = sum(sents_test, [])
        oov = reduce(lambda a, b: a + 1 if not b in d else a, [w for (w, _) in tokens], 0)
        self.oov = (oov * 100.0) / len(tokens)

        self.training_time = time() - t

        return self

    def get_result(self, metric):
        if metric == 'accuracy':
            return -1
        elif metric == 'precision':
            return -1
        elif metric == 'recall':
            return -1
        elif metric == 'f1':
            return -1
        elif metric == 'training_time':
            return self.training_time
        elif metric == 'prediction_time':
            return -1
        elif metric == 'oov':
            return self.oov
