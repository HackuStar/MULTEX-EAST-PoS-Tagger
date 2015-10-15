# -*- coding: UTF-8 -*-
"""This File contains the Implementation for the Evaluation of  the NLTK-Brill Classifier.
"""
from time import time

from MTEPosTaggerEval.AbstractPoSTaggerImpl import AbstractPoSTaggerImpl
from MTEPosTaggers.MTEBrillTagger import MTEBrillTagger

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class nltkBrill(AbstractPoSTaggerImpl):
    config_options = ['baseline',
                      'fntbl37',
                      'brill24',
                      'nltkdemo18',
                      'nltkdemo18plus']
    max_rules = 250
    min_score = 2
    min_acc = None
    results = None

    def evaluate(self, sents_train, sents_test, config_option):
        t = time()
        brill = MTEBrillTagger(sents_train, max_rules=self.max_rules, min_score=self.min_score, min_acc=self.min_acc,
                               template=config_option)
        self.training_time = time() - t

        t = time()
        self.results = brill.metrics(sents_test, printout=False)
        self.prediction_time = time() - t

        return self

    def get_result(self, metric):
        if metric == 'accuracy':
            return self.results[0]
        elif metric == 'precision':
            return self.results[1]
        elif metric == 'recall':
            return self.results[2]
        elif metric == 'f1':
            return self.results[3]
        elif metric == 'training_time':
            return self.training_time
        elif metric == 'prediction_time':
            return self.prediction_time
        elif metric == 'oov':
            return -1
