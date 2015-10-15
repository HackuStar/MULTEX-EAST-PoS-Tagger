# -*- coding: UTF-8 -*-
"""This File contains the Abstract Implementation for the Evaluation of Scikit-Learn based classifiers
"""
from MTEPosTaggerEval.AbstractPoSTaggerImpl import AbstractPoSTaggerImpl
from MTEPosTaggerEval.SKLearnVectorGenerators import *

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class AbstractSKLearnPoSTaggerImpl(AbstractPoSTaggerImpl):
    """
    Abstract class which is implemented for all scikit-learn based taggers. It will contained commonly used methods.

    """
    results = None

    def evaluate(self, sents_train, sents_test, config_option):
        pass

    config_options = [baseline(), suf_baseline(),
                      around3(),  suf_around3(),
                      around2(),  suf_around2(),
                      around1(),  suf_around1(),
                      left3(),    suf_left3(),
                      left2(),    suf_left2(),
                      left1(),    suf_left1(),
                      right3(),   suf_right3(),
                      right2(),   suf_right2(),
                      right1(),   suf_right1()]

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
