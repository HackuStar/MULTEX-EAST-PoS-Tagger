# -*- coding: UTF-8 -*-
"""This File contains the Implementation for the Evaluation of the Scikit-Learn Perceptron Classifier.
"""
from time import time

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import Perceptron

from MTEPosTaggerEval.AbstractSKLearnPoSTaggerImpl import AbstractSKLearnPoSTaggerImpl
from MTEPosTaggers.MTESKTagger import MTESKTagger

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class skPerceptron(AbstractSKLearnPoSTaggerImpl):
    """
    This class implements the evaluation for the perceptron classifier.
    """

    def evaluate(self, sents_train, sents_test, config_option):
        t = time()
        tagger = MTESKTagger(tagged_sents=sents_train, classifier=Perceptron(n_iter=50), vectorizer=DictVectorizer(),
                             context_window_generator=config_option)
        self.training_time = time() - t

        t = time()
        self.results = tagger.metrics(sents_test, printout=False)
        self.prediction_time = time() - t

        return self
