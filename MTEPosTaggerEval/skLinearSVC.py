# -*- coding: UTF-8 -*-
"""This File contains the Implementation for the Evaluation of the LiniearSVC classifier (support vector machine).
"""
from time import time

from sklearn.feature_extraction import DictVectorizer

from sklearn.svm import LinearSVC

from MTEPosTaggerEval.AbstractSKLearnPoSTaggerImpl import AbstractSKLearnPoSTaggerImpl
from MTEPosTaggers.MTESKTagger import MTESKTagger

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class skLinearSVC(AbstractSKLearnPoSTaggerImpl):
    """
    This class implements the evaluation for the LiniearSVC classifier (support vector machine).
    """

    def evaluate(self, sents_train, sents_test, config_option):
        t = time()
        tagger = MTESKTagger(tagged_sents=sents_train, classifier=LinearSVC(), vectorizer=DictVectorizer(),
                             context_window_generator=config_option)
        self.training_time = time() - t

        t = time()
        self.results = tagger.metrics(sents_test, printout=False)
        self.prediction_time = time() - t

        return self
