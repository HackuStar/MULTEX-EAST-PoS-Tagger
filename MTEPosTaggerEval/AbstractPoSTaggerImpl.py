# -*- coding: UTF-8 -*-
"""This File contains the Abstract Implementation for the Evaluation of Part-of-Speech-Taggers
"""
from abc import abstractmethod

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class AbstractPoSTaggerImpl(object):
    """
    This abstract class defines the methods an implementation of a part of speech tagger must provide
    to be runned within the evaluation framework.
    """
    config_options = []
    training_time = 0
    prediction_time = 0

    @abstractmethod
    def evaluate(self, sents_train, sents_test, config_option):
        """
        This method evaluates the tagger for a given configuration option and set of sents. After evaluation it stores
         the result within the instance, so they can be obtained via the get_result-method.

        :param sents_train: Tagged sentences to train the tagger.
        :type sents_train: [[(word:str, tag:str)]]
        :param sents_test:
        :type sents_test: [[(word:str, tag:str)]]
        :param config_option: Configuration to pass to the tagger
        :type config_option: Object
        :return:
        """
        pass

    @abstractmethod
    def get_result(self, metric):
        """
        this method returns the result for a given metric, which is usually identified by a string.

        :param metric: Metric to return
        :type metric: string
        :return: Result
        :rtype: float
        """
        pass
