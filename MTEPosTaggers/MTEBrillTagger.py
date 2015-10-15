# -*- coding: UTF-8 -*-
"""This File Contains the MULTEX-East enabled NLTK-Brill Tagger.
"""

from nltk.tag.brill import *

from nltk.tag.util import untag
from nltk.metrics import *

from nltk.tag import BrillTaggerTrainer, UnigramTagger
from nltk.tbl.template import Template

try:
    from functools import reduce
except ImportError:
    "No python 3"

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class MTEBrillTagger:
    '''
    This is a BrillTagger for text annotated using the MTE tag set.
    It should not be used for other tag sets, as it works with MTE tags internally.
    '''

    def __init__(self, tagged_sents, anonProperNouns=False, initialTagger=None, max_rules=250, min_score=2,
                 min_acc=None, template='fntbl37'):
        '''
        Construct a new MTEBrillTagger and train it with the sentences from tagged_sents.

        :param tagged_sents: Tagged sentences to train the tagger.
        :type tagged_sents: [[(word:str, tag:str)]]
        :param anonProperNouns: Set 'True' to replace every proper noun with an anonymous string. Currently only for MTE tags.
        :type anonProperNouns: bool
        :param initialTagger: If None or unset, use UnigramTagger as initial tagger; use specified one else ('self._tagger = initialTagger')
        :type initialTagger: Tagger
        :param max_rules: tagger generates at most max_rules rules
        :type max_rules: int
        :param min_score: stop training when no rules better than min_score can be found
        :type min_score: int
        :param min_acc: discard any rule with lower accuracy than min_acc
        :type min_acc: float or None
        :param template: template set to use to train Brill Tagger. Can be the name of a function from nltk.tag.brill that returns a template set or a list of templates.
        :type template: str or list
 
        '''

        self._tagged_sents = []
        ANON = "anon"

        if anonProperNouns:
            for s in tagged_sents:
                tmp = []
                for (w, tag) in s:
                    if tag.startswith("#Np"):
                        tmp.append((ANON, "#Np"))
                    else:
                        tmp.append((w, tag))
                self._tagged_sents.append(tmp)
        else:
            self._tagged_sents = tagged_sents

        Template._cleartemplates()

        # If 'template' parameter is 'None' default to fntbl37 template set
        if template is None:
            templates = fntbl37()

        # Check if 'template' parameter is a list. If it is try to use it directly
        elif type(template) is list:
            templates = template

        # Check if 'template' is a string. If it is try to get the template set from nltk
        elif type(template) is str:
            if template == "fntbl37":
                templates = fntbl37()
            elif template == "brill24":
                templates = brill24()
            elif template == "nltkdemo18":
                templates = nltkdemo18()
            elif template == "nltkdemo18plus":
                templates = nltkdemo18plus()
            elif template == "baseline":
                templates = None
            else:
                raise ValueError("Method returning templates not found!")

        # If it is any other type, raise error
        else:
            raise ValueError(
                "Please specify the name of a function that returns a list of templates or a list of templates directly!")

        if initialTagger is None:
            self._tagger = UnigramTagger(self._tagged_sents)
        else:
            self._tagger = initialTagger

        if templates is not None:
            self._tagger = BrillTaggerTrainer(self._tagger, templates, trace=3)
            self._tagger = self._tagger.train(self._tagged_sents, max_rules=max_rules, min_score=min_score,
                                              min_acc=min_acc)

    def evaluate(self, test_sents):
        '''
        Gives the precision of the tagger with the given test sentences.
        *Use the metrics method for more output!*
        :param test_sents: The sentences to thest the tagger with
        :type test_sents: [[(str, str)]]
        '''
        return self._tagger.evaluate(test_sents)

    def metrics(self, gold, printout=True, confusion_matrix=False, oov=True):
        '''
        More sophisticated evalution method gives more numbers.

        :param gold: The sentences to use for testing
        :type gold: [[(str, str)]]
        :param printout: Should I print the results or just return them?
        :type printout: bool
        :param confusion_matrix: Should I create a Confusion Matrix?
        :type confusion_matrix: bool
        :param oov: Should the out of vocabulary words be calculated
        :type oov: bool
        :return: (acc, prec, rec, fsc, aov, None) the first five are the accuracy, precision, recall, fscore, and out of vocabulary words. The last one is the Confusion Matrix if requested, else None
        :rtype: (double, double, double, double, double, ConfusionMatrix or None)
        '''

        tagger_out = self._tagger.tag_sents(untag(sent) for sent in gold)
        gold_tokens = sum(gold, [])
        test_tokens = sum(tagger_out, [])
        gold_tokens_set = set(gold_tokens)
        test_tokens_set = set(test_tokens)

        gold_tags = [t for (_, t) in gold_tokens]
        test_tags = [t for (_, t) in test_tokens]

        # calculate out of vocabulary words
        if oov:
            d = {word: True for (word, _) in reduce(lambda a, b: a + b, self._tagged_sents, [])}
            aov = reduce(lambda a, b: a + 1 if not b in d else a, [w for (w, _) in gold_tokens], 0)
            aov = (aov * 100.0) / len(gold_tokens)
        else:
            aov = '-1'

        acc = accuracy(gold_tokens, test_tokens)
        prc = precision(gold_tokens_set, test_tokens_set)
        rec = recall(gold_tokens_set, test_tokens_set)
        fms = f_measure(gold_tokens_set, test_tokens_set)
        cfm = None

        if confusion_matrix:
            cfm = ConfusionMatrix(gold_tags, test_tags)

        if printout:
            print("accuracy:          " + str(acc))
            print("precision:         " + str(prc))
            print("recall:            " + str(rec))
            print("f-score:           " + str(fms))
            print("out of vocabulary: " + str(aov) + " %")
            if confusion_matrix:
                print(cfm)

        return acc, prc, rec, fms, aov, cfm
