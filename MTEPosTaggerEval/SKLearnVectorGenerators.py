# -*- coding: UTF-8 -*-
"""This File contains the different Vector Generators for the Scikit-Learn based classifiers.

 There are two generators, the PositionContextWindowVectorGenerator which takes the whole word and the
 SuffixContextWindowVectorGenerator takes the last two letters of each word. Both are position based they are
 getting the position of the words, by the context_window_positions variables.
"""
from MTEPosTaggers.MTESKTagger import AbstractSKLearnVectorGenerator

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class PositionContextWindowVectorGenerator(AbstractSKLearnVectorGenerator):
    """"
    this class implements a position bases context window generator. it takes only the words indicated
    by the list context_window_positions to the context window.
    """
    context_window_positions = []

    def __str__(self):
        return u"PConWin(%s)" % self.context_window_positions

    def generate_vector(self, tagged_sents, is_test=False):

        X = []
        Y = []
        for sentence in tagged_sents:
            for (position, _) in enumerate(sentence):
                current_sent = {}
                for offset in self.context_window_positions:
                    if offset + position < 0 or offset + position > len(sentence) - 1:
                        continue
                    (word, tag) = sentence[position + offset]
                    if not offset == 0:
                        if not is_test:
                            current_sent.update({'word(%s)' % (offset): word,
                                                 'pos(%s)' % (offset): tag})
                        else:
                            current_sent.update({'word(%s)' % (offset): word})
                    else:
                        current_sent.update({'word(%s)' % (offset): word})
                __, tag_at_pos = sentence[position]

                X.append(current_sent)
                Y.append(tag_at_pos)

        return X, Y


class baseline(PositionContextWindowVectorGenerator):
    context_window_positions = [0]
    __metaclass__ = PositionContextWindowVectorGenerator


class left3(PositionContextWindowVectorGenerator):
    context_window_positions = [-3, -2, -1, 0]
    __metaclass__ = PositionContextWindowVectorGenerator


class left2(PositionContextWindowVectorGenerator):
    context_window_positions = [-2, -1, 0]
    __metaclass__ = PositionContextWindowVectorGenerator


class left1(PositionContextWindowVectorGenerator):
    context_window_positions = [-1, 0]
    __metaclass__ = PositionContextWindowVectorGenerator


class right3(PositionContextWindowVectorGenerator):
    context_window_positions = [3, 2, 1, 0]
    __metaclass__ = PositionContextWindowVectorGenerator


class right2(PositionContextWindowVectorGenerator):
    context_window_positions = [2, 1, 0]
    __metaclass__ = PositionContextWindowVectorGenerator


class right1(PositionContextWindowVectorGenerator):
    context_window_positions = [1, 0]
    __metaclass__ = PositionContextWindowVectorGenerator


class around3(PositionContextWindowVectorGenerator):
    context_window_positions = [-3, -2, -1, 0, 1, 2, 3]
    __metaclass__ = PositionContextWindowVectorGenerator


class around2(PositionContextWindowVectorGenerator):
    context_window_positions = [-2, -1, 0, 1, 2]
    __metaclass__ = PositionContextWindowVectorGenerator


class around1(PositionContextWindowVectorGenerator):
    context_window_positions = [-1, 0, 1, ]
    __metaclass__ = PositionContextWindowVectorGenerator


class SuffixContextWindowVectorGenerator(AbstractSKLearnVectorGenerator):
    """"
    this class implements a suffix based context window generator. it takes only the words indicated
    by the list context_window_positions to the context window. from the words, only the suffixes (last 2 chars)
    are taken
    """
    context_window_positions = []

    def __str__(self):
        return u"SConWin(%s)" % self.context_window_positions

    def generate_vector(self, tagged_sents, is_test=False):

        X = []
        Y = []
        for sentence in tagged_sents:
            for (position, _) in enumerate(sentence):
                current_sent = {}
                for offset in self.context_window_positions:
                    if offset + position < 0 or offset + position > len(sentence) - 1:
                        continue
                    (word, tag) = sentence[position + offset]
                    if not offset == 0:
                        if not is_test:
                            current_sent.update({'word(%s)' % (offset): word[-2:],
                                                 'pos(%s)' % (offset): tag})
                        else:
                            current_sent.update({'word(%s)' % (offset): word[-2:]})
                    else:
                        current_sent.update({'word(%s)' % (offset): word[-2:]})
                __, tag_at_pos = sentence[position]

                X.append(current_sent)
                Y.append(tag_at_pos)

        return X, Y


class suf_baseline(SuffixContextWindowVectorGenerator):
    context_window_positions = [0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_left3(SuffixContextWindowVectorGenerator):
    context_window_positions = [-3, -2, -1, 0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_left2(SuffixContextWindowVectorGenerator):
    context_window_positions = [-2, -1, 0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_left1(SuffixContextWindowVectorGenerator):
    context_window_positions = [-1, 0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_right3(SuffixContextWindowVectorGenerator):
    context_window_positions = [3, 2, 1, 0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_right2(SuffixContextWindowVectorGenerator):
    context_window_positions = [2, 1, 0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_right1(SuffixContextWindowVectorGenerator):
    context_window_positions = [1, 0]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_around3(SuffixContextWindowVectorGenerator):
    context_window_positions = [-3, -2, -1, 0, 1, 2, 3]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_around2(SuffixContextWindowVectorGenerator):
    context_window_positions = [-2, -1, 0, 1, 2]
    __metaclass__ = SuffixContextWindowVectorGenerator


class suf_around1(SuffixContextWindowVectorGenerator):
    context_window_positions = [-1, 0, 1, ]
    __metaclass__ = SuffixContextWindowVectorGenerator
