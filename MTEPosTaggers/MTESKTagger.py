# -*- coding: UTF-8 -*-
"""This File Contains the Scikit-Learn based Part of Speech Tagger with it's helper classes.
"""

from nltk.metrics import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

try:
    from functools import reduce
except ImportError:
    "No python 3"

__author__ = "Alexander Böhm [jwacalex], Thomas Stieglmaier, Thomas Ziegler"
__credits__ = ["Alexander Böhm [jwacalex]", "Thomas Stieglmaier", "Thomas Ziegler", "Behrang Qasemizadeh"]
__license__ = "LGPL"


class AbstractSKLearnVectorGenerator():
    """
    this is the abstract implementation of a a vector generator for scikit learn based taggers. For more information
    how to use it, please refer to the generate_vector-method.
    """

    def __init__(self):
        """
        create a new learning vector generator.
        :return: Instance of VectorGenerator
        """

    def __repr__(self):
        """
        representation of class
        :return: representation of class
        :rtype: String
        """
        return self.__str__()

    def __unicode__(self):
        """
        unicoderepresentation of class
        :return: unicoderepresentation of class
        :rtype: String
        """
        return self.__str__()

    def __str__(self):
        """
        string representation of generator . should be csv safe and human readable
        :return: string representation of generator
        :rtype: String
        """
        raise NotImplementedError("__str__ must be implemented by child")

    def generate_vector(self, tagged_sents, is_test=False):
        """
        this function generates a X and Y vectors based the tagged sentences. it returns a tuple of two lists, the X
        and the Y vector.

        The X-Vector is a list of dictionaries. The same features must be named as the same values within the X-Dict
        and the tag of the current word will go th the Y-Vector.

        For each word in each sentence the following values will be appended to the X and Y Vector-List:
        X-Vector: dict()-of features
        Y-Vector: Tag

        Example:
        Sentence: This//TAG1, is//TAG2, test//TAG3 (Tags are after each word, separated by doubleslashes)
        Vector to generate: the word left to the current word (left1).

        This//TAG1

        X-Vector(.append): {"word(0)": "This"}
        Y-Vector(.append): TAG1


        is//TAG2
        X-Vector(.append): {"word(-1)": "This", "tag(-1)", "TAG1", "word(0)"; "is"}
        Y-Vector(.append): TAG2


        test//TAG3
        X-Vector(.append): {"word(-1)": "is", "tag(-1)", "TAG2", "word(0)"; "test"}
        Y-Vector(.append): TAG3


        Resulting Vectors:

        X: {"word(0)": "This"}, {"word(-1)": "This", "tag(-1)", "TAG1", "word(0)"; "is"}, {"word(-1)": "is", "tag(-1)", "TAG2", "word(0)"; "test"}
        Y: TAG1,TAG2,TAG3


        For testing  the classifier under real world examples it might be useful to strip all tags of the words  or
        to omit some information (in the example above omitting the tags for the words). Such a situation will be
        indicated by switching the is_test-parameter to True.

        :param tagged_sents: sents to generate vector for
        :type tagged_sents: List
        :return: X and Y vector for the current sentence
        :param is_test: indicates wheather the input should be vectorized for testing (stripping tags) or training
        :type is_test: bool
        :rtype: List(dict()), List
        """
        raise NotImplementedError("generate_vector must be implemented by child")


class MTESKTagger:
    '''
    This is a scikit-learn based tagger for text annotated using the MTE tag set.
    It should not be used for other tag sets, as it works with MTE tags internally.
    '''

    def __init__(self, tagged_sents, anonProperNouns=False, classifier=MultinomialNB(), vectorizer=DictVectorizer(),
                 context_window_generator=AbstractSKLearnVectorGenerator()):
        '''
        Construct a new MTESKTagger and train it with the sentences from tagged_sents.

        :param tagged_sents: Tagged sentences to train the tagger.
        :type tagged_sents: [[(word:str, tag:str)]]
        :param anonProperNouns: Set 'True' to replace every proper noun with an anonymous string. Currently only for MTE tags.
        :type anonProperNouns: bool
        :param classifier: scikit-learn classifier to use for part of speech tagging. Classifier must be able to perform multiclass classification.
        :type classifier: BaseEstimator
        :param vectorizer: vectorizer to perform vectorizsation of
        :type vectorizer: BaseEstimator
        :param context_window_generator: Context Window Generator which will get the current sentence to handle
        :type context_window_generator: AbstractSKLearnVectorGenerator
 
        '''

        self.__vectorizer = None
        self.__tagged_sents = []
        ANON = "anon"

        if anonProperNouns:
            for s in tagged_sents:
                tmp = []
                for (w, tag) in s:
                    if tag.startswith("#Np"):
                        tmp.append((ANON, "#Np"))
                    else:
                        tmp.append((w, tag))
                self.__tagged_sents.append(tmp)
        else:
            self.__tagged_sents = tagged_sents

        self.__context_window_generator = context_window_generator
        self.__classifier = classifier
        self.__vectorizer = vectorizer

        self.__X_train, self.__Y_train = self.__apply_context_window(tagged_sents, self.__context_window_generator,
                                                                     is_test=False)
        self.__classifier.fit(self.__vectorizer.fit_transform(self.__X_train), self.__Y_train)

    def __apply_context_window(self, tagged_sents, context_window_generator, is_test=False):
        """
        This function applies the context window to all sents and returns the X and y Vectors for scikit-learn classifiers.

        :param tagged_sents: Tagged sentences to train the tagger.
        :type tagged_sents: [[(word:str, tag:str)]]
        :param context_window_generator: Context Window with relative position to the currently learned word (indicated by 0).
        :type context_window_generator: List
        :param is_test: indicates wheather the input should be vectorized for testing (stripping tags) or training
        :type is_test: bool
        :return: X and y vector for scikit-learn classifiers and metrics
        :rtype (List,List)
        """
        X = []
        Y = []

        assert isinstance(context_window_generator,
                          AbstractSKLearnVectorGenerator), "context_window_generator is not an instance of " \
                                                           "AbstractSKLearnVectorGenerator"
        return context_window_generator.generate_vector(tagged_sents, is_test)

    def evaluate(self, gold):
        """
        Score the accuracy of the tagger against the gold standard.
        Strip the tags from the gold standard text, retag it using
        the tagger, then compute the accuracy score.

        :type gold: list(list(tuple(str, str)))
        :param gold: The list of tagged sentences to score the tagger on.
        :rtype: float
        """

        X_test, Y_test = self.__apply_context_window(gold, self.__context_window_generator, is_test=True)
        predicted = self.__classifier.predict(self.__vectorizer.transform(X_test))

        return accuracy(Y_test, predicted)

    def __mkSet(self, X_test, Y_test, predicted):
        """
        this function converts a scikit learn output of a classifer into a NLTK compatible one.

        :param X_test: X-Vector of the dataset to test with.
        :type X_test: List
        :param Y_test: List of gold standard values to test the classifier with.
        :type Y_test: List
        :param predicted: List of predicted values.
        :type predicted: List
        :return: Tuple of Sets. First Set is gold standard, second one is the current result.
        :rtype (Set,Set)
        """
        X_test = list(X_test)
        Y_test = list(Y_test)
        Y_predicted = list(predicted)

        gold = set()
        result = set()
        for x, tag_gold, tag_predicted in zip(X_test, Y_test, Y_predicted):
            word = x['word(0)']

            gold.add((word, tag_gold))
            result.add((word, tag_predicted))

        return gold, result

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

        X_test, Y_test = self.__apply_context_window(gold, self.__context_window_generator, is_test=True)
        predicted = self.__classifier.predict(self.__vectorizer.transform(X_test))

        gold_tokens_set, test_tokens_set = self.__mkSet(X_test, Y_test, predicted)

        gold_tokens = sum(gold, [])
        test_tokens = list(test_tokens_set)
        gold_tags = [t for (_, t) in gold_tokens]
        test_tags = [t for (_, t) in test_tokens]


        # calculate out of vocabulary words
        if oov:
            d = {word: True for (word, _) in reduce(lambda a, b: a + b, self.__tagged_sents, [])}
            aov = reduce(lambda a, b: a + 1 if not b in d else a, [w for (w, _) in gold_tokens], 0)
            aov = (aov * 100.0) / len(gold_tokens)
        else:
            aov = '-1'

        acc = accuracy(Y_test, predicted)
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
