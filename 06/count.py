#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
# cat data.txt | python bigram-count-simple.py map ru | sort -k1,1 | python bigram-count-simple.py reduce > output.txt
#

from __future__ import division

import sys
import re
from collections import Counter, defaultdict as ddict
from math import log
from Queue import PriorityQueue as PQueue
from heapq import heappush, heappop, heappushpop, merge as heapmerge, nlargest

SKIPPED_WORDS = set([
    u'в', u'и', u'на', u'с', u'по', u'года', u'из', u'году', u'не', u'а', u'к', u'от',
    u'для', u'был', u'что', u'его', u'как', u'за', u'до', u'также',
] + [
    'the', 'of', 'and', 'in', 'a', 'to', 'was', 'is', 'for', 'as', 'on',
    'by', 'with', 'he', 'at', 'that', 'from', 'his', 'it', 'an'
])

MAX_ARTICLES = 20


class Mapper:

    re_en = re.compile(ur'[a-z]+')
    re_ru = re.compile(ur'[а-яё]+')

    def __init__(self, lang):
        if lang == 'en':
            self.re = Mapper.re_en
            self.docs_number = 4268154

        else:
            self.re = Mapper.re_ru
            self.docs_number = 1236325

    def run_mapper(self):
        data = self.read_mapper_input()
        n = 0
        words = []
        words_ques = ddict(list)
        texts_ctrs = ddict(int)
        for docid, text in data:
            matches = self.re.finditer(text.lower())
            ctr = ddict(int)
            words_number = 0
            for word in matches:
                word = word.group(0)
                if word in SKIPPED_WORDS:
                    continue
                words_number += 1
                ctr[word] += 1

            for word, number in ctr.items():
                que = words_ques[word]
                item = (number / words_number, docid)
                if len(que) < 20:
                    heappush(que, item)
                else:
                    heappushpop(que, item)

                texts_ctrs[word] += 1

        for word, que in words_ques.items():
            docs = map(lambda _ : '%08f\t%s' %_,  words_ques[word])
            print('%s\t%d\t%s' % (word, texts_ctrs[word], '\t'.join(docs))).encode('utf-8')

    def read_mapper_input(self):
        for line in sys.stdin:
            yield unicode(line, 'utf8').strip().split('\t', 1)

    def read_reducer_input(self):
        for line in sys.stdin:
            yield unicode(line, 'utf8').strip().split('\t')

    def process_words(self, iterator):
        docs_number = self.docs_number
        counter = 0
        prev_word = None
        docs = []

        for line in iterator:
            word = line[0]
            if word != prev_word and prev_word is not None:
                idf = log(docs_number / counter)
                articles = map(lambda _: '%d:%f' % (_[1], _[0] * idf),
                               nlargest(MAX_ARTICLES, docs))
                print('%s\t%s' % (prev_word, '\t'.join(articles))).encode('utf-8')
                counter = 0

            new_docs = []
            for i in range(2, len(line), 2):
                new_docs.append((float(line[i]), int(line[i + 1])))

            docs = heapmerge(docs, new_docs)

            counter += int(line[1])

            prev_word = word

        if prev_word is not None:
            idf = log(docs_number / counter)
            articles = map(lambda _: '%d:%f' % (_[1], _[0] * idf),
                           nlargest(MAX_ARTICLES, docs))
            print('%s\t%s' % (prev_word, '\t'.join(articles))).encode('utf-8')

    def run_reducer(self):
        self.process_words(self.read_reducer_input())


if __name__ == '__main__':
    mr_func = sys.argv[1]
    if mr_func == 'map':
        lang = sys.argv[2]
        mapper = Mapper(lang)
        mapper.run_mapper()
    elif mr_func == 'reduce':
        lang = sys.argv[2]
        reducer = Mapper(lang)
        reducer.run_reducer()
