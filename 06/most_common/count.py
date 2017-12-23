#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
# cat data.txt | python bigram-count-simple.py map ru | sort -k1,1 | python bigram-count-simple.py reduce > output.txt
#

import sys
import re
from collections import Counter, defaultdict
from heapq import heappush, heappop, heappushpop, merge as heapmerge, nlargest

MC_NUMBER = 20

class Mapper:

    re_en = re.compile(ur'[a-z]+')
    re_ru = re.compile(ur'[а-яё]+')

    def __init__(self, lang):
        if lang == 'en':
            self.re = Mapper.re_en
        else:
            self.re = Mapper.re_ru

    def run(self):
        texts = 0
        words = defaultdict(int)
        for line in sys.stdin:
            texts += 1
            _, text = unicode(line, "utf-8").rsplit('\t', 1)
            matches = self.re.finditer(text.lower())
            for word in matches:
                words[word.group(0)] += 1

        for word, num in words.items():
            print ('%s\t%d' % (word, num)).encode("utf-8")


class Reducer:

    def run(self):
        cur_word = None
        cur_count = 0
        top20 = []

        for line in sys.stdin:
            word, count = unicode(line, "utf-8").split('\t')
            count = int(count)

            if cur_word == word:
                cur_count += count
            else:
                if len(top20) < MC_NUMBER:
                    heappush(top20, (cur_count, cur_word))
                else:
                    heappushpop(top20, (cur_count, cur_word))

                cur_count = count
                cur_word = word

        if len(top20) < MC_NUMBER:
            heappush(top20, (cur_count, cur_word))
        else:
            heappushpop(top20, (cur_count, cur_word))

        for count, word in nlargest(MC_NUMBER, top20):
            print ('%s\t%d' % (word, count)).encode("utf-8")



if __name__ == '__main__':
    mr_func = sys.argv[1]
    if mr_func == 'map':
        lang = sys.argv[2]
        mapper = Mapper(lang)
        mapper.run()
    elif mr_func == 'reduce':
        reducer = Reducer()
        reducer.run()
