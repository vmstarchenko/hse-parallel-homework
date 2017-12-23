#!/usr/bin/env python
import sys
cur_word = None
cur_count = 0
for line in sys.stdin:
    data = unicode(line, "utf-8").split('\t')
    word = data[0]
    count = int(data[ 1])
    if cur_word == word:
        cur_count += count
    else:
        if cur_word:
            print ('%s\t%d' % (cur_word, cur_count)).encode( "utf-8")
        cur_count = count
        cur_word = word
print ('%s\t%d' % (cur_word, cur_count)).encode( "utf-8")
