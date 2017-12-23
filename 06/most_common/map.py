#!/usr/bin/env python
import sys
for line in sys.stdin:
    lower = unicode(line, "utf-8").lower()
    cleaned = ''.join(c for c in lower if c.isalnum() or c==' ')
    words = cleaned.split()
    for word in words:
        print ('%s\t%d' % (word, 1)).encode("utf-8")
