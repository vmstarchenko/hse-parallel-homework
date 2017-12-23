src_dir=src
tests_dir=tests
l=ru
v=2

start:
	rm output.txt -rf && cat data-$l.txt | (python$v count.py map $l)  | sort -k1,1  | (python$v count.py reduce $l) > output.txt
