#!/usr/bin/python
import re 

textfile = open('..\test_text', 'r')
for line in textfile:
    print (line)
    line = re.sub("\\[.*?\\]", "",line)
textfile.close()
