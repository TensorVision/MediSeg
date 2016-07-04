#!/usr/bin/env python

"""Create filelists to use for training and testing."""

import os
import json

# Adjust to your needs:
os.environ['DATA_PATH'] = '/home/moose/GitHub/MediSeg/DATA'

# train_data
train_data = []
test_data = []
for opi in [i for i in range(1, 6 + 1)]:
    op = "OP%i" % opi
    directory = os.path.join(os.environ['DATA_PATH'], op)
    files_data = [os.path.join(directory, f)
                  for f in sorted(os.listdir(directory))
                  if f.endswith('.png')]
    for i, el in enumerate(sorted(files_data)):
        if i < 40 * 2 and opi <= 4:
            if i % 2 == 0:
                train_data.append({'raw': el, 'mask': None})
            else:
                train_data[-1]['mask'] = el
        else:
            if i % 2 == 0:
                test_data.append({'raw': el, 'mask': None})
            else:
                test_data[-1]['mask'] = el

# write data
with open('trainfiles.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4, separators=(',', ': '))

with open('testfiles.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4, separators=(',', ': '))
