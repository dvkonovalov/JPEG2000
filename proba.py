import os
from array import *


file = open('test.bin', 'wb')
data = open('file.jpeg2000', 'r')
for i in range(5):
    bitstring = data.readline()

for i in range(3):
    bitstring = data.readline()
    print(len(bitstring))
    splits = [bitstring[x:x + 8] for x in range(0, len(bitstring)-8, 8)]

    bin_array_in = array('B')
    for split in splits:
        bin_array_in.append(int(split, 2))
    bin_array_in.tofile(file)

