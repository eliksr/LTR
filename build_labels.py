# import csv
#
# fname = '/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/CXR-labels.txt'
# with open(fname) as f:
#     content = f.readlines()
#
# map = {}
# for line in content:
#     pair = line.split()
#     map[pair[0]] = pair[1]
#
# w = csv.writer(open("labels.csv", "w"))
# for key, val in map.items():
#     w.writerow([key, val])
import os

fname = '/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/data/chestx/MeSH/original/openi.mesh.top'
outputFile = '/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/data/chestx/MeSH/original/openi.mesh.top.new'
croppedDir = '/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/croped'
# croppedSet = set()

croppedSet = {filename for filename in os.listdir(croppedDir)}

with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content]

with open(outputFile, 'w') as output:
    for line in content:
        image = line.split('|')[0][1:]
        if image in croppedSet:
            output.write(line + os.linesep)
