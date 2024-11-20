import re
from sys import argv

rst = open(argv[1]).readlines()
nvtxml = open('nvt.xml').readlines()

pos_prng = list()
pos_system = list()
pos_ensemble = list()
for i, line in enumerate(rst):
    if re.search("prng", line):
        pos_prng.append(i)
    elif re.search("system", line):
        pos_system.append(i)
    elif re.search("ensemble", line):
        pos_ensemble.append(i)

assert len(pos_prng) == 2
assert len(pos_system) == 2
assert len(pos_ensemble) == 2

prng = rst[pos_prng[0]:pos_prng[1]+1]
system1 = rst[pos_system[0]:pos_ensemble[0]]
system2 = rst[pos_ensemble[1]+1:pos_system[1]+1]
ensemble = rst[pos_ensemble[0]:pos_ensemble[1]+1]

fp = open("nvt.xml", "w")
for line in nvtxml:
    if re.search("PRNG", line):
        for c in prng:
            fp.write(c)
    elif re.search("SYSTEM", line):
        for c in system1:
            fp.write(c)
        fp.write(ensemble[0])
        print("         <bias>", file=fp)
        print("            <force forcefield=\"plumed\" nbeads=\"1\"></force>", file=fp)
        print("         </bias>", file=fp)
        for c in ensemble[1:-1]:
            fp.write(c)
        fp.write(ensemble[-1])
        for c in system2:
            fp.write(c)
    else:
        fp.write(line)
fp.close()
