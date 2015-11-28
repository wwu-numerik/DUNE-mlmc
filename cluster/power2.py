#!/usr/bin/env python

"""Simple demonstration of solving the Poisson equation in 2D using pyMOR's builtin discretizations.

Usage:
    speedup.py [options] TEMPLATE STARTNODE POWER MACRO MICRO

Arguments:
    TEMPLATE        Which template file to use

    STARTNODE       Number of nodes to begin speedup run with

    POWER           How often to double the number of nodes

    MACRO           How many macro cells per dimension

    MICRO           How many micro cells per dimension

Options:
    -h, --help   	 Show this message.
    --threads=THREADS    Threads per rank [default: 1]
"""

from jinja2 import FileSystemLoader, Environment
from os.path import dirname, realpath
from docopt import docopt
from pprint import pprint
import math

try:
    loader = FileSystemLoader(dirname(realpath(__file__)), followlinks=True)
except TypeError as t:
    loader = FileSystemLoader(dirname(realpath(__file__)))

inargs = docopt(__doc__)
inargs['THREADS'] = inargs['--threads']
tpl_fn = inargs['TEMPLATE']
tpl = Environment(loader=loader).get_template(tpl_fn)

args = {'THREADS': 1, 'NODES': 2, 'MACRO': 8, 'MICRO': 4, 'POWER': 3, 'STARTNODE': 0 }
for key, value in args.items():
    try:
        args[key] = int(inargs.get(key, value))
    except ValueError:
        continue

cfg = {'supermuc' : (28, [2, 3, 5, 10, 19, 37, 74, 147, 293, 586]),
       'cheops': (12, [2, 3, 6, 11, 22, 43, 86, 171, 342, 683])}
for hpc, (cores, nodes) in cfg.items():
    max_int = max(args['POWER'], len(nodes)-args['STARTNODE'])
    for i in range(args['STARTNODE'], max_int):
        n = nodes[i]
        args['NODES'] = n
        ranks = int(math.pow(2, math.floor(math.log2(n*cores))))
        args['POWER2_RANKS'] = ranks
        print(ranks)

        fn = '{2}_batch_speedup_pow2_{0:06}_{1}'.format(ranks, tpl_fn.replace('/', '_'), hpc)
        with open(fn, 'wb') as out:
            out.write(bytes(tpl.render(**args), 'UTF-8'))
        print('$SUBMIT {}'.format(fn))

