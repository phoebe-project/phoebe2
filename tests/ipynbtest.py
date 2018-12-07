"""
requires runipy (pip install runipy)
requires ipython (sudo pip install jupyter; sudo apt-get install ipython-notebook)

usage:
    >>> import ipynbtest
    >>> passed, failures = ipynbtest.test_notebook('mynotebook.ipynb')
    or
    >>> ipynbtest.test_all_in_dir('mydirectory')

"""

from runipy.notebook_runner import NotebookRunner
from IPython.nbformat.current import read
from copy import deepcopy
from glob import glob
import os
import json

def _filter_exceptions(l):
    # this should return True when a line should be compared with the expected output
    # whenever this returns False, that line will be ignored
    return 'WARNING' not in l \
        and 'ShimWarning' not in l \
        and '<matplotlib' not in l \
        and 'DeprecationWarning' not in l \
        and 'cmarching.reproject' not in l \
        and 'cmarching.discretize' not in l \
        and 'ceclipse.graham_scan_inside_hull' not in l

def test_notebook(filename):
    nb = read(open(filename), 'json')

    expected = deepcopy(nb['worksheets'][0]['cells'])

    # TODO: figure out how to set the log-level for IPython here
    r = NotebookRunner(nb)
    r.run_notebook()



    actual = r.nb['worksheets'][0]['cells']

    failed = []

    for exp, act in zip(expected, actual):
        if exp['cell_type'] == 'code':
            #print("comparing outputs for ", exp['input'])
            for eo,ao in zip(exp['outputs'], act['outputs']):
                #print("\t", eo['text'], ao['text'], eo['text']==ao['text'])
                eotext = ['\n'.join(l for l in eo['text'].split('\n') if _filter_exceptions(l))]
                aotext = ['\n'.join(l for l in ao['text'].split('\n') if _filter_exceptions(l))]
                if eotext != aotext:
                    #print("\tFAILED")
                    failed.append({'prompt_number': exp['prompt_number'], 'input': exp['input'], 'expected_output': eotext, 'actual_output': aotext})


    return len(failed)==0, failed, r.nb


def test_all_in_dir(dir, interactive=True, skip=[]):
    os.chdir(dir)
    for fname in sorted(glob('*ipynb')):
        if os.path.basename(fname) in skip:
            print("SKIPPING", fname)
            continue
        print("TESTING IPYTHON NOTEBOOK", fname)
        passed, failures, actual_nb = test_notebook(fname)
        if not passed:
            print("TUTORIAL FAILED: {}".format(fname))
            for failure in failures:
                print("\tLine: {}\n\tInput:\n\t\t{}".format(failure['prompt_number'], failure['input'].replace('\n', '\n\t\t')))
                print("\tExpected Output:\n\t\t{}\n\tActual Output:\n\t\t{}".format(failure['expected_output'][0], failure['actual_output'][0]))
            print()
            if interactive:
                print("Options ({}):\n\ti: (ignore) discrepancy is acceptable".format(fname))
                # print "\ta: (accept) revise ipynb file to match actual output
                print("\tq: (quit) quit testing to fix ipynb or source")

                resp = raw_input("choice : ")
                if 'i' in resp or 'I' in resp:
                    continue
                # TODO: add retry option (so that you don't need to start all tests over after fixing a tutorial)
                # TODO: make 'a' work correctly
                # elif resp in ['a', 'A']:
                #     print("UPDATING {}".format(fname))
                #     f = open(fname, 'w')
                #     f.write(json.dumps(actual_nb, indent=0))
                #     f.close()
                else:
                    print("aborting tests - fix ipynb file or source and try again")
                    exit()


if __name__ == '__main__':
    import sys

    if not len(sys.argv) == 1:
        print("usage: ipynbtest.py notebook.ipynb")

    passed, failures = test_notebook(sys.argv[0])
    print("PASSED: ", passed)
    for f in failures:
        print(f)
