Benchmarking
***********************************

benchmarking/profiling tests can be run individually in the tests/benchmark directory by calling

$: python -m cProfile -o [script.profile] [script.py]

or all can be run at once with a summary of times, by calling

$: python run_tests.py benchmark

in the tests directory.

You can then open the [script.profile] files using a visualizer like run snake run or snakeviz.


Below are the current times for all existing benchmarks on a laptop.

binary_eccentric_blackbody_rv.py: 18.0
binary_circular_atm_lc.py: 12.14
binary_eccentric_blackbody_lc.py: 17.65
binary_circular_blackbody_lc.py: 11.78
frontend_default_binary.py: 4.98
binary_circular_blackbody_rv.py: 12.0
binary_circular_blackbody_lc_storemesh.py: 130.28


