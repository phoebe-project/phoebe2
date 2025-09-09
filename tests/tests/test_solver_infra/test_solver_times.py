import phoebe


def test_parse_solver_times():
    b = phoebe.default_binary()
    b.add_dataset('lc', times=phoebe.linspace(0,1,301), compute_phases=phoebe.linspace(0,1,101))
    b.set_value('mask_enabled', True)
    b.set_value('mask_phases', [(-0.1, 0.1), (0.45,0.55)])

    b.add_solver('optimizer.nelder_mead')
    b.set_value('solver_times', 'times')
    b.parse_solver_times()


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_parse_solver_times()