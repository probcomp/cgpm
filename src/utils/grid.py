import numpy

dd_alpha = numpy.logspace(-1, 2, 12).tolist()

pos_logspace = numpy.logspace(-8, 8, 100).tolist()
neg_logspace = (-numpy.logspace(-8, 8, 100)).tolist()

def uniform(min_val, max_val, point_count):
    grid = numpy.array(range(point_count)) + 0.5
    grid *= (max_val - min_val) / float(point_count)
    grid += min_val
    return grid


def center_heavy(min_val, max_val, point_count):
    grid = uniform(-1, 1, point_count)
    grid = numpy.arcsin(grid) / numpy.pi + 0.5
    grid *= max_val - min_val
    grid += min_val
    return grid


def left_heavy(min_val, max_val, point_count):
    grid = uniform(0, 1, point_count)
    grid = grid ** 2
    grid *= max_val - min_val
    grid += min_val
    return grid


def right_heavy(min_val, max_val, point_count):
    grid = left_heavy(max_val, min_val, point_count)
    return grid[::-1].copy()


def pitman_yor(
        min_alpha=0.1,
        max_alpha=100,
        min_d=0,
        max_d=0.5,
        alpha_count=20,
        d_count=10):
    '''
    For d = 0, this degenerates to the CRP, where the expected number of
    tables is:
        E[table_count] = O(alpha log(customer_count))
    '''
    min_alpha = float(min_alpha)
    max_alpha = float(max_alpha)
    min_d = float(min_d)
    max_d = float(max_d)

    lower_triangle = [
        (x, y)
        for x in center_heavy(0, 1, alpha_count)
        for y in left_heavy(0, 1, d_count)
        if x + y < 1
    ]
    alpha = lambda x: min_alpha * (max_alpha / min_alpha) ** x
    d = lambda y: min_d + (max_d - min_d) * y
    grid = [
        {'alpha': alpha(x), 'd': d(y)}
        for (x, y) in lower_triangle
    ]
    return grid

pitman_yor_grid = pitman_yor()

DEFAULTS = {
    'topology': pitman_yor_grid,
    'clustering': pitman_yor_grid,
    'bb': {
        'alpha': dd_alpha,
        'beta': dd_alpha,
    },
    'dd': {
        'alpha': dd_alpha,
    },
    'dpd': {
        'gamma': (10 ** left_heavy(-1, 2, 30)).tolist(),
        'alpha': (10 ** right_heavy(-1, 1, 20)).tolist(),
    },
    'gp': {
        'alpha': numpy.logspace(-1, 5, 100).tolist(),
        'inv_beta': numpy.logspace(-5, 1, 100).tolist(),
    },
    'bnb': {
        'alpha': [2.0 ** p for p in range(-3, 1)],
        'beta': [2.0 ** p for p in range(13)],
        'r': [2 ** p for p in range(13)],
    },
    'nich': {
        'mu': neg_logspace + [0] + pos_logspace,
        'sigmasq': pos_logspace,
        'kappa': numpy.logspace(-2, 2, 30).tolist(),
        'nu': numpy.logspace(0, 2, 30).tolist(),
    },
}