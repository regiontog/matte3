from functools import wraps
from inspect import signature
from IPython.display import display
import matplotlib.pylab as plt

import sympy as sp

#
# General
#


def disallow_none_kwargs(f):
    required_kwargs = []
    for param in signature(f).parameters.values():
        if param.default is None:
            required_kwargs.append(param.name)

    @wraps(f)
    def wrapper(*args, **kwargs):
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise Exception(f"Keyword argument {kwarg} is required.")

        return f(*args, **kwargs)
    return wrapper


def stringify(value):
    if isinstance(value, float):
        return str(value if not value.is_integer() else int(value))
    else:
        return str(value)


def pp_table(table, v_sep='|', h_sep='-', cross_sep='+'):
    just = []
    for key, col in table.items():
        just.append(max(len(stringify(key)), *(len(stringify(cell)) for cell in col)))

    print(f" {v_sep} ".join(header.ljust(just[i]) for i, header in enumerate(table.keys())))
    print(f"{h_sep}{cross_sep}{h_sep}".join(h_sep*just[i] for i, _ in enumerate(table.keys())))

    for row in zip(*table.values()):
        print(f" {v_sep} ".join(stringify(cell).ljust(just[i]) for i, cell in enumerate(row)))


def group_dicts(dicts):
    iterable = iter(dicts)
    head = next(iterable)
    keys = head.keys()

    result = {key: [] for key in keys}
    for key, value in head.items():
        result[key].append(value)

    for dict in iterable:
        assert dict.keys() == keys, "Dictionaries must have same shape"
        for key, value in dict.items():
            result[key].append(value)

    return result


def pp(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        pp_table(group_dicts(fn(*args, **kwargs)))

    return wrapper


def with_error(results, y, x_key='x', y_key='y'):
    for result in results:
        y_i = y(result[x_key])
        yield {**result, "error": abs(y_i - result[y_key]), 'exact': y_i}


#
# ODE, Euler
#


def ivp(expr, x, ivs):
    eqs = (sp.Eq(expr.subs(x, iv[0]), iv[1]) for iv in ivs)
    free_symbols_solutions = sp.solve(eqs, dict=True)

    if len(free_symbols_solutions) == 0:
        raise Exception(f"Free symbols in expr has no solutions")
    elif len(free_symbols_solutions) > 1:
        raise Exception(f"Free symbols in expr has multiple solutions\n{list(free_symbols_solutions)}")

    return expr.subs(free_symbols_solutions[0])


def euler_normal(f, h, t):
    return lambda w, i: w + h * f(t(i-1), w)


def euler_trapezoid(f, h, t):
    def trapezoid(w, i):
        t_i = t(i-1)
        w_n = f(t_i, w)
        return w + h*(w_n + f(t_i+h, w + h*w_n))/2

    return trapezoid


def euler_midpoint(f, h, t):
    def midpoint(w, i):
        t_i = t(i-1)
        w_n = f(t_i, w)
        return w + h*f(t_i+h/2, w + h*w_n/2)

    return midpoint


def euler_rk4(f, h, t):
    def rk4(w, i):
        t_i = t(i-1)

        s1 = f(t_i, w)
        s2 = f(t_i + (h / 2), w + (h / 2) * s1)
        s3 = f(t_i + (h / 2), w + (h / 2) * s2)
        s4 = f(t_i + h, w + h * s3)
        return w + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4)

    return rk4


@disallow_none_kwargs
def euler(f, h=1, t=None, iv=None, method=euler_normal):
    fro, to = iv[0], t

    n = (to - fro) / h
    if not (n.is_integer() and n > 0):
        raise Exception("Number of iterations must be a positive integer.")

    n = int(n)

    t_i = lambda i: fro + i/(1/h)  # Trying to avoid floating point rounding errors
    step = method(f, h, t_i)
    w = iv[1]
    yield dict(i=0, t=t_i(0), w=iv[1])

    for i in range(1, n+1):
        w = step(w, i)

        yield dict(i=i, t=t_i(i), w=w)


@disallow_none_kwargs
def euler_error(f, iv=None, multiple_eqs_strategy=lambda eqs: eqs[0], **kwargs):
    y = sp.Function('y')
    t = sp.Symbol('t')

    y_d = sp.Eq(y(t).diff(t), f(t, y(t)))
    diff_eq = sp.dsolve(y_d)

    if isinstance(diff_eq, list):
        diff_eq = multiple_eqs_strategy(diff_eq)

    exact = ivp(diff_eq.rhs, t, [iv])

    display(y_d)
    display(sp.Eq(y(t), exact))

    y_fn = sp.lambdify(t, exact)

    return with_error(euler(f, iv=iv, **kwargs), y_fn, x_key='t', y_key='w')


def plot(results, exact_key='exact', estimate_key='y', x_key='x'):
    w = []
    y = []
    t = []

    for result in results:
        w.append(result[estimate_key])
        y.append(result[exact_key])
        t.append(result[x_key])

    plt.plot(t, w)
    plt.plot(t, y)


def plot_gen(results, y_keys=None, x_keys=None):
    ys = [[] for _ in y_keys]
    xs = [[] for _ in x_keys]

    for result in results:
        for key, y in zip(y_keys, ys):
            y.append(result[key])

        for key, x in zip(x_keys, xs):
            x.append(result[key])

    for y, x in zip(ys, xs):
        plt.plot(x, y)


def body3(y, m1, m2, m3, g):
    return sp.Matrix([
        y[1],
        (g * m2 * (y[4] - y[0])) / ((y[4] - y[0]) ** 2 + (y[6] - y[2]) ** 2) ** (3 / 2) + (g * m3 * (y[8] - y[0])) / (
        (y[8] - y[0]) ** 2 + (y[10] - y[2]) ** 2) ** (3 / 2),
        y[3],
        (g * m2 * (y[6] - y[2])) / ((y[4] - y[0]) ** 2 + (y[6] - y[2]) ** 2) ** (3 / 2) + (g * m3 * (y[10] - y[2])) / (
        (y[8] - y[0]) ** 2 + (y[10] - y[2]) ** 2) ** (3 / 2),
        y[5],
        (g * m1 * (y[0] - y[4])) / ((y[0] - y[4]) ** 2 + (y[2] - y[6]) ** 2) ** (3 / 2) + (g * m3 * (y[8] - y[4])) / (
        (y[8] - y[4]) ** 2 + (y[10] - y[6]) ** 2) ** (3 / 2),
        y[7],
        (g * m1 * (y[2] - y[6])) / ((y[0] - y[4]) ** 2 + (y[2] - y[6]) ** 2) ** (3 / 2) + (g * m3 * (y[10] - y[6])) / (
        (y[8] - y[4]) ** 2 + (y[10] - y[6]) ** 2) ** (3 / 2),
        y[9],
        (g * m2 * (y[4] - y[8])) / ((y[4] - y[8]) ** 2 + (y[6] - y[10]) ** 2) ** (3 / 2) + (g * m1 * (y[0] - y[8])) / (
        (y[0] - y[8]) ** 2 + (y[2] - y[10]) ** 2) ** (3 / 2),
        y[11],
        (g * m2 * (y[6] - y[10])) / ((y[4] - y[8]) ** 2 + (y[6] - y[10]) ** 2) ** (3 / 2) + (g * m1 * (y[2] - y[10])) / (
        (y[0] - y[8]) ** 2 + (y[2] - y[10]) ** 2) ** (3 / 2)
    ])


def body3_y():
    x1, y1, v1x, v1y = sp.symbols("x_1 y_1 v_1_x v_1_y")
    x2, y2, v2x, v2y = sp.symbols("x_2 y_2 v_2_x v_2_y")
    x3, y3, v3x, v3y = sp.symbols("x_3 y_3 v_3_x v_3_y")

    return sp.Matrix([x1, v1x, y1, v1y, x2, v2x, y2, v2y, x3, v3x, y3, v3y])


def body3_iv(b1, v1, b2, v2, b3, v3):
    return sp.Matrix([b1[0], v1[0], b1[1], v1[1], b2[0], v2[0], b2[1], v2[1], b3[0], v3[0], b3[1], v3[1]])
