try:
    from numba import njit
    AVAILABLE = True
except:
    AVAILABLE = False


def check_numba(f):
    if AVAILABLE:
        return njit(f)
    else:
        return f
