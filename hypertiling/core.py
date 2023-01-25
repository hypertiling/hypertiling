from .ion import htprint
from .kernel_abc import Tiling, Graph
from .kernel.SRG import StaticRotationalGraph
from .kernel.SR import StaticRotational
from .kernel.SRL import StaticRotationalLegacy
from .kernel.DUN import LegacyDunham
from .kernel.GR import GenerativeReflection
from .kernel.GRG import GenerativeReflectionGraph
from .kernel.SGRG import StaticReflectionGraph
from enum import Enum


class Tilings(Enum):
    StaticRotational = StaticRotational
    StaticRotationalGraph = StaticRotationalGraph
    StaticRotationalLegacy = StaticRotationalLegacy
    LegacyDunham = LegacyDunham
    GenerativeReflection = GenerativeReflection


class Graphs(Enum):
    GenerativeReflectionGraph = GenerativeReflectionGraph
    StaticReflectionGraph = StaticReflectionGraph


def HyperbolicTiling(p: int, q: int, n: int, kernel: Tilings = Tilings.StaticRotational, **kwargs) -> Tiling:
    """
    The factory pattern function which invokes a hyperbolic tiling

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : str
        selects the construction algorithm
    """
    if not isinstance(kernel, Tilings):
        raise AttributeError("Provided kernel is not a Tiling")
    kernel = kernel.value

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError(
            "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == StaticRotationalLegacy:
        htprint("Warning", "This kernel is deprecated! Better use the 'SR' kernel instead!")
    if kernel == GenerativeReflection:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
    if kernel in [StaticRotational, StaticRotationalGraph, StaticRotationalLegacy]:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
    if kernel == LegacyDunham:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        htprint("Warning", "Dunham kernel is only implemented for legacy reasons and largely untested. Use with care!")

    # if kernel not in TILINGS:
    #     raise KeyError("[hypertiling] Error: No valid kernel specified")

    return kernel(p, q, n, **kwargs)


def HyperbolicGraph(p: int, q: int, n: int, kernel: Graphs = Graphs.StaticRotationalGraph, **kwargs) -> Graph:
    """
    The factory pattern  function which invokes a hyperbolic graph

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : str
        selects the construction algorithm
    """

    if not isinstance(kernel, Graphs):
        raise AttributeError("Provided kernel is not a Graph")
    kernel = kernel.value

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError(
            "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == GenerativeReflectionGraph:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")

    # if kernel not in GRAPHS:
    #     raise KeyError("[hypertiling] Error: No valid kernel specified")

    return kernel(p, q, n, **kwargs)
