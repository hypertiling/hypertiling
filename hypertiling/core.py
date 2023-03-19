from typing import Union
from .ion import htprint
from .kernel_abc import Tiling, Graph
from .kernel.SRG import StaticRotationalGraph
from .kernel.SR import StaticRotational
from .kernel.SRL import StaticRotationalLegacy
from .kernel.DUN86 import LegacyDunham
from .kernel.DUN07 import Dunham
from .kernel.GR import GenerativeReflection
from .kernel.GRG import GenerativeReflectionGraph
from .kernel.GRGS import GenerativeReflectionGraphStatic
from enum import Enum

TILINGS = {
    "SR": StaticRotational,
    "SRG": StaticRotationalGraph,
    "SRL": StaticRotationalLegacy,
    "DUN86": LegacyDunham,
    "DUN07": Dunham,
    "GR": GenerativeReflection
}

GRAPHS = {
    "GRG": GenerativeReflectionGraph,
    "GRGS": GenerativeReflectionGraphStatic
}


class TilingKernels(Enum):
    StaticRotational = "SR"
    StaticRotationalGraph = "SRG"
    StaticRotationalLegacy = "SRL"
    LegacyDunham = "DUN86"
    Dunham = "DUN07"
    GenerativeReflection = "GR"


class GraphKernels(Enum):
    GenerativeReflectionGraph = "GRG"
    GenerativeReflectionGraphStatic = "GRGS"


def HyperbolicTiling(p: int, q: int, n: int, kernel: Union[TilingKernels, str] = TilingKernels.StaticRotational,
                     **kwargs) -> Tiling:
    """
    The factory pattern function which invokes a hyperbolic tiling
    Choose your kernel using the "kernel" attribute

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : Tiling
        selects the construction kernel
    **kwargs : dictionary
        further keyword arguments to be passed to the kernel
    """

    if isinstance(kernel, TilingKernels):
        kernel = kernel.value

    if not (kernel in TILINGS):
        raise AttributeError("Provided kernel is not a TilingKernel")

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError(
            "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == StaticRotationalLegacy:
        htprint("Warning", "This kernel is deprecated! Better use the 'SR' kernel instead!")
    if kernel == GenerativeReflection:
        htprint("Status", "Parameter n is interpreted as number of reflective layers. Compare documentation.")
    if kernel in [StaticRotational, StaticRotationalGraph, StaticRotationalLegacy, Dunham, LegacyDunham]:
        htprint("Status", "Parameter n is interpreted as number of layers. Compare documentation.")

    return TILINGS[kernel](p, q, n, **kwargs)


def HyperbolicGraph(p: int, q: int, n: int, kernel: Union[GraphKernels, str] = GraphKernels.GenerativeReflectionGraph,
                    **kwargs) -> Graph:
    """
    The factory pattern  function which invokes a hyperbolic graph
    Choose your kernel using the "kernel" attribute
    
    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : Graph
        selects the construction kernel
    **kwargs : dictionary
        further keyword arguments to be passed to the kernel
    """

    if isinstance(kernel, GraphKernels):
        kernel = kernel.value

    if not (kernel in GRAPHS):
        raise AttributeError("Provided kernel is not a GraphKernel")

    if (p - 2) * (q - 2) <= 4:
        raise AttributeError(
            "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    if p > 20 or q > 20 and n > 5:
        htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == GenerativeReflectionGraph:
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")

    return GRAPHS[kernel](p, q, n, **kwargs)
