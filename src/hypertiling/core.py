from typing import Union
from .ion import htprint
from .kernel_abc import Tiling, Graph
from .kernel.SRG import StaticRotationalGraph
from .kernel.SRS import StaticRotationalSector
from .kernel.SRL import StaticRotationalLegacy
from .kernel.DUN86 import LegacyDunham
from .kernel.DUN07 import Dunham
from .kernel.DUN07X import DunhamX
from .kernel.GR import GenerativeReflection
from .kernel.GRG import GenerativeReflectionGraph
from .kernel.GRGS import GenerativeReflectionGraphStatic
from enum import Enum

TILINGS = {
    "SRS": StaticRotationalSector,
    "SRG": StaticRotationalGraph,
    "SRL": StaticRotationalLegacy,
    "DUN86": LegacyDunham,
    "DUN07": Dunham,
    "DUN07X": DunhamX,
    "GR": GenerativeReflection
}

GRAPHS = {
    "GRG":  GenerativeReflectionGraph,
    "GRGS": GenerativeReflectionGraphStatic
}


class TilingKernels(Enum):
    StaticRotationalSector = "SRS"
    StaticRotationalGraph  = "SRG"
    StaticRotationalLegacy = "SRL"
    LegacyDunham = "DUN86"
    Dunham = "DUN07"
    DunhamX = "DUN07X"
    GenerativeReflection = "GR"


class GraphKernels(Enum):
    GenerativeReflectionGraph       = "GRG"
    GenerativeReflectionGraphStatic = "GRGS"


def HyperbolicTiling(p: int, q: int, n: int, kernel: Union[TilingKernels, str] = TilingKernels.StaticRotationalSector,
                     **kwargs) -> Tiling:
    """
    The factory pattern function which invokes a hyperbolic tiling
    Select your kernel using the "kernel" attribute

    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : Tiling
        sets the construction kernel
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
    if kernel in [StaticRotationalSector, StaticRotationalGraph, StaticRotationalLegacy, Dunham, DunhamX, LegacyDunham]:
        htprint("Status", "Parameter n is interpreted as number of layers. Compare documentation.")

    return TILINGS[kernel](p, q, n, **kwargs)


def HyperbolicGraph(p: int, q: int, n: int, kernel: Union[GraphKernels, str] = GraphKernels.GenerativeReflectionGraph,
                    **kwargs) -> Graph:
    """
    The factory pattern  function which invokes a hyperbolic graph
    Select your kernel using the "kernel" attribute
    
    Parameters
    ----------
    p : int
        number of vertices per cells
    q : int
        number of cells meeting at each vertex
    n : int
        number of layers to be constructed
    kernel : Graph
        sets the construction kernel
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
