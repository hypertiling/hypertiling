from typing import Union
from .ion import htprint
from .kernel_abc import Tiling, GraphExtended
from .kernel.SRG import StaticRotationalGraph
from .kernel.SRS import StaticRotationalSector
from .kernel.SRL import StaticRotationalLegacy
from .kernel.DUN86 import LegacyDunham
from .kernel.DUN07 import Dunham
from .kernel.DUN07X import DunhamX
from .kernel.GR import GenerativeReflection
from .kernel.GRG import GenerativeReflectionGraph
from .kernel.GRGS import GenerativeReflectionGraphStatic
from .kernel.GRC import GRC
from .kernel.GRCT import GRCT
from enum import Enum

TILINGS = {
    "SRS": StaticRotationalSector,
    "SRG": StaticRotationalGraph,
    "SRL": StaticRotationalLegacy,
    "DUN86": LegacyDunham,
    "DUN07": Dunham,
    "DUN07X": DunhamX,
    "GR": GenerativeReflection,
    "GRC": GRC,
    "GRCT": GRCT
}

GRAPHS = {
    "GRG": GenerativeReflectionGraph,
    "GRGS": GenerativeReflectionGraphStatic,
    "GRC": GRC,
    "GRCT": GRCT
}


class TilingKernels(Enum):
    StaticRotationalSector = "SRS"
    StaticRotationalGraph = "SRG"
    StaticRotationalLegacy = "SRL"
    LegacyDunham = "DUN86"
    Dunham = "DUN07"
    DunhamX = "DUN07X"
    GenerativeReflection = "GR"
    GRC = "GRC"
    GRCT = "GRCT"


class GraphKernels(Enum):
    GenerativeReflectionGraph = "GRG"
    GenerativeReflectionGraphStatic = "GRGS"
    GRC = "GRC"
    GRCT = "GRCT"


def HyperbolicTiling(*args, kernel: Union[TilingKernels, str] = TilingKernels.StaticRotationalSector,
                     **kwargs) -> Tiling:
    """
    The factory pattern function which invokes a hyperbolic tiling
    Select your kernel using the "kernel" attribute

    Parameters
    ----------
    *args: List[int]
        p: int
        q: int
        Optional[r: int]
        n: int
    kernel : Tiling
        sets the construction kernel
    **kwargs : dictionary
        further keyword arguments to be passed to the kernel
    """

    if isinstance(kernel, TilingKernels):
        kernel = kernel.value

    if not (kernel in TILINGS):
        raise AttributeError("Provided kernel is not a TilingKernel")

    # if (p - 2) * (q - 2) <= 4:
    #    raise AttributeError(
    #        "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    # if p > 20 or q > 20 and n > 5:
    #    htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == "SRL":
        htprint("Warning", "This kernel is deprecated! Better use the 'SR' kernel instead!")
    elif kernel == "GR":
        htprint("Status", "Parameter n is interpreted as number of reflective layers. Compare documentation.")
        htprint("Warning", "This kernel is deprecated! Better use the 'GRC' kernel instead!")
    elif kernel == "GRC":
        htprint("Status", "Parameter n is interpreted as number of reflective layers. Compare documentation.")
        kwargs["tiling"] = True
    elif kernel == "GRCT":
        htprint("Status", "Schwarzian triangle with (p, q, r) with n layers selected")
        kwargs["tiling"] = True
    elif kernel in [StaticRotationalSector, StaticRotationalGraph, StaticRotationalLegacy, Dunham, DunhamX,
                    LegacyDunham]:
        htprint("Status", "Parameter n is interpreted as number of layers. Compare documentation.")

    return TILINGS[kernel](*args, **kwargs)


def HyperbolicGraph(*args, kernel: Union[GraphKernels, str] = GraphKernels.GenerativeReflectionGraph,
                    **kwargs) -> GraphExtended:
    """
    The factory pattern  function which invokes a hyperbolic graph
    Select your kernel using the "kernel" attribute
    
    Parameters
    ----------
    *args: List[int]
        p: int
        q: int
        Optional[r: int]
        n: int
    kernel : GraphExtended
        sets the construction kernel
    **kwargs : dictionary
        further keyword arguments to be passed to the kernel
    """

    if isinstance(kernel, GraphKernels):
        kernel = kernel.value

    if not (kernel in GRAPHS):
        raise AttributeError("Provided kernel is not a GraphKernel")

    # if (p - 2) * (q - 2) <= 4:
    #    raise AttributeError(
    #        "[hypertiling] Error: Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")

    # if p > 20 or q > 20 and n > 5:
    #    htprint("Warning", "The lattice might become very large with your parameter choice!")

    if kernel == "GRG":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        htprint("Warning", "This kernel is deprecated! Better use the 'GRC' kernel instead!")
    elif kernel == "GRGS":
        htprint("Status", "Parameter n is interpreted as number of reflective layer. Compare documentation.")
        htprint("Warning", "This kernel is deprecated! Better use the 'GRC' kernel instead!")
    elif kernel == "GRC":
        htprint("Status", "Parameter n is interpreted as number of reflective layers. Compare documentation.")
        kwargs["nbrs"] = True
    elif kernel == "GRCT":
        htprint("Status", "Schwarzian triangle with (p, q, r) with n layers selected")
        kwargs["nbrs"] = True

    return GRAPHS[kernel](*args, **kwargs)
