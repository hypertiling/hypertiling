import os
import numpy as np
from ...geodesics import geodesic_arc
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import SVG, display
from typing import Optional



from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

class SvgElement(ABC):
    """Abstract base class for SVG elements with common styling."""
    
    def __init__(
        self,
        fill: str = "none",
        edgecolor: str = "black",
        lw: float = 1.0,
        digits: int = 5,
        **svg_attrs,
    ):
        self.fill = fill
        self.edgecolor = edgecolor
        self.lw = lw
        self.digits = digits
        self.svg_attrs = svg_attrs
    
    @abstractmethod
    def to_svg(self) -> str:
        """Generate the SVG string representation."""
        pass
    
    def __repr__(self):
        """Developer-friendly representation."""
        attrs = ", ".join(f"{k}={v!r}" for k, v in self._get_repr_attrs().items())
        return f"{self.__class__.__name__}({attrs})"
    
    def __str__(self):
        """User-friendly representation."""
        return self._get_str_repr()
    
    def _get_repr_attrs(self) -> dict:
        """
        Override in subclasses to specify which attributes to show in repr.
        Default shows common styling attributes.
        """
        return {
            "fill": self.fill,
            "edgecolor": self.edgecolor,
            "lw": self.lw,
        }
    
    def _get_str_repr(self) -> str:
        """
        Override in subclasses for custom user-friendly string.
        Default returns class name.
        """
        return f"{self.__class__.__name__}"
    
    # Common setters
    def set_fill(self, color: str):
        """Change fill color."""
        self.fill = color
        return self
    
    def set_edgecolor(self, color: str):
        """Change edge color."""
        self.edgecolor = color
        return self
    
    def set_linewidth(self, lw: float):
        """Change line width."""
        self.lw = lw
        return self
    
    def set_attr(self, **attrs):
        """Set additional SVG attributes."""
        self.svg_attrs.update(attrs)
        return self
    
    def _build_base_attrs(self, extra: dict = None) -> dict:
        """Build common attributes dict."""
        attrs = {
            "fill": self.fill,
            "stroke": self.edgecolor,
            "stroke-width": self.lw,
        }
        if extra:
            attrs.update(extra)
        return attrs
    

    


class SvgCanvas:
    """
    Canvas holding SVG elements that can be modified before rendering.
    
    Parameters
    ----------
    center : tuple[float, float], optional
        Center coordinates (cx, cy) in disk space.
    radius : float, optional
        Disk radius in disk space.
    padding : float, optional
        Padding around the viewBox.
    width : int, optional
        SVG width in pixels.
    height : int, optional
        SVG height in pixels.
    """
    
    def __init__(
        self,
        center: tuple[float, float] = (0, 0),
        radius: float = 1.0,
        padding: float = 5,
        width: int = 600,
        height: int = 600,
    ):
        self.elements: list[SvgElement] = []
        
        # SVG parameters
        self.center = center
        self.radius = radius
        self.padding = padding
        self.width = width
        self.height = height
    
    def add(self, *elements: SvgElement):
        """Add one or more elements."""
        self.elements.extend(elements)
        return self
    
    def remove(self, element: SvgElement):
        """Remove an element."""
        self.elements.remove(element)
        return self
    
    def clear(self):
        """Remove all elements."""
        self.elements.clear()
        return self
    
    def _build_header(self) -> str:
        """Build SVG opening tag with viewBox."""
        cx, cy = self.center
        r = self.radius
        vb_min_x = cx - r - self.padding
        vb_min_y = cy - r - self.padding
        vb_width = 2 * r + 2 * self.padding
        vb_height = 2 * r + 2 * self.padding
        
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' "
            f"width='{self.width}' height='{self.height}' "
            f"viewBox='{vb_min_x} {vb_min_y} {vb_width} {vb_height}'>\r"
        )
    
    def _build_footer(self) -> str:
        """Build SVG closing tag."""
        return "</svg>\r"
    
    def render(self) -> str:
        """Render complete SVG document."""
        parts = [self._build_header()]
        
        # Add all elements in order
        for elem in self.elements:
            parts.append(elem.to_svg())
            parts.append("\r")
        
        parts.append(self._build_footer())
        return "".join(parts)
    
    def save(self, filename: str):
        """Save SVG to file."""
        with open(filename, 'w') as f:
            f.write(self.render())
        return self
    
    # Collection interface
    def __len__(self): 
        return len(self.elements)
    
    def __getitem__(self, idx): 
        return self.elements[idx]
    
    def __iter__(self): 
        return iter(self.elements)
    
    def __repr__(self):
        return f"SvgCanvas({len(self.elements)} elements)"

def build_svg_attrs(base_attrs: dict, **svg_attrs) -> str:
    """
    Merge base attributes with additional SVG attributes.
    Converts underscores to hyphens (stroke_width -> stroke-width).
    svg_attrs override base_attrs.
    
    Example:
        build_svg_attrs({"fill": "red", "stroke": "black"}, opacity=0.5)
        # Returns: "fill='red' stroke='black' opacity='0.5'"
    """
    base_attrs.update({k.replace('_', '-'): v for k, v in svg_attrs.items()})
    return " ".join(f"{k}='{v}'" for k, v in base_attrs.items())


def to_px(z, factor=100, offset=1):
    """
    Transforms complex number to px coordinates

    Arguments:
    ----------
    z : np.complex
        coordinate in the complex plane
    factor : int
        some large scaling factor to conform to px scale
    offset : int
        offset plot region
        
    """
    x = np.real(z) + offset
    x *= factor
    y = np.imag(z) + offset
    y *= factor
    return x, y


class svgString():
    """
    Helper class, makes working with strings more convenient
    
    """

    def __init__(self):
        header = f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' " \
                 f"width='500px' height='500px' viewBox='0 0 200 200'>" + "\r\n"
        self.string = header

    def write(self, string):
        self.string += string

    def newline(self):
        self.string += "\n"

    def tabstop(self):
        self.string += "\t"

    def print(self):
        return self.string


# def display(canvas: SvgCanvas):
#     """
#     Display an SvgCanvas in Jupyter/IPython.
    
#     Parameters
#     ----------
#     canvas : SvgCanvas
#         The canvas to display.
#     """
#     from IPython.display import SVG, display as ipython_display
#     ipython_display(SVG(canvas.render()))

def draw_svg(content: str):
    """
    Use IPython display API for displaying SVG
    """
    display(SVG(content))


def write_svg(fname: str, content: svgString):
    """
    Write svgString to file
    """
    os.remove(fname) if os.path.exists(fname) else None
    svgfile = open(fname, 'w')
    svgfile.write(content)
    svgfile.close()


def norm_0_1(x, cmin=None, cmax=None):
    """
    Normalize an array like x linearly between 0 and 1

    Arguments:
    __________
    x : 1d array like
        contains data to be normalized between 0 and 1
    cmin : float, default = None
        the value that is mapped to 0
        if None, the minimal value of x is taken
    cmax : float, default = None
        the value that is mapped to 1
        if None, the maximal value of x is taken

    """
    if not cmin:
        cmin = min(x)
    else:
        cmin = cmin
    if not cmax:
        cmax = max(x)
    else:
        cmax = cmax
    x = np.array(x)
    return (x - cmin) / (cmax - cmin)


def array_to_rgb(x, cmap):
    """
    Takes an array like in the range of [0,1] and return a 2d array containing the rgb values in the range [0, 255]
    in respect to cmap

    Arguments:
    __________
    x : 1d array like
        contains data in the range [0,1] to be mapped to rgb values
    cmap :  matplotlib.colors.LinearSegmentedColormap
        the colormap that is used to calculate the rgb values

    """
    rgb = np.zeros((len(x), 3))
    for idx, val in enumerate(x):
        rgb[idx] = cmap(val)[:3]
    return (rgb * 255).astype(int)
