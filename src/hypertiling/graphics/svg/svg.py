# In hypertiling.graphics.svg

def make_svg(*args, **kwargs):
    """
    Tombstone function for the legacy make_svg API.
    """
    error_msg = (
        "\n\n"
        "================================================================================\n"
        "  CRITICAL API CHANGE (v1.5)\n"
        "================================================================================\n"
        "  The function 'make_svg' has been removed in favor of a new object-oriented API.\n\n"
        "  To fix your code, replace 'make_svg' with the 'SVGCanvas' workflow as"
        "  as demonstrated in the demo notebook advanced_visualization.ipynb.\n\n"
        "================================================================================\n"
    )
    raise NotImplementedError(error_msg)

def write_svg(*args, **kwargs):
    """
    Tombstone function for the legacy write_svg API.
    """
    error_msg = (
        "\n\n"
        "  The function 'write_svg' has been removed.\n"
        "  Please use standard Python file I/O instead:\n\n"
        "      with open('filename.svg', 'w') as f:\n"
        "          f.write(svg_content)\n"
    )
    raise NotImplementedError(error_msg)


def draw_svg(*args, **kwargs):
    """
    Tombstone function for the legacy draw_svg API.
    """
    error_msg = (
        "\n\n"
        "================================================================================\n"
        "  CRITICAL API CHANGE (v1.5)\n"
        "================================================================================\n"
        "  The function 'draw_svg' has been removed in favor of a new object-oriented API.\n\n"
        "  This is demonstrated in the demo notebook advanced_visualization.ipynb.\n\n"
        "================================================================================\n"
    )
    raise NotImplementedError(error_msg)