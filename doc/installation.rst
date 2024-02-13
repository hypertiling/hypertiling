.. Copyright (c) 2024, Manuel Schrauth, Florian Goth



Installation
============

``hypertiling`` package is available in the `PyPI <https://pypi.org/>`__ package index and can be installed using

.. code-block:: bash

    pip install hypertiling

For optimal performance, we recommend using hypertiling together with python-numba. If it is not already present on your system, it can be installed automatically using the ``[numba]``-suffix, i.e.

.. code-block:: bash

    pip install hypertiling[numba]

The package can also be locally installed from our public git repository via

.. code-block:: bash

        git clone https://git.physik.uni-wuerzburg.de/hypertiling/hypertiling
        cd hypertiling
        pip install .

For developer installation use

.. code-block:: bash

    pip install -e .

