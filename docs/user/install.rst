.. _install:

Installation
============

BISIP is compatible with Python 3.6+.

Requirements
----------------

The following packages are required and should be installed automatically on setup:

- `numpy <https://numpy.org/>`_
- `cython <https://cython.org/>`_
- `matplotlib <https://matplotlib.org/>`_
- `emcee <https://emcee.readthedocs.io/en/stable/>`_

These optional packages are used for progress bars and corner plots:

- `tqdm <https://tqdm.github.io/>`_
- `corner <https://corner.readthedocs.io/en/latest/>`_

Package managers
----------------

TODO: Add BISIP to conda-forge.

From source
-----------

BISIP is developed on `GitHub <https://github.com/clberube/bisip2>`_.
Clone the repository to your computer.
Then navigate to the bisip directory.
Finally run the setup.py script with Python.

.. code-block:: bash

  git clone https://github.com/clberube/ml4rocks
  cd bisip2
  python setup.py install -f

Testing
-----------
