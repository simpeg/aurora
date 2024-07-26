.. image:: docs/figures/aurora_logo.png
   :width: 900
   :alt: AURORA

|

.. image:: https://img.shields.io/pypi/v/aurora.svg
    :target: https://pypi.python.org/pypi/aurora

.. image:: https://img.shields.io/conda/v/conda-forge/aurora.svg
    :target: https://anaconda.org/conda-forge/aurora

.. image:: https://img.shields.io/pypi/l/aurora.svg
    :target: https://pypi.python.org/pypi/aurora

Aurora is an open-source package that robustly estimates single station and remote reference electromagnetic transfer functions (TFs) from magnetotelluric (MT) time series.  Aurora is part of an open-source processing workflow that leverages the self-describing data container `MTH5 <https://github.com/kujaku11/mth5>`_, which in turn leverages the general `mt-metadata <https://github.com/kujaku11/mth5>`_ framework to manage metadata.  These pre-existing packages simplify the processing by providing managed data structures, transfer functions to be generated with only a few lines of code.  The processing depends on two inputs -- a table defining the data to use for TF estimation, and a JSON file specifying the processing parameters, both of which are generated automatically, and can be modified if desired.  Output TFs are returned as mt-metadata objects, and can be exported to a variety of common formats for plotting, modeling and inversion.  

Key Features
-------------

- Tabular data indexing and management (Pandas dataframes), 
- Dictionary-like processing parameters configuration
- Programmatic or manual editing of inputs
- Largely automated workflow 

Documentation for the Aurora project can be found at http://simpeg.xyz/aurora/

Installation
---------------

Suggest using PyPi as the default repository to install from

``pip install aurora``

Can use Conda but that is not updated as often

``conda -c conda-forge install aurora``

General Work Flow
-------------------

1. Convert raw time series data to MTH5 format, see `MTH5 Documentation and Examples <https://mth5.readthedocs.io/en/latest/index.html>`_.
2. Understand the time series data and which runs to process for local station `RunSummary`.
3. Choose remote reference station ``KernelDataset``.
4. Create a recipe for how the data will be processed ``Config``.
5. Estimate transfer function `process_mth5` and out put as a ``mt_metadata.transfer_function.core.TF`` object which can output [ EMTFXML | EDI | ZMM | ZSS | ZRR ] files. 


