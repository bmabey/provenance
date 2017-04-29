==========
provenance
==========

.. image:: https://travis-ci.org/bmabey/provenance.svg?branch=master
    :target: https://travis-ci.org/bmabey/provenance

provenance is a Python library for function-level provenance. By decorating
functions you are able to cache the results, i.e. artifacts, to memory, disk, or S3.
The artifact stores can be layered, e.g. you can write to a local disk store and
and a global team S3 one. Every artifact also keeps metadata associated with it
on how it was created (i.e. the function and arguments used) so you can always
answer the question "Where did this come from?". The library is general
purpose but was built for machine learning pipelines and plays nicely with numpy and
other pydata libraries. You can basically think of this as joblib's memoize but on
steroids.



Status
=======

The library was extracted from a production system and has been used in other
systems and has helped multiple teams collaborate (e.g. sharing models and features
easily over s3). The API is stable and basic documentation exists but automatic
documentation is not yet hooked up. We will be adding additional documentation,
a documentation build, and logging soon.


Example
=======

For an explanation of this example please see the `Introductory Guide notebook <https://github.com/bmabey/provenance/blob/master/notebook-docs/Introductory%20Guide.ipynb>`_.

.. code-block:: python

    import provenance as p

    p.load_config(...)

    import time
    
    @p.provenance()
    def expensive_add(a, b):
        time.sleep(2)
        return a + b
    
    
    @p.provenance()
    def expensive_mult(a, b):
        time.sleep(2)
        return a * b


    a1 = expensive_add(4, 3)
    a2 = expensive_add(1, 1)

    result = expensive_mult(a1, a2)

    vis.visualize_lineage(result)


.. image:: https://raw.githubusercontent.com/bmabey/provenance/master/docs/source/images/lineage_example.png



Installation
============


.. code:: bash

    pip install provenance


Getting Started
===============

For an quick overview of the basics please see the `Introductory Guide notebook <https://github.com/bmabey/provenance/blob/master/notebook-docs/Introductory%20Guide.ipynb>`_.
The `examples <https://github.com/bmabey/provenance/tree/master/examples>`_ directory contains some user contributed example setups with READMEs.

Additional guides and docs built from the docstrings will be made available soon.

Development
===========

Assuming you have conda installed, the following commands can be used to create a development environment.

Initial environment creation

.. code:: bash

    conda env create
    source activate provenance-dev
    pip install -r requirements.txt
    pip install -r test_requirements.txt

Reactivating the environment after it has been created

.. code:: bash

    source activate provenance-dev
