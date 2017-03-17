==========
provenance
==========

.. image:: https://travis-ci.org/Savvysherpa/provenance.svg?branch=master
    :target: https://travis-ci.org/Savvysherpa/provenance

provenance is a Python library for function-level provenance. By decorating
functions you are able to cache the results, i.e. artifacts, to memory, disk, or S3.
The artifact stores can be layered, e.g. you can write to a local disk store and
and a global team S3 one. Every artifact also keeps metadata associated with it
on how it was created (i.e. the function and arguments used) so you can always
answer the question "Where did this come from?".  The most featureful metadata store
is backed by Postgres and is recommended for any serious use. The library is general
purpose but was built for machine learning pipelines and plays nicely with numpy and
other pydata libraries. You can basically think of this as joblib's memoize but on
steroids.



Status
=======

The library was extracted from a production system and has been used help
collaboration on a number of other projects (e.g. sharing models and features
easily over s3). The API should be pretty stable but as of now the only documentation
are the tests. We will be adding proper documentation and logging soon.


Example
=======

TODO

.. code:: python

    >>> import provenance as p

Installation
============


.. code:: bash

    conda install -c conda-forge provenance

    or

    pip install provenance

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
