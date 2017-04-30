==========
provenance
==========

|version status| |build status| |docs|


.. |version status| image:: https://img.shields.io/pypi/v/provenance.svg
   :target: https://pypi.python.org/pypi/provenance
   :alt: Version Status
.. |build status| image:: https://travis-ci.org/bmabey/provenance.png?branch=master
   :target: https://travis-ci.org/bmabey/provenance
   :alt: Build Status
.. |docs| image:: https://readthedocs.org/projects/provenance/badge/?version=latest
   :target: https://provenance.readthedocs.org
   :alt: Documentation Status

provenance is a Python library for function-level provenance. By decorating
functions you are able to cache the results, i.e. artifacts, to memory, disk, or S3.
The artifact stores can be layered, e.g. you can write to a local disk store and
and a global team S3 one. Every artifact also keeps metadata associated with it
on how it was created (i.e. the function and arguments used) so you can always
answer the question "Where did this come from?". The library is general
purpose but was built for machine learning pipelines and plays nicely with numpy and
other pydata libraries. You can basically think of this as joblib's memoize but on
steroids.

Example
=======

For an explanation of this example please see the `Introductory Guide`_.

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


.. _Introductory Guide: http://provenance.readthedocs.io/en/latest/intro-guide.html

Installation
============


.. code:: bash

    pip install provenance


