.. :changelog:

History
=======


0.14.1 (2020-12-02)
------------
Relaxes the s3fs depenency version.

0.14.0 (2020-10-22)
------------

Thanks to Anderson Banihirwe, @andersy005, for this release!

* Updates joblib pin "joblib>=0.15.0" and related code.
* Tests and code formatting improvements!

0.13.0 (2019-12-02)
------------

Thanks to Dan Maljovec, @dmaljovec, for these fixes and additions!

* Updates ``wrapt`` dependency and makes Artifact proxies compatible.
* Adds optional PyTorch model serialization.
* Adds helpful error message when a user does not set a default repo.

0.12.0 (2018-10-08)
------------
* Change default hashing algorithm to MD5 since SHA1 for performance considerations.
* Extends serialziaiton so the type used is inferred off of type.
* Makes the default serializer for Pandas DataFrames and Series to use Parquet.
* (breaking change!) Remove names from ArtifactSets, use a JSONB of labels instead.
* Doc tweaks.

0.11.0 (2018-08-23)
------------
* Optional Google Storage support.
* Adds `persistent_connections` option to Postgres repo so NullPoll can be used when appropriate.
* Doc tweaks.


0.10.0 (2016-04-30)
------------

* Change the default artifact name from the function name to the fully qualified module and function name.
  This will invalidate previously cached artifacts unless the names are migrated or explicitly set.
* Documentation! A start at least, more docstrings and guides will be added soon.
* Adds ``use_cache`` parameter and config option for when you only want to track provenance but not look for cache hits.
* Adds ``check_mutations`` option to prevent ``Artifact`` value mutations.
* Adds ``tags`` parameter to the ``provenance`` decorator for when you only want to track provenance but not look for cache hits.
* Adds experimental (alpha!) ``keras`` support.
* Adds a visualization module, pretty basic and mostly for docs and to illustrate what is possible.
* Adds ``ensure_proxies`` decorator to guard against non ``ArtifactProxy`` being sent to functions.

0.9.4.2 (2016-03-23)
---------------------

* Improved error reporing when paramiko not present for SFTP store.

0.9.4.1 (2016-03-22) (0.9.4 was a bad release)
---------------------

* Adds ability for a database and/or schema to be created when it doesn't exist.
* Adds SFTP blobstore as separate package provenance[sftp].
* Adds examples to illustrate how the library is used.

0.9.3 (2016-02-17)
---------------------

* Patch release to fix packaging problems in 0.9.2.

0.9.2 (2016-02-17)
---------------------

* Adds archive_file feature.

0.9.1 (2015-10-05)
---------------------

* Python versions now supported: 2.7, 3.3, 3.4, 3.5

0.9.0 (2015-10-05)
---------------------

* First release on PyPI. Basic functionality but lacking in docs.
