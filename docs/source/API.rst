API
===

.. currentmodule:: provenance

Primary API
~~~~~~~~~~~~~~~~

.. autosummary::
   provenance
   load_artifact
   load_proxy
   ensure_proxies
   promote
   provenance_set
   capture_set
   create_set
   load_set_by_id
   load_set_by_name
   archive_file

Configuration
~~~~~~~~~~~~~

.. autosummary::
   from_config
   load_config
   load_yaml_config
   current_config
   get_repo_by_name
   set_default_repo
   get_default_repo
   set_check_mutations
   get_check_mutations
   set_run_info_fn
   get_use_cache
   set_use_cache
   using_repo


Utils
~~~~~

.. autosummary::
   is_proxy
   lazy_dict
   lazy_proxy_dict

Visualization
~~~~~~~~~~~~~

.. currentmodule:: provenance.vis

.. autosummary::
   visualize_lineage


Detailed Docs
~~~~~~~~~~~~~

.. currentmodule:: provenance


Primary API

.. autofunction:: provenance
.. autofunction:: load_artifact
.. autofunction:: load_proxy
.. autofunction:: ensure_proxies
.. autofunction:: promote
.. autofunction:: provenance_set
.. autofunction:: capture_set
.. autofunction:: create_set
.. autofunction:: load_set_by_id
.. autofunction:: load_set_by_name
.. autofunction:: archive_file


Configuration

.. autofunction:: from_config
.. autofunction:: load_config
.. autofunction:: load_yaml_config
.. autofunction:: current_config
.. autofunction:: get_repo_by_name
.. autofunction:: set_default_repo
.. autofunction:: get_default_repo
.. autofunction:: set_check_mutations
.. autofunction:: get_check_mutations
.. autofunction:: set_run_info_fn
.. autofunction:: get_use_cache
.. autofunction:: set_use_cache
.. autofunction:: using_repo


Utils

.. autofunction:: is_proxy
.. autofunction:: lazy_dict
.. autofunction:: lazy_proxy_dict

Visualization (beta)

.. currentmodule:: provenance.vis

.. autofunction:: visualize_lineage
