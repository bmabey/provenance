from boltons import funcutils as bfu
from google.cloud import storage as gs
from memoized_property import memoized_property

from . import blobstores as bs

# TODO: catch and retry w/new client on
# BrokenPipeError: [Errno 32] Broken pipe
# ConnectionResetError: [Errno 54] Connection reset by peer
# more?


def retry(f, max_attempts=2):

    @bfu.wraps(f)
    def with_retry(store, *args, **kargs):
        actual_attempts = 0
        while True:
            try:
                return f(store, *args, **kargs)
            except (BrokenPipeError, ConnectionError) as e:
                actual_attempts += 1
                if actual_attempts >= max_attempts:
                    raise e
                else:
                    store._setup_client()

    return with_retry


class GSStore(bs.RemoteStore):

    def __init__(
        self,
        cachedir,
        bucket,
        basepath='',
        project=None,
        read=True,
        write=True,
        read_through_write=True,
        delete=False,
        on_duplicate_key='skip',
        cleanup_cachedir=False,
        always_check_remote=False,
    ):
        """
        Parameters
        ----------
        always_check_remote : bool
           When True GS (Google Storage) will be checked with every __contains__ call. Otherwise it will
        short-circuit if the blob is found in the cachedir. For performance reasons this
        should always be set to False. The only reason why you would want to use this
        is if you are using a GSStore and a DiskStore in a ChainedStore together for
        some reason. Since the GSStore basically doubles as a DiskStore with it's cachedir
        chaining the two doesn't really make sense though.
        """
        super(GSStore, self).__init__(
            always_check_remote=always_check_remote,
            cachedir=cachedir,
            basepath=basepath,
            cleanup_cachedir=cleanup_cachedir,
            read=read,
            write=write,
            read_through_write=read_through_write,
            delete=delete,
            on_duplicate_key=on_duplicate_key,
        )

        self.bucket_name = bucket
        self.project = project

    def _setup_client(self):
        del self._client
        del self._bucket
        # force re-memoization
        assert self.bucket is not None

    @memoized_property
    def client(self):
        return gs.Client(project=self.project)

    @memoized_property
    def bucket(self):
        return self.client.get_bucket(self.bucket_name)

    @retry
    def _exists(self, path):
        blobs = list(self.bucket.list_blobs(prefix=path))
        return len(blobs) == 1

    @retry
    def _delete_remote(self, path):
        self.blob(path).delete()

    def _blob(self, path):
        return self._bucket.blob(path)

    @retry
    def _upload_file(self, filename, path):
        self._blob(path).upload_from_filename(filename)

    @retry
    def _download_file(self, remote_path, dest_filename):
        self._blob(remote_path).download_to_filename(dest_filename)
