import os

import paramiko

from .. import blobstores as bs


def _ssh_client(ssh_config):
    client = paramiko.SSHClient()
    client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    # There still seems to be problems with some types keys.
    # See https://github.com/paramiko/paramiko/issues/243
    # So you might try uncommenting if you are using an ecdsa-sha2-nistp256
    # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(**ssh_config)
    return client


class SFTPStore(bs.RemoteStore):

    def __init__(
        self,
        cachedir,
        basepath,
        ssh_config=None,
        ssh_client=None,
        sftp_client=None,
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
           When True the SFTP server will be checked with every __contains__ call. Otherwise it will
        short-circuit if the blob is found in the cachedir. For performance reasons this
        should always be set to False. The only reason why you would want to use this
        is if you are using a SFTPStore and a DiskStore in a ChainedStore together for
        some reason. Since the SFTPStore basically doubles as a DiskStore with it's cachedir
        chaining the two doesn't really make sense though.
        """
        super(SFTPStore, self).__init__(
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

        self.ssh_client = None
        if ssh_config is not None:
            self.ssh_client = _ssh_client(ssh_config)
        if self.ssh_client is not None:
            sftp_client = paramiko.SFTPClient.from_transport(self.ssh_client._transport)
        if sftp_client is not None:
            self.sftp_client = sftp_client
        else:
            # This is to allow testing the importing/subpackage aspect without
            # having to actually test the class by mocking an ssh connection.
            if cachedir is None and basepath is None:
                return
            raise ValueError(
                'You must specify a SFTP client by passing in one of: sftp_client, ssh_config, ssh_client'
            )

    def _exists(self, path):
        try:
            self.sftp_client.stat(path)
            return True
        except FileNotFoundError:
            return False

    def _delete_remote(self, path):
        self.sftp_client.remove(path)

    def _upload_file(self, filename, path):
        self.sftp_client.put(filename, path)

    def _download_file(self, remote_path, dest_filename):
        self.sftp_client.get(remote_path, dest_filename)
