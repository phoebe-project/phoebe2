
import os as _os
import subprocess as _subprocess
import json as _json

from .common import __version__, _run_cmd

from .localthread import LocalThreadJob, LocalThreadServer
from .remotethread import RemoteThreadJob, RemoteThreadServer
from .remoteslurm import RemoteSlurmJob, RemoteSlurmServer


def list_servers():
    """
    List server configurations already saved.

    These can be opened via <crimpl.load_server> and removed with
    <crimpl.remove_server>.

    Returns
    ----------
    * list
    """
    directory = _os.path.expanduser("~/.crimpl/servers/")
    try:
        return [_os.path.basename(f)[:-5] for f in _run_cmd("ls {}".format(directory), log_cmd=False).split()]
    except _subprocess.CalledProcessError:
        return []

def load_server(name):
    """
    Load a server configuration from disk.

    Returns
    ----------
    * the appropriate server object (<LocalThreadServer>, <RemoteThreadServer>,
        <RemoteSlurmServer>)
    """
    filename = _os.path.join(_os.path.expanduser("~/.crimpl/servers"), "{}.json".format(name))
    if not _os.path.exists(filename):
        raise ValueError("could not find configuration at {}".format(filename))
    with open(filename, 'r') as f:
        d = _json.load(f)

    if 'crimpl' not in d.keys():
        raise ValueError("input configuration missing 'crimpl' entry")
    classname = d.pop('crimpl')
    version = d.pop('crimpl.version', None)
    return globals().get(classname)(**d)

def remove_server(name):
    """
    Remove a server configuration from disk (this is not reversible!)
    """
    filename = _os.path.join(_os.path.expanduser("~/.crimpl/servers"), "{}.json".format(name))
    if not _os.path.exists(filename):
        raise ValueError("could not find configuration at {}".format(filename))
    _os.remove(filename)
