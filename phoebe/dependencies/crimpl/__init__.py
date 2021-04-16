
import os as _os
import subprocess as _subprocess
import json as _json
from .common import __version__, _run_cmd
from .awsec2 import AWSEC2Job, AWSEC2Server, list_awsec2_volumes, list_awsec2_instances, terminate_awsec2_instance, delete_awsec2_volume, terminate_all_awsec2_instances, delete_all_awsec2_volumes
from .remoteslurm import RemoteSlurmJob, RemoteSlurmServer


def list_servers():
    """
    List server configurations already saved.

    These can be opened via <crimpl.open>

    Returns
    ----------
    * list
    """
    directory = _os.path.expanduser("~/.crimpl/servers/")
    try:
        return [_os.path.basename(f)[:-5] for f in _run_cmd("ls {}".format(directory)).split()]
    except _subprocess.CalledProcessError:
        return []

def load_server(name):
    """
    Load a server configuration from disk.

    Returns
    ----------
    * the appropriate server object
    """
    filename = _os.path.join(_os.path.expanduser("~/.crimpl/servers"), "{}.json".format(name))
    if not _os.path.exists(filename):
        raise ValueError("could not find configuration at {}".format(filename))
    with open(filename, 'r') as f:
        d = _json.load(f)
    classname = d.pop('Class')
    return globals().get(classname)(**d)
