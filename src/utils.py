import tempfile
import os

def get_root(root):
    directory = os.environ.get(root)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    return root_dir