import os
DISDIR = os.path.dirname(os.path.abspath(__file__))

from . import satpops, hosts, relations, plot
from . import baryons, dark_matter
from . import observations

from .core import list_alternatives, get_alternative
