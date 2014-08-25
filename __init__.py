"""Open Drug Discovery Toolkit
==============================
Universal and easy to use resource for various drug discovery tasks, ie docking, virutal screening, rescoring.

    Attributes
    ----------
    toolkit : module,
        Toolkits backend module, currenlty OpenBabel [ob] and RDKit [rdk].
        This setting is toolkit-wide, and sets given toolkit as default
"""

import numpy as np
from .toolkits import ob, rdk

toolkit = ob
__all__ = ['toolkit']
