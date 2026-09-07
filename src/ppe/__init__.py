"""PPE compliance detection.

Architecture in one sentence: the detector finds ``person``, ``helmet`` and
``vest`` as *independent* classes, and a dependency-free rule layer decides
compliance geometrically, so the business rule is configuration, not weights.
"""

__version__ = "0.1.0"
