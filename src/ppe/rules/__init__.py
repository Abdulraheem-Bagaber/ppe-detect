"""The compliance rule layer.

**Import discipline, enforced by ``tests/test_import_boundaries.py``:** nothing
in this package may import ``ultralytics``, ``cv2``, ``supervision``, ``torch``
or ``fastapi``. Permitted imports are the standard library, ``numpy``,
``pydantic`` (for config types) and ``ppe.core``.

That constraint is the point of the project. It means the business rule --
"a person in the work area must have a helmet and a vest" -- is expressed in
plain, hand-testable geometry that can be tuned from a YAML file, rather than
being distributed across model weights where it can only be changed by
retraining and can never be explained to a safety officer.

``association`` and ``zones`` are pure functions. ``debounce`` is the single
exception: temporal confirmation is inherently stateful, so it is an explicit
deterministic state machine with no clock and no I/O -- time is passed in.
"""
