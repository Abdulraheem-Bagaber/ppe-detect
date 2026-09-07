"""Detector backends.

This ``__init__`` stays free of backend imports on purpose: importing
``ppe.detection.base`` must never drag ``ultralytics`` (and therefore
``torch``) into the process. Import the concrete module you want, or use
:func:`ppe.detection.factory.build_detector`, which imports lazily.
"""
