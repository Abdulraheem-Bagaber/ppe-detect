"""Shared vocabulary for the whole system.

This package depends on nothing but the standard library and numpy. Every
other package imports it; it imports no other package. Keeping it empty of
behaviour is what allows ``ppe.rules`` to stay free of ``ultralytics``,
``cv2``, ``supervision`` and ``fastapi``.
"""
