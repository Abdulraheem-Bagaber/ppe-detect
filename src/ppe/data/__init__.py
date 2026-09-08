"""Dataset preparation: source exports in, our three-class YOLO layout out.

Kept out of the runtime path -- nothing in ``ppe.pipeline`` or ``ppe.api``
imports this. It exists as a package rather than living in ``scripts/`` so
the label rewriting can be unit-tested without any images on disk.
"""
