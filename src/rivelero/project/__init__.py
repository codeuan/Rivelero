"""Rivelero project persistence (P1).

A Rivelero project is a single ``.rivelero`` file: a ZIP container of
explicit JSON documents and NumPy ``.npz`` arrays (never pickle). It stores
the scientific state of an analysis - source survey data, the physical
World, visibility assumptions, optional derived observability products and
saved design scenarios - and references large external resources such as
terrain rasters instead of copying them. The visibility cache is never part
of a project.

The package is independent of Qt and of the GUI; the GUI converts its
ApplicationState to and from :class:`rivelero.project.io.ProjectData`.
"""
