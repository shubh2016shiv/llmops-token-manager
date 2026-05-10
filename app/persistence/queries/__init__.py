"""
Query module package for persistence-layer SQL constants.

This package stores named SQL statements separately from the persistence
functions that execute them. The first extraction pass is intentionally
non-behavioral: persistence modules continue using their inline queries
until each import-based replacement is validated individually.
"""
