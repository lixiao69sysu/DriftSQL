"""Deterministic schema and metric drift primitives."""

from .factory import (
    ColumnRenameExample,
    DriftExample,
    QueryFingerprint,
    build_add_column_star_example,
    build_add_column_projection_example,
    build_clean_example,
    build_column_rename_example,
    build_column_replacement_example,
    build_compound_drift_example,
    build_table_rename_example,
    fingerprint_query,
    materialize_column_rename,
    materialize_column_addition,
    materialize_column_replacement,
    materialize_schema_diff,
    materialize_table_rename,
)
from .schema import ColumnRename, SchemaDiff, rewrite_sql_identifier

__all__ = [
    "ColumnRename",
    "ColumnRenameExample",
    "DriftExample",
    "QueryFingerprint",
    "SchemaDiff",
    "build_add_column_star_example",
    "build_add_column_projection_example",
    "build_clean_example",
    "build_column_rename_example",
    "build_column_replacement_example",
    "build_compound_drift_example",
    "build_table_rename_example",
    "fingerprint_query",
    "materialize_column_rename",
    "materialize_column_addition",
    "materialize_column_replacement",
    "materialize_schema_diff",
    "materialize_table_rename",
    "rewrite_sql_identifier",
]
