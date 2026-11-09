# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-level CSV Sources that keep each row readable as an exact original record.

Every data row becomes one ``CAPTURED`` :class:`CSVRowSource` whose identity is
derived only from the document identity and the primary-key value, so the same
row keeps the same Source identity across re-imports, reordering and edits of
non-key fields.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_TYPE = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
CSV_ROW_IDENTITY_PREFIX = "csv"
CSV_ROW_EVIDENCE_NAME = "csv-row"


class CSVDocumentInput(BaseModel):
    """One offline CSV document to expand into row-level Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """The adapter-native value of one parsed CSV record."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row whose identity survives reordering and non-key edits."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the delimiter-safe, bounded identity of one CSV row.

    Both components are percent-encoded with no safe characters, so a ``:`` or
    ``/`` inside either component can never merge two distinct rows.
    """

    _require_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-empty string")  # noqa: TRY003

    identity = f"{CSV_ROW_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate a whole CSV document, or fail without partial rows.

    The whole document is validated before any row is returned: an invalid
    header, a duplicated primary key, a malformed record or invalid UTF-8 makes
    the call raise :class:`ValueError` instead of returning a prefix.
    """

    _require_document_id(value.document_id)
    primary_key = value.primary_key
    if not isinstance(primary_key, str) or not primary_key.strip():
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003

    header, records = _read_records(value.content)
    _validate_header(header)
    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} does not exactly match a header column")  # noqa: TRY003

    key_index = header.index(primary_key)
    columns = tuple(header)
    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records:
        if len(record) != len(header):
            raise ValueError(  # noqa: TRY003
                f"record has {len(record)} fields but the header declares {len(header)}"
            )
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("primary_key value must not be empty or blank")  # noqa: TRY003
        if primary_key_value in seen:
            raise ValueError(f"duplicate primary_key value {primary_key_value!r}")  # noqa: TRY003
        seen.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def _require_document_id(document_id: object) -> None:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not have leading or trailing whitespace")  # noqa: TRY003


def _read_records(content: bytes) -> tuple[list[str], list[list[str]]]:
    """Decode, parse and split a CSV document without interpreting any field."""

    if not isinstance(content, bytes | bytearray | memoryview):
        raise ValueError("CSV content must be bytes")  # noqa: TRY003
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        records = [record for record in reader if record]
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error
    if not records:
        raise ValueError("CSV document must contain a header record")  # noqa: TRY003
    return records[0], records[1:]


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV header record must not be empty")  # noqa: TRY003
    seen: set[str] = set()
    for name in header:
        if not name.strip():
            raise ValueError("CSV header names must be non-empty and not blank")  # noqa: TRY003
        if name in seen:
            raise ValueError(f"duplicate CSV header name {name!r}")  # noqa: TRY003
        seen.add(name)


class _CSVRowAdapter:
    """Resolve :class:`CSVRowInput` into captured :class:`CSVRowSource` values."""

    name = CSV_ROW_SOURCE_TYPE
    version = CSV_ROW_SOURCE_VERSION
    input_class = CSVRowInput
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            definition_version=CSV_ROW_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowEvidenceProjection:
    """Project one row into the standard ``powercontext.text-evidence`` shape."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_TYPE,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    adapter=_CSVRowAdapter(),
    version=CSV_ROW_SOURCE_VERSION,
    projections=(_CSVRowEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_EVIDENCE_NAME",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_TYPE",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
