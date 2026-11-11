import asyncio

import pytest

from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.models import SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def test_parse_csv_document():
    document = CSVDocumentInput(
        document_id="students.csv",
        primary_key="id",
        content=b"id,name,score\n1,Alice,90\n2,Bob,85\n",
    )

    rows = parse_csv_document(document)

    assert len(rows) == 2
    assert rows[0].primary_key_value == "1"
    assert rows[1].primary_key_value == "2"
    assert rows[0].columns == ("id", "name", "score")
    assert rows[0].values == ("1", "Alice", "90")


def test_csv_row_source_id():
    assert csv_row_source_id("students.csv", "1") == "csv:students.csv:1"
    assert csv_row_source_id("students.csv", "2") != csv_row_source_id(
        "students.csv", "1"
    )


def test_csv_source_and_text_evidence():
    asyncio.run(_check_csv_source_and_text_evidence())


async def _check_csv_source_and_text_evidence():
    document = CSVDocumentInput(
        document_id="students.csv",
        primary_key="id",
        content=b"id,name,score\n1,Alice,90\n2,Bob,85\n",
    )

    rows = parse_csv_document(document)
    definition = CSV_ROW_SOURCE_DEFINITION

    for row in rows:
        source = await definition.resolve(row)

        assert source.materialization == SourceMaterialization.CAPTURED
        assert source.source_id == csv_row_source_id(
            row.document_id, row.primary_key_value
        )

        original = await definition.read(source)
        assert original["id"] == row.primary_key_value

        projection = definition.projections[0].project(source)

        assert definition.projections[0].name == TEXT_EVIDENCE_PROJECTION_KEY.name
        assert projection.source_id == source.source_id
        assert projection.source_type == "csv-row"
        assert projection.metadata["document_id"] == "students.csv"
        assert projection.metadata["row"] == original


def test_duplicate_primary_key_rejected():
    document = CSVDocumentInput(
        document_id="students.csv",
        primary_key="id",
        content=b"id,name\n1,Alice\n1,Bob\n",
    )

    with pytest.raises(ValueError, match="duplicate primary-key"):
        parse_csv_document(document)


def test_missing_primary_key_column_rejected():
    document = CSVDocumentInput(
        document_id="students.csv",
        primary_key="missing",
        content=b"id,name\n1,Alice\n",
    )

    with pytest.raises(ValueError, match="primary_key"):
        parse_csv_document(document)
