"""PDF page parsing, stable page Sources, and exact Memory citations."""

from __future__ import annotations

import hashlib
import io
import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.runtime import (
    GetMemoryEntryRequest,
    SearchMemoryRequest,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    TextEvidence,
)


class PDFDocumentInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")
        if value != value.strip():
            raise ValueError("document_id must be trimmed")
        return value


class PDFPageInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("invalid document_id")
        return value

    @field_validator("document_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid document SHA256")
        return value


class PDFPageSource(Source):
    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str


def parse_pdf_document(
    value: PDFDocumentInput,
) -> tuple[PDFPageInput, ...]:
    """Extract every physical PDF page atomically."""

    raw = value.content

    if not raw:
        raise ValueError("PDF content must not be empty")

    if b"%PDF-" not in raw[:1024]:
        raise ValueError("input is not a PDF document")

    digest = hashlib.sha256(raw).hexdigest()

    try:
        reader = PdfReader(io.BytesIO(raw), strict=False)

        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise ValueError("PDF requires a non-empty password")

        pages = []

        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                PDFPageInput(
                    document_id=value.document_id,
                    document_sha256=digest,
                    page_number=index,
                    text=text,
                )
            )

        if not pages:
            raise ValueError("PDF contains no physical pages")

        return tuple(pages)

    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF parsing failed") from error


class PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class PDFPageSourceAdapter:
    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(
        self,
        value: PDFPageInput,
        /,
    ) -> PDFPageSource:
        identity = json.dumps(
            [
                value.document_id,
                value.document_sha256,
                value.page_number,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        identity_hash = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        return PDFPageSource(
            name=f"pdf-page:{identity_hash}",
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(
        self,
        source: PDFPageSource,
        /,
    ) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()

PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=PDF_PAGE_SOURCE_ADAPTER,
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)



_PDF_STOP = set(
    "a an the of to in on at by for from with and or as is are was "
    "were be been being do does did how what which who when where why "
    "much many its it their this that these those company corporation "
    "inc year fiscal during between percent percentage calculate "
    "calculated amount total".split()
)


def _pdf_terms(text):
    words = re.findall(r"[a-z]+|\d+", text.lower())
    return [
        word[:-1]
        if len(word) > 4 and word.endswith("s")
        and not word.endswith(("ss", "us", "is"))
        else word
        for word in words if word not in _PDF_STOP
    ]


async def search_pdf_pages(
    memory, question: str, limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Rank live scoped PDF pages and return exact persisted citations."""
    from collections import Counter
    import math

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be blank")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")

    terms = _pdf_terms(question)
    query = " ".join(dict.fromkeys(terms)) or question.strip()
    seed = await memory.search(SearchMemoryRequest(
        query=query,
        limit=min(100, max(40, limit * 4)),
        mode="fts",
    ))
    seed_rank = {
        (hit.entry_id, hit.entry_version_id): rank
        for rank, hit in enumerate(seed.hits)
    }

    # Read only live entries through the public scoped Memory API.
    snapshot = await memory.list()
    pages = []
    seen = set()
    for record in snapshot.entries:
        if record.state != "active":
            continue
        entry = record.entry
        if len(entry.sources) != 1:
            continue
        source = entry.sources[0]
        if source.source_type != "pdf-page" or source.source_id in seen:
            continue
        seen.add(source.source_id)
        document_id, _, text = entry.text.partition("\n")
        tokens = _pdf_terms(text)
        pages.append((record, document_id, tokens, Counter(tokens)))

    if not pages:
        return ()

    count = len(pages)
    average = sum(len(tokens) for _, _, tokens, _ in pages) / count or 1
    frequency = Counter()
    for _, _, _, counts in pages:
        frequency.update(counts.keys())

    weights = {term: 1.0 for term in terms}
    groups = (
        ("revenue", "sale"),
        ("profit", "income", "earning"),
        ("capex", "capital", "expenditure"),
        ("buyback", "repurchase"),
        ("dividend", "distribution"),
        ("depreciation", "amortization"),
    )
    for group in groups:
        if set(group).intersection(weights):
            for word in group:
                weights.setdefault(word, 0.22)

    compact_question = re.sub(r"[^a-z0-9]", "", question.lower())
    document_names = {}
    for _, doc, _, _ in pages:
        name = re.split(r"(?:19|20)\d{2}", doc.lower(), maxsplit=1)[0]
        document_names[doc] = re.sub(r"[^a-z0-9]", "", name)
    matched_docs = {
        doc for doc, name in document_names.items()
        if len(name) >= 2 and name in compact_question
    }
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    phrase_terms = [term for term in terms if not term.isdigit()]
    phrases = list(zip(phrase_terms, phrase_terms[1:]))

    ranked = []
    for record, doc, tokens, counts in pages:
        norm = 1.2 * (0.25 + 0.75 * len(tokens) / average)
        score = 0.0
        covered = 0
        for term, weight in weights.items():
            tf = counts[term]
            if not tf:
                continue
            idf = math.log1p(
                (count - frequency[term] + 0.5) /
                (frequency[term] + 0.5)
            )
            score += weight * idf * tf * 2.2 / (tf + norm)
            covered += term in terms
        pairs = set(zip(tokens, tokens[1:]))
        score += 0.3 * sum(pair in pairs for pair in phrases)
        score *= 1 + 0.3 * covered / max(1, len(set(terms)))
        if matched_docs:
            score *= 2.5 if doc in matched_docs else 0.15
        if years.intersection(re.findall(r"(?:19|20)\d{2}", doc)):
            score *= 1.15
        key = (record.entry.entry_id, record.entry.entry_version_id)
        if key in seed_rank:
            score += 0.05 / (1 + seed_rank[key])
        ranked.append((score, record))

    ranked.sort(key=lambda item: (-item[0], item[1].entry.entry_id))
    citations = []
    for _, record in ranked[:limit]:
        citation = record.citation
        await memory.get(GetMemoryEntryRequest(citation=citation))
        citations.append(citation)
    return tuple(citations)
