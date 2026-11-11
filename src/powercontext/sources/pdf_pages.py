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
    "a an the of to in on at by for from with and or as is are was were "
    "be been being do does did how what which who when where why much many "
    "its it their this that these those company corporation inc year fiscal "
    "during between percent percentage calculate calculated amount total "
    "answer respond response question assuming perspective investment analyst "
    "only use using shown details give relying define directly explicitly "
    "outlined then state any has have had important ongoing reported report "
    "unadjusted usd million billion fy".split()
)


def _pdf_terms(text):
    terms = []
    for word in re.findall(r'[a-z]+|\d+', text.lower()):
        if word in _PDF_STOP:
            continue
        if len(word) > 4 and word.endswith('ies'):
            word = word[:-3] + 'y'
        elif len(word) > 4 and word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
            word = word[:-1]
        terms.append(word)
    return terms


def _pdf_rank_pages(pages, question, seed_rank=None):
    """Rank live pages using only their text and the caller's question."""
    from collections import Counter
    import math

    compact = re.sub(r'[^a-z0-9]', '', question.lower())
    names = {
        doc: re.sub(r'[^a-z0-9]', '', re.split(r'(?:19|20)\d{2}', doc.lower())[0])
        for _, doc, _ in pages
    }
    matched = {
        doc for doc, name in names.items()
        if len(name) >= 2 and name in compact
    }
    pool = [
        (record, doc, text, _pdf_terms(text))
        for record, doc, text in pages
    ]
    corpus = [row for row in pool if not matched or row[1] in matched]
    frequency = Counter()
    for _, _, _, tokens in corpus:
        frequency.update(set(tokens))
    count = len(corpus)
    average = sum(len(row[3]) for row in corpus) / max(1, count) or 1

    weights = {
        term: 1.0 for term in _pdf_terms(question)
        if not term.isdigit()
    }
    for name in names.values():
        if name in weights:
            weights[name] = 0.05

    groups = (
        ('geography geographic geographical geographies region regions countries country international america europe asia africa', 0.55),
        ('legal battle litigation lawsuit lawsuits court proceedings proceeding contingency contingencies', 0.70),
        ('revenue revenues sale sales turnover', 0.45),
        ('income profit profits earning earnings', 0.45),
        ('capex capital expenditure expenditures investing', 0.55),
        ('depreciation amortization', 0.55),
        ('repurchase buyback repurchased', 0.50),
        ('dividend distribution dividends', 0.45),
        ('restructuring reorganization severance', 0.50),
        ('debt borrowing borrowings loan loans', 0.45),
    )
    for words, weight in groups:
        synonyms = set(_pdf_terms(words))
        if synonyms.intersection(weights):
            for word in synonyms:
                weights.setdefault(word, weight)

    q = question.lower()
    sections = []
    if re.search(r'cash\s+flow|capex|ebitda|d\s*&\s*a|depreciation|amortization|dividend|repurchase|buyback', q):
        sections.append(r'(?:consolidated\s+)?statements?\s+of\s+cash\s+flows?')
    if re.search(r'income|earning|profit|ebitda|p\s*&\s*l|restructur|margin|revenue|\bsales\b|\beps\b', q):
        sections.append(r'(?:consolidated\s+)?(?:statements?\s+of\s+(?:income|operations|earnings)|income\s+statements?)')
    if re.search(r'balance\s+sheet|asset|liabilit|equity|working\s+capital|current\s+ratio|leverage|liquidity|debt|\bro[ae]\b', q):
        sections.append(r'(?:consolidated\s+)?balance\s+sheets?')
    if re.search(r'geograph|countr|region|where|primarily\s+operat|segment', q):
        sections.append(r'(?:item\s*1[.\s]+business|our\s+(?:divisions|business|operations)|(?:geographic|segment)\s+information)')
    if re.search(r'legal|litigat|lawsuit|court|battl', q):
        sections.append(r'(?:item\s*3[.\s]+)?legal\s+proceedings|(?:commitments?\s+and\s+)?contingencies')

    ranked = []
    for record, doc, text, tokens in pool:
        counts = Counter(tokens)
        norm = 1.2 * (0.25 + 0.75 * len(tokens) / average)
        score = 0.0
        for term, weight in weights.items():
            tf = counts[term]
            if tf:
                df = min(count, frequency[term])
                idf = math.log1p((count - df + 0.5) / (df + 0.5))
                score += weight * idf * tf * 2.2 / (tf + norm)

        section = any(
            re.search(
                r'(?m)^\s*(?:' + pattern +
                r')[. \t]*(?:\(continued\))?[. \t]*$',
                text.lower(),
            )
            for pattern in sections
        )
        if section:
            score += 14.0
        if matched and doc not in matched:
            score *= 0.08
        key = (record.entry.entry_id, record.entry.entry_version_id)
        if seed_rank and key in seed_rank:
            score += 0.05 / (1 + seed_rank[key])
        ranked.append((score, record, doc, text, section))

    ranked.sort(key=lambda row: (-row[0], row[1].entry.entry_id))
    return ranked



_STATEMENT_KINDS = (
    ("income", r"consolidated\s+(?:statements?\s+of\s+(?:operations|income|earnings)|income\s+statements?)"),
    ("balance", r"consolidated\s+balance\s+sheets?"),
    ("cash", r"consolidated\s+statements?\s+of\s+cash\s+flows?"),
    ("summary", r"(?:table\s+\d+[.:]?\s*)?summary\s+of\s+(?:financial\s+performance|financial\s+condition)"),
    ("segment", r"(?:summary\s+of\s+)?(?:business\s+)?segment\s+(?:data|information)"),
)

_NUMERIC_QUESTION = re.compile(
    r"how\s+much|how\s+many|rate|ratio|margin|percent|amount|cost|revenue|sales|"
    r"profit|income|tax|ebitda|capex|expense|spend|price|value|balance|debt|"
    r"dividend|buyback|repurchase",
    re.I,
)

_CONTINUED = re.compile(r"\s*\(continued\)\s*$")


def _statement_kinds(text):
    """Return statement kinds whose title is a heading line on this page."""
    lines = [
        line.strip().lower().rstrip(".")
        for line in text.splitlines()[:6]
        if line.strip()
    ]
    kinds = []
    for kind, pattern in _STATEMENT_KINDS:
        for line in lines:
            if re.fullmatch(pattern, _CONTINUED.sub("", line)):
                kinds.append(kind)
                break
    return kinds


async def search_pdf_pages(
    memory, question: str, limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search scoped Memory and rerank genuine page citations offline."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must not be blank')
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('limit must be a positive integer')

    query = ' '.join(dict.fromkeys(_pdf_terms(question))) or question.strip()
    seed = await memory.search(SearchMemoryRequest(
        query=query, limit=max(40, min(100, limit * 4)), mode='fts',
    ))
    seed_rank = {
        (hit.entry_id, hit.entry_version_id): rank
        for rank, hit in enumerate(seed.hits)
    }
    snapshot = await memory.list()
    pages, seen = [], set()
    for record in snapshot.entries:
        if record.state != 'active':
            continue
        refs = record.entry.sources
        if len(refs) != 1 or refs[0].source_type != 'pdf-page':
            continue
        if refs[0].source_id in seen:
            continue
        seen.add(refs[0].source_id)
        doc, _, text = record.entry.text.partition('\n')
        pages.append((record, doc, text))
    if not pages:
        return ()

    ranked = _pdf_rank_pages(pages, question, seed_rank)
    head = list(ranked[:12])

    # Financial questions often need the primary statements, which lexical
    # ranking tends to bury under notes. Keep the best page of each kind.
    if _NUMERIC_QUESTION.search(question) and limit > 12:
        near_docs = {row[2] for row in head[:6]}
        picked = {}
        for row in ranked:
            if row[2] not in near_docs:
                continue
            for kind in _statement_kinds(row[3]):
                picked.setdefault(kind, row)
        for row in picked.values():
            if len(head) >= min(limit, 18):
                break
            if any(row[1] is item[1] for item in head):
                continue
            head.append(row)

    chosen = {id(row[1]) for row in head}
    for row in ranked:
        if len(head) >= limit:
            break
        if id(row[1]) in chosen:
            continue
        head.append(row)
        chosen.add(id(row[1]))

    citations = []
    for row in head[:limit]:
        citation = row[1].citation
        await memory.get(GetMemoryEntryRequest(citation=citation))
        citations.append(citation)
    return tuple(citations)
