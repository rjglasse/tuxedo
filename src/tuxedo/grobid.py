"""Grobid client for PDF extraction."""

import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

from tuxedo.logging import get_logger
from tuxedo.models import Author, Paper


# Default retry configurations - different parameter combinations to try
DEFAULT_RETRY_CONFIGS = [
    # First retry: disable header consolidation (less aggressive parsing)
    {"consolidateHeader": "0", "consolidateCitations": "0"},
    # Second retry: minimal config
    {"consolidateHeader": "0", "consolidateCitations": "0", "teiCoordinates": "false"},
]

# TEI namespace
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidError(Exception):
    """Base exception for Grobid-related errors."""

    pass


class GrobidConnectionError(GrobidError):
    """Raised when cannot connect to Grobid service."""

    def __init__(self, url: str, cause: Exception | None = None):
        self.url = url
        self.cause = cause
        message = f"Cannot connect to Grobid at {url}"
        if cause:
            message += f": {cause}"
        super().__init__(message)


class GrobidProcessingError(GrobidError):
    """Raised when Grobid fails to process a PDF."""

    def __init__(
        self,
        pdf_path: Path,
        status_code: int | None = None,
        response_body: str | None = None,
        cause: Exception | None = None,
    ):
        self.pdf_path = pdf_path
        self.status_code = status_code
        self.response_body = response_body
        self.cause = cause

        if status_code:
            message = f"Failed to process {pdf_path.name}: HTTP {status_code}"
            if response_body:
                # Truncate long responses
                body_preview = (
                    response_body[:200] + "..." if len(response_body) > 200 else response_body
                )
                message += f" - {body_preview}"
        elif cause:
            message = f"Failed to process {pdf_path.name}: {cause}"
        else:
            message = f"Failed to process {pdf_path.name}"

        super().__init__(message)


class GrobidParsingError(GrobidError):
    """Raised when TEI XML parsing fails."""

    def __init__(self, pdf_path: Path, cause: Exception | None = None):
        self.pdf_path = pdf_path
        self.cause = cause
        message = f"Failed to parse Grobid output for {pdf_path.name}"
        if cause:
            message += f": {cause}"
        super().__init__(message)


@dataclass
class ProcessingResult:
    """Result of processing a single PDF."""

    pdf_path: Path
    paper: Paper | None = None
    error: GrobidError | None = None
    attempts: int = 1  # Number of attempts made
    retried: bool = False  # Whether this succeeded on a retry

    @property
    def success(self) -> bool:
        return self.paper is not None


class GrobidClient:
    """Client for Grobid PDF extraction service."""

    def __init__(self, base_url: str = "http://localhost:8070"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)
        self.log = get_logger("grobid")

    def is_alive(self) -> bool:
        """Check if Grobid service is available."""
        try:
            resp = self.client.get(f"{self.base_url}/api/isalive")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    def check_connection(self) -> None:
        """Check connection to Grobid and raise detailed error if unavailable."""
        try:
            resp = self.client.get(f"{self.base_url}/api/isalive")
            if resp.status_code != 200:
                raise GrobidConnectionError(
                    self.base_url,
                    cause=Exception(f"Service returned HTTP {resp.status_code}"),
                )
        except httpx.RequestError as e:
            raise GrobidConnectionError(self.base_url, cause=e) from e

    def process_pdf(
        self,
        pdf_path: Path,
        max_retries: int = 2,
        retry_configs: list[dict] | None = None,
    ) -> Paper:
        """Process a PDF and extract structured content.

        Args:
            pdf_path: Path to the PDF file
            max_retries: Maximum number of retry attempts (default: 2)
            retry_configs: List of alternative parameter configs for retries.
                          If None, uses DEFAULT_RETRY_CONFIGS.

        Raises:
            GrobidProcessingError: If Grobid fails to process the PDF after all retries
            GrobidParsingError: If the TEI XML response cannot be parsed
            GrobidConnectionError: If cannot connect to Grobid
        """
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
        except OSError as e:
            raise GrobidProcessingError(pdf_path, cause=e) from e

        # Generate stable ID from file content
        paper_id = hashlib.sha256(pdf_content).hexdigest()[:12]

        # Default config for first attempt
        default_config = {"consolidateHeader": "1", "consolidateCitations": "0"}

        # Build list of configs to try
        configs_to_try = [default_config]
        if max_retries > 0:
            retry_list = retry_configs if retry_configs is not None else DEFAULT_RETRY_CONFIGS
            configs_to_try.extend(retry_list[:max_retries])

        last_error: GrobidProcessingError | None = None
        self.log.info(f"Processing PDF: {pdf_path.name} ({len(pdf_content)} bytes)")

        for attempt, config in enumerate(configs_to_try):
            try:
                start_time = time.time()
                resp = self._make_grobid_request(pdf_path, pdf_content, config)
                elapsed = time.time() - start_time

                if resp.status_code == 200:
                    self.log.info(f"Grobid response: {pdf_path.name} in {elapsed:.2f}s")
                    try:
                        paper = self._parse_tei(resp.text, paper_id, pdf_path)
                        self.log.debug(
                            f"Parsed: {paper.title[:50] if paper.title else 'No title'}..."
                        )
                        return paper
                    except ET.ParseError as e:
                        # Parsing errors are not retryable with different configs
                        self.log.error(f"TEI parsing failed: {pdf_path.name}: {e}")
                        raise GrobidParsingError(pdf_path, cause=e) from e

                # Non-200 response - save error and maybe retry
                last_error = GrobidProcessingError(
                    pdf_path,
                    status_code=resp.status_code,
                    response_body=resp.text,
                )

                # Only retry on server errors (5xx) or specific client errors
                if resp.status_code < 500 and resp.status_code != 400:
                    self.log.error(f"Grobid error: {pdf_path.name} HTTP {resp.status_code}")
                    raise last_error

                self.log.warning(
                    f"Grobid HTTP {resp.status_code}: {pdf_path.name}, attempt {attempt + 1}/{len(configs_to_try)}"
                )

            except httpx.RequestError as e:
                self.log.error(f"Grobid connection error: {pdf_path.name}: {e}")
                raise GrobidConnectionError(self.base_url, cause=e) from e

            # Wait before retry (exponential backoff: 1s, 2s, 4s...)
            if attempt < len(configs_to_try) - 1:
                wait_time = 2**attempt
                self.log.debug(f"Retrying {pdf_path.name} in {wait_time}s...")
                time.sleep(wait_time)

        # All retries exhausted
        self.log.error(f"All retries exhausted for {pdf_path.name}")
        if last_error:
            raise last_error
        raise GrobidProcessingError(pdf_path)

    def _make_grobid_request(
        self,
        pdf_path: Path,
        pdf_content: bytes,
        config: dict,
    ) -> httpx.Response:
        """Make a request to Grobid with the given configuration."""
        return self.client.post(
            f"{self.base_url}/api/processFulltextDocument",
            files={"input": (pdf_path.name, pdf_content, "application/pdf")},
            data=config,
        )

    def process_pdf_with_result(
        self,
        pdf_path: Path,
        max_retries: int = 2,
    ) -> ProcessingResult:
        """Process a PDF and return a ProcessingResult with retry information.

        This is a convenience method that wraps process_pdf and returns
        a ProcessingResult object that includes information about whether
        the result came from a retry attempt.
        """
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
        except OSError as e:
            return ProcessingResult(
                pdf_path=pdf_path,
                error=GrobidProcessingError(pdf_path, cause=e),
                attempts=1,
            )

        paper_id = hashlib.sha256(pdf_content).hexdigest()[:12]
        default_config = {"consolidateHeader": "1", "consolidateCitations": "0"}

        configs_to_try = [default_config]
        if max_retries > 0:
            configs_to_try.extend(DEFAULT_RETRY_CONFIGS[:max_retries])

        last_error: GrobidError | None = None

        for attempt, config in enumerate(configs_to_try):
            try:
                resp = self._make_grobid_request(pdf_path, pdf_content, config)

                if resp.status_code == 200:
                    try:
                        paper = self._parse_tei(resp.text, paper_id, pdf_path)
                        return ProcessingResult(
                            pdf_path=pdf_path,
                            paper=paper,
                            attempts=attempt + 1,
                            retried=attempt > 0,
                        )
                    except ET.ParseError as e:
                        return ProcessingResult(
                            pdf_path=pdf_path,
                            error=GrobidParsingError(pdf_path, cause=e),
                            attempts=attempt + 1,
                        )

                last_error = GrobidProcessingError(
                    pdf_path,
                    status_code=resp.status_code,
                    response_body=resp.text,
                )

                if resp.status_code < 500 and resp.status_code != 400:
                    return ProcessingResult(
                        pdf_path=pdf_path,
                        error=last_error,
                        attempts=attempt + 1,
                    )

            except httpx.RequestError as e:
                return ProcessingResult(
                    pdf_path=pdf_path,
                    error=GrobidConnectionError(self.base_url, cause=e),
                    attempts=attempt + 1,
                )

            if attempt < len(configs_to_try) - 1:
                time.sleep(2**attempt)

        return ProcessingResult(
            pdf_path=pdf_path,
            error=last_error or GrobidProcessingError(pdf_path),
            attempts=len(configs_to_try),
        )

    def _parse_tei(self, tei_xml: str, paper_id: str, pdf_path: Path) -> Paper:
        """Parse TEI XML into Paper model."""
        root = ET.fromstring(tei_xml)

        # Extract title
        title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        title = (
            title_elem.text.strip() if title_elem is not None and title_elem.text else pdf_path.stem
        )

        # Extract authors
        authors = []
        for author_elem in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
            persname = author_elem.find("tei:persName", TEI_NS)
            if persname is not None:
                forename = persname.find("tei:forename", TEI_NS)
                surname = persname.find("tei:surname", TEI_NS)
                name_parts = []
                if forename is not None and forename.text:
                    name_parts.append(forename.text)
                if surname is not None and surname.text:
                    name_parts.append(surname.text)
                if name_parts:
                    affil_elem = author_elem.find(".//tei:affiliation/tei:orgName", TEI_NS)
                    affil = affil_elem.text if affil_elem is not None and affil_elem.text else None
                    authors.append(Author(name=" ".join(name_parts), affiliation=affil))

        # Extract abstract
        abstract_elem = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        abstract = None
        if abstract_elem is not None:
            abstract_parts = []
            for p in abstract_elem.findall(".//tei:p", TEI_NS):
                if p.text:
                    abstract_parts.append(p.text.strip())
            # Also try direct text content
            if not abstract_parts:
                abstract_text = "".join(abstract_elem.itertext()).strip()
                if abstract_text:
                    abstract_parts.append(abstract_text)
            abstract = " ".join(abstract_parts) if abstract_parts else None

        # Extract year
        year = None
        date_elem = root.find(".//tei:sourceDesc//tei:date[@when]", TEI_NS)
        if date_elem is not None:
            when = date_elem.get("when", "")
            if when and len(when) >= 4:
                try:
                    year = int(when[:4])
                except ValueError:
                    pass

        # Extract DOI
        doi = None
        for idno in root.findall(".//tei:sourceDesc//tei:idno[@type='DOI']", TEI_NS):
            if idno.text:
                doi = idno.text.strip()
                break

        # Extract publisher
        publisher = None
        publisher_elem = root.find(".//tei:publicationStmt/tei:publisher", TEI_NS)
        if publisher_elem is not None and publisher_elem.text:
            publisher = publisher_elem.text.strip()

        # Extract journal or booktitle from monogr
        journal = None
        booktitle = None
        monogr = root.find(".//tei:sourceDesc//tei:monogr", TEI_NS)
        if monogr is not None:
            monogr_title = monogr.find("tei:title", TEI_NS)
            if monogr_title is not None and monogr_title.text:
                title_text = monogr_title.text.strip()
                level = monogr_title.get("level", "")
                # level="j" indicates journal, level="m" indicates monograph/proceedings
                if level == "j":
                    journal = title_text
                elif (
                    level == "m"
                    or "proceedings" in title_text.lower()
                    or "conference" in title_text.lower()
                ):
                    booktitle = title_text
                else:
                    # Default to booktitle for conference papers
                    booktitle = title_text

        # Extract volume, number (issue), pages
        volume = None
        number = None
        pages = None

        volume_elem = root.find(".//tei:sourceDesc//tei:biblScope[@unit='volume']", TEI_NS)
        if volume_elem is not None and volume_elem.text:
            volume = volume_elem.text.strip()

        issue_elem = root.find(".//tei:sourceDesc//tei:biblScope[@unit='issue']", TEI_NS)
        if issue_elem is not None and issue_elem.text:
            number = issue_elem.text.strip()

        page_elem = root.find(".//tei:sourceDesc//tei:biblScope[@unit='page']", TEI_NS)
        if page_elem is not None:
            page_from = page_elem.get("from", "")
            page_to = page_elem.get("to", "")
            if page_from and page_to:
                pages = f"{page_from}--{page_to}" if page_from != page_to else page_from
            elif page_elem.text:
                pages = page_elem.text.strip()

        # Extract arXiv ID
        arxiv_id = None
        for idno in root.findall(".//tei:sourceDesc//tei:idno[@type='arXiv']", TEI_NS):
            if idno.text:
                # Clean up arXiv ID (e.g., "arXiv:2212.05113v1[cs.CY]" -> "2212.05113")
                arxiv_text = idno.text.strip()
                if arxiv_text.startswith("arXiv:"):
                    arxiv_text = arxiv_text[6:]
                # Remove version and category suffix
                arxiv_id = arxiv_text.split("v")[0].split("[")[0]
                break

        # Construct URL from DOI or arXiv
        url = None
        if doi:
            url = f"https://doi.org/{doi}"
        elif arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        # Extract sections
        sections = {}
        body = root.find(".//tei:body", TEI_NS)
        if body is not None:
            for div in body.findall("tei:div", TEI_NS):
                head = div.find("tei:head", TEI_NS)
                if head is not None and head.text:
                    section_name = head.text.strip().lower()
                    section_text = []
                    for p in div.findall("tei:p", TEI_NS):
                        text = "".join(p.itertext()).strip()
                        if text:
                            section_text.append(text)
                    if section_text:
                        sections[section_name] = " ".join(section_text)

        # Extract keywords
        keywords = []
        for term in root.findall(".//tei:profileDesc//tei:keywords//tei:term", TEI_NS):
            if term.text:
                keywords.append(term.text.strip())

        return Paper(
            id=paper_id,
            pdf_path=pdf_path,
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            sections=sections,
            keywords=keywords,
            journal=journal,
            booktitle=booktitle,
            publisher=publisher,
            volume=volume,
            number=number,
            pages=pages,
            arxiv_id=arxiv_id,
            url=url,
        )

    def process_directory(self, pdf_dir: Path) -> list[ProcessingResult]:
        """Process all PDFs in a directory.

        Returns a list of ProcessingResult objects, each containing either
        a Paper on success or a GrobidError on failure. This allows callers
        to handle errors appropriately rather than silently skipping failures.
        """
        results = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            try:
                paper = self.process_pdf(pdf_path)
                results.append(ProcessingResult(pdf_path=pdf_path, paper=paper))
            except GrobidError as e:
                results.append(ProcessingResult(pdf_path=pdf_path, error=e))
        return results

    def process_directory_papers(self, pdf_dir: Path) -> list[Paper]:
        """Process all PDFs in a directory, returning only successful papers.

        This is a convenience method that filters out failed results.
        Use process_directory() if you need to handle individual failures.
        """
        results = self.process_directory(pdf_dir)
        return [r.paper for r in results if r.paper is not None]

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
