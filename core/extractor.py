import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import httpx

@dataclass
class PageContent:
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentExtract:
    document_id: str
    title: str
    source_type: str  # "pdf", "url", "text"
    total_pages: int
    pages: List[PageContent]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentExtractor:
    """Extracts text and structural metadata from PDFs, Web URLs, and Raw text."""
    
    @staticmethod
    def extract_from_pdf(file_path: Path, doc_id: Optional[str] = None) -> DocumentExtract:
        pages: List[PageContent] = []
        full_text_parts: List[str] = []
        title = file_path.stem
        metadata = {
            "source": str(file_path.name),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
        }

        # Primary extraction using PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            metadata["pdf_meta"] = doc.metadata or {}
            if doc.metadata and doc.metadata.get("title"):
                title = doc.metadata["title"].strip() or title
                
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text") or ""
                # Clean up null characters or formatting quirks
                text = text.replace("\x00", "")
                page_obj = PageContent(
                    page_number=page_index + 1,
                    text=text.strip(),
                    metadata={"page": page_index + 1, "char_count": len(text)}
                )
                pages.append(page_obj)
                if page_obj.text:
                    full_text_parts.append(page_obj.text)
            doc.close()
        except ImportError:
            # Fallback extraction using pypdf
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                page_obj = PageContent(
                    page_number=page_index + 1,
                    text=text.strip(),
                    metadata={"page": page_index + 1, "char_count": len(text)}
                )
                pages.append(page_obj)
                if page_obj.text:
                    full_text_parts.append(page_obj.text)
                    
        full_text = "\n\n".join(full_text_parts)
        doc_id = doc_id or f"doc_{abs(hash(str(file_path)))}"

        return DocumentExtract(
            document_id=doc_id,
            title=title,
            source_type="pdf",
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            metadata=metadata
        )

    @staticmethod
    async def extract_from_url(url: str, doc_id: Optional[str] = None) -> DocumentExtract:
        """Fetch webpage HTML, strip tags, and extract clean text content."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LearningAssistantBot/1.0"
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        # Extract title
        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url

        # Remove scripts, styles, heads, navs, footers
        clean = re.sub(r"<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        # Strip remaining tags
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Decode common HTML entities
        clean = re.sub(r"&nbsp;", " ", clean)
        clean = re.sub(r"&amp;", "&", clean)
        clean = re.sub(r"&lt;", "<", clean)
        clean = re.sub(r"&gt;", ">", clean)
        clean = re.sub(r"&quot;", '"', clean)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()

        doc_id = doc_id or f"url_{abs(hash(url))}"
        page_obj = PageContent(page_number=1, text=clean, metadata={"source_url": url})

        return DocumentExtract(
            document_id=doc_id,
            title=title,
            source_type="url",
            total_pages=1,
            pages=[page_obj],
            full_text=clean,
            metadata={"url": url}
        )

    @staticmethod
    def extract_from_raw_text(text: str, title: str = "Pasted Text", doc_id: Optional[str] = None) -> DocumentExtract:
        doc_id = doc_id or f"txt_{abs(hash(text[:100]))}"
        page_obj = PageContent(page_number=1, text=text.strip(), metadata={"source": "manual_entry"})
        return DocumentExtract(
            document_id=doc_id,
            title=title,
            source_type="text",
            total_pages=1,
            pages=[page_obj],
            full_text=text.strip(),
            metadata={"source": "text_input"}
        )
