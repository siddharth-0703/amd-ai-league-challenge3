"""Defensive document parsers for Mini-Challenge 3.

Handles:
- .pdf (PyMuPDF / pypdf, catching encrypted PDFs)
- .docx (wordprocessingML paragraphs & tables)
- .xlsx (spreadsheetML worksheets, inlineStr & sharedStrings)
- .csv (tabular rows)
- .txt, .log (text lines)
- .py (code files, comments, constants)
- .png, .jpg (OCR / visual text extraction)
- Gracefully ignores unsupported / corrupt / unreadable files.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger("parsers")


def parse_text_or_code(path: Path) -> str:
    """Read plaintext or code file defensively."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Error reading {path}: {e}")
        return ""


def parse_csv(path: Path) -> str:
    """Parse CSV files into structured text lines."""
    lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row or not any(row):
                    continue
                if header is None:
                    header = [h.strip() for h in row]
                    lines.append("Columns: " + ", ".join(header))
                else:
                    row_pairs = [f"{h}: {val.strip()}" for h, val in zip(header, row) if val.strip()]
                    lines.append(" | ".join(row_pairs))
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Error parsing CSV {path}: {e}")
        return ""


def parse_docx(path: Path) -> str:
    """Parse DOCX paragraphs and tables without external dependencies."""
    lines = []
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")
        tree = ET.fromstring(xml_content)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        body = tree.find('.//w:body', ns)
        if body is None:
            return ""

        for child in body:
            tag = child.tag.split('}')[-1]
            if tag == 'p':
                texts = [node.text for node in child.iterfind('.//w:t', ns) if node.text]
                p_text = "".join(texts).strip()
                if p_text:
                    lines.append(p_text)
            elif tag == 'tbl':
                lines.append("\n[Table]")
                for row in child.iterfind('.//w:tr', ns):
                    row_vals = []
                    for cell in row.iterfind('.//w:tc', ns):
                        c_texts = [node.text for node in cell.iterfind('.//w:t', ns) if node.text]
                        row_vals.append("".join(c_texts).strip())
                    if any(row_vals):
                        lines.append(" | ".join(row_vals))
                lines.append("")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Error parsing DOCX {path}: {e}")
        return ""


def parse_xlsx(path: Path) -> str:
    """Parse XLSX worksheets, handling inlineStr and sharedStrings without external deps."""
    lines = []
    try:
        with zipfile.ZipFile(path) as z:
            shared_strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                ss_tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                for si in ss_tree.iterfind('.//main:si', ns):
                    text = "".join(t.text for t in si.iterfind('.//main:t', ns) if t.text)
                    shared_strings.append(text)

            sheet_names = {}
            if "xl/workbook.xml" in z.namelist():
                wb_tree = ET.fromstring(z.read("xl/workbook.xml"))
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                for idx, sheet in enumerate(wb_tree.iterfind('.//main:sheet', ns), start=1):
                    sheet_names[f"sheet{idx}.xml"] = sheet.get("name", f"Sheet{idx}")

            sheet_files = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
            for sf in sheet_files:
                file_name = sf.split("/")[-1]
                sname = sheet_names.get(file_name, file_name)
                lines.append(f"\n[Sheet: {sname}]")
                stree = ET.fromstring(z.read(sf))
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                header = None
                for row in stree.iterfind('.//main:row', ns):
                    row_vals = []
                    for c in row.iterfind('.//main:c', ns):
                        t = c.get('t')
                        val_str = ""
                        if t == 'inlineStr':
                            t_elem = c.find('.//main:t', ns)
                            if t_elem is not None and t_elem.text:
                                val_str = t_elem.text
                        elif t == 's':
                            v = c.find('main:v', ns)
                            if v is not None and v.text:
                                idx = int(v.text)
                                val_str = shared_strings[idx] if idx < len(shared_strings) else v.text
                        else:
                            v = c.find('main:v', ns)
                            if v is not None and v.text:
                                val_str = v.text
                        row_vals.append(val_str.strip())
                    if any(row_vals):
                        if header is None:
                            header = row_vals
                            lines.append("Columns: " + " | ".join(header))
                        else:
                            row_pairs = [f"{h}: {v}" for h, v in zip(header, row_vals) if v]
                            lines.append(" | ".join(row_pairs))
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Error parsing XLSX {path}: {e}")
        return ""


def parse_pdf(path: Path) -> str:
    """Parse PDF documents, returning empty string if encrypted or unreadable."""
    try:
        import pymupdf
        doc = pymupdf.open(path)
        if doc.is_encrypted:
            logger.info(f"Skipping encrypted PDF: {path}")
            return ""
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        if reader.is_encrypted:
            logger.info(f"Skipping encrypted PDF: {path}")
            return ""
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"Error reading PDF {path}: {e}")
        return ""


def parse_image(path: Path) -> str:
    """Parse image via tesseract CLI or pytesseract or basic fallback."""
    # Try tesseract binary directly if available
    try:
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "--oem", "1", "-l", "eng"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass

    # Try pytesseract if installed
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        if text.strip():
            return text.strip()
    except Exception:
        pass

    return ""


def parse_file(path: Path) -> str:
    """Main dispatcher for parsing files based on extension."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".log", ".py"):
        return parse_text_or_code(path)
    elif suffix == ".csv":
        return parse_csv(path)
    elif suffix == ".docx":
        return parse_docx(path)
    elif suffix == ".xlsx":
        return parse_xlsx(path)
    elif suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in (".png", ".jpg", ".jpeg"):
        return parse_image(path)
    else:
        # Unknown binary / unsupported format (.dat, etc.) - skip safely
        return ""
