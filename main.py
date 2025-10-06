#!/usr/bin/env python3
"""
Enhanced PDF Handwritten OCR v5.8
- Aggressive multi-pass cleaning with unicode and repetition filtering
- Robust answer extraction with fuzzy OCR corrections
- Clear metadata separation
"""

import os
import sys
import time
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import logging

from pdf2image import convert_from_path
from PIL import Image
import io
from google.oauth2 import service_account
from google.cloud import vision

# Suppress Google Cloud warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AdvancedTextCleaner:
    def __init__(self, max_passes: int = 3):
        self.max_passes = max_passes
        self.cleaning_stats = {}
        self.metadata_noise_patterns = [
            r"={3,}", r"-{3,}", r"OCR\s+PROCESSING\s+RESULTS?", r"TOTAL\s+PAGES?:\s*\d+",
            r"AVERAGE\s+CONFIDENCE:\s*\d+", r"SOURCE\s+FILE:.*?\.pdf", r"---\s*PAGE\s+\d+\s*---"
        ]
        self.institutional_patterns = [
            r"ALLEN", r"CAREER\s+INSTITUTS?", r"KOTA\s*\(PAJASTHAN", r"KOTA\s*\(RAJASTHAN", 
            r"MARKS", r"Q\.?\s*NO\.?", r"CANDIDATE\s+ANSWER"
        ]
        self.short_line_patterns = [
            r"^[A-Za-z0-9]{1,2}$", r"^[\W_]{1,2}$", r"^[A-Za-z]\s+[A-Za-z]$", r"^\s*[☐]\s*$"
        ]
        self.garbage_patterns = [
            r"^[Il1|]{1,3}$", r"^[O0]{1,3}$", r"^[+\-*_:]{1,3}$", r"^\s*I anA\s*$",
            r"^\s*☐\s*I.?anA\s*$", r"^\s*I\s*anA\s*$"
        ]
        
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        kept_lines = text.splitlines()
        total_removed = 0

        for pass_num in range(1, self.max_passes + 1):
            new_kept = []
            removed = 0
            for line in kept_lines:
                l = line.strip()
                if self._is_noise_line(l):
                    removed += 1
                    continue
                new_kept.append(line)
            kept_lines = new_kept
            total_removed += removed
            logger.info(f"Cleaning pass {pass_num} finished - lines removed: {removed}")

            if removed == 0:
                break

        cleaned_text = "\n".join(kept_lines)
        total_lines = len(text.splitlines())
        kept_lines_count = len(kept_lines)

        self.cleaning_stats = {
            "total_lines_start": total_lines,
            "total_lines_kept": kept_lines_count,
            "total_lines_removed": total_lines - kept_lines_count,
            "total_lines_removed_percentage": round((total_lines - kept_lines_count) / total_lines * 100, 2) if total_lines else 0,
            "passes_completed": pass_num,
        }
        return cleaned_text

    def _is_noise_line(self, line: str) -> bool:
        if not line:
            return True
        for p in self.metadata_noise_patterns:
            if re.search(p, line, re.IGNORECASE):
                return True
        for p in self.institutional_patterns:
            if re.search(p, line, re.IGNORECASE):
                return True
        for p in self.short_line_patterns:
            if re.match(p, line):
                return True
        for p in self.garbage_patterns:
            if re.match(p, line):
                return True

        # Remove lines with >50% non-ASCII chars (likely foreign/unreadable text)
        if (sum(1 for c in line if ord(c) > 127) / max(len(line), 1)) > 0.5:
            return True

        # Remove lines with 6 or more repetition of same char
        if re.search(r'(.)\1{5,}', line):
            return True

        # Remove lines with less than 20% alphanumeric characters
        if (sum(c.isalnum() for c in line) / max(len(line), 1)) < 0.2:
            return True

        return False


class RobustAnswerExtractor:
    def __init__(self):
        # Capture numbering as digit or single letter (fuzzy to handle OCR errors)
        self.pats = [
            r"(?:^|\n)\s*Ans\s*([0-9A-Za-z])\.?\s*(.*?)(?=(?:\n\s*Ans\s*[0-9A-Za-z])|\Z)",
            r"(?:^|\n)\s*([0-9A-Za-z])\.\s*(.*?)(?=(?:\n\s*[0-9A-Za-z]\.)|\Z)"
        ]
        # Sub-question patterns - lowercase letters and roman numerals
        self.subp = [
            r"\(([a-z])\)\s*(.*?)(?=(?:\([a-z]\))|\Z)",
            r"\(([ivxlc]+)\)\s*(.*?)(?=(?:\([ivxlc]+\))|\Z)"
        ]
        # Map common OCR misreads of digits to actual digits
        self.ocr_corrections = str.maketrans({
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', 'Z': '2', 'S': '5', 'B': '8', 'Y': '4'
        })

    def extract(self, text: str) -> Dict[str, Union[str, Dict[str, str]]]:
        out: Dict[str, Union[str, Dict[str, str]]] = {}
        for pat in self.pats:
            for num, body in re.findall(pat, text, flags=re.IGNORECASE | re.DOTALL):
                # Correct OCR misreads in the captured numbering
                num_clean = num.translate(self.ocr_corrections)
                if not num_clean.isdigit():
                    continue
                content = re.sub(r"\s+", " ", body).strip()
                if not content:
                    continue
                subs = self._subs(content)
                if subs:
                    out[num_clean] = subs
                else:
                    out[num_clean] = content
        return out

    def _subs(self, content: str) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for sp in self.subp:
            for key, val in re.findall(sp, content, flags=re.IGNORECASE | re.DOTALL):
                d[f"({key})"] = re.sub(r"\s+", " ", val).strip()
        return d


class EnhancedPDFOCR:
    def __init__(self, credentials_path: str):
        if not os.path.exists(credentials_path):
            raise FileNotFoundError("credentials.json missing")
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = vision.ImageAnnotatorClient(credentials=creds)
        self.cleaner = AdvancedTextCleaner()
        self.extractor = RobustAnswerExtractor()
        self.processing_meta = {}

    def pdf_to_images(self, path: str, dpi: int = 300) -> List[Image.Image]:
        pages = convert_from_path(path, dpi=dpi)
        fsize = os.stat(path).st_size
        self.processing_meta = {
            "source_file": path,
            "file_size_bytes": fsize,
            "file_size_mb": round(fsize / (1024 * 1024), 2),
            "total_pages": len(pages),
            "start_time": datetime.now().isoformat(),
        }
        return pages

    def image_to_text(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("L").save(buf, format="PNG")
        buf.seek(0)
        vimg = vision.Image(content=buf.getvalue())
        feat = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
        req = vision.AnnotateImageRequest(image=vimg, features=[feat])
        resp = self.client.annotate_image(request=req)
        return resp.full_text_annotation.text if resp.full_text_annotation else ""

    def process(self, pdf_path: str) -> Dict:
        start = time.time()
        pages = self.pdf_to_images(pdf_path)
        full_text = ""
        word_count = 0
        char_count = 0
        for idx, page in enumerate(pages, 1):
            txt = self.image_to_text(page)
            full_text += txt + "\n"
            word_count += len(txt.split())
            char_count += len(txt)
            time.sleep(0.5)
        cleaned = self.cleaner.clean(full_text)
        answers = self.extractor.extract(cleaned)
        self.processing_meta.update({
            "processing_time_seconds": round(time.time() - start, 2),
            "total_words_extracted": word_count,
            "total_characters_extracted": char_count,
            "total_answers_found": len(answers),
            "cleaning_stats": self.cleaner.cleaning_stats,
            "end_time": datetime.now().isoformat(),
        })
        return {"answers": answers, "metadata": self.processing_meta, "raw_text": cleaned}

    def save_results(self, results: Dict, output_prefix: Optional[str] = None) -> Tuple[str, str]:
        if "error" in results:
            return "", ""
        if not output_prefix:
            prefix = Path(results["metadata"]["source_file"]).stem
            output_prefix = f"{prefix}_processed"
        json_file = f"{output_prefix}.json"
        txt_file = f"{output_prefix}.txt"
        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump({"answers": results["answers"], "metadata": results["metadata"]}, jf, indent=2)
        with open(txt_file, "w", encoding="utf-8") as tf:
            tf.write(results["raw_text"])
        logger.info(f"Saved JSON: {json_file}; Text: {txt_file}")
        return json_file, txt_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <PDF_PATH>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    creds_path = "credentials.json"
    ocr = EnhancedPDFOCR(creds_path)
    results = ocr.process(pdf_path)
    ocr.save_results(results)


if __name__ == "__main__":
    main()
