#!/usr/bin/env python3
"""
PDF Line and Background Removal Utility

Accurately removes horizontal and vertical lines from PDFs,
preserving handwritten text, outputting a cleaned PDF.
"""

import sys
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
from pathlib import Path
import tempfile

def preprocess_image_remove_lines_and_bg(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = 255 - binary

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    detected_lines = cv2.add(detected_horizontal, detected_vertical)
    no_lines = cv2.subtract(binary_inv, detected_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, kernel, iterations=1)
    result = 255 - cleaned

    text_mask = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text_only = cv2.bitwise_and(gray, gray, mask=text_mask)
    white_bg = np.ones_like(gray) * 255
    final = np.where(text_mask > 0, text_only, white_bg).astype(np.uint8)
    return final

def pdf_to_images(pdf_path: Path, dpi: int = 300):
    print(f"Converting PDF to images (DPI={dpi})...")
    pil_images = convert_from_path(str(pdf_path), dpi=dpi)
    images = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) for im in pil_images]
    return images

def images_to_pdf(images, output_pdf: Path):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_files = []
        for i, img in enumerate(images):
            temp_path = Path(tempdir) / f"page_{i+1:03d}.png"
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(temp_path)
            temp_files.append(str(temp_path))

        pdf_bytes = img2pdf.convert(temp_files)
        if pdf_bytes is None:
            raise RuntimeError("img2pdf conversion failed, returned None")
        with open(output_pdf, "wb") as f:
            f.write(pdf_bytes)

def main():
    pdf_path_str = input("Enter the PDF file path: ").strip('" ')
    dpi_str = input("Enter DPI for conversion (default 300): ").strip()
    dpi = int(dpi_str) if dpi_str.isdigit() else 300

    pdf_path = Path(pdf_path_str)
    if not pdf_path.is_file():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("Loading PDF and processing pages...")
    images = pdf_to_images(pdf_path, dpi)
    cleaned_images = []
    for idx, img in enumerate(images, 1):
        print(f"Processing page {idx} of {len(images)}")
        cleaned_img = preprocess_image_remove_lines_and_bg(img)
        cleaned_images.append(cleaned_img)

    output_pdf = pdf_path.with_name(pdf_path.stem + "_cleaned.pdf")
    print(f"Saving cleaned PDF to {output_pdf} ...")
    images_to_pdf(cleaned_images, output_pdf)
    print("Processing complete.")

if __name__ == "__main__":
    main()
