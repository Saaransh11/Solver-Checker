
#!/usr/bin/env python3
"""
PDF Handwritten Text OCR using Google Cloud Vision API
Converts PDF pages to images and extracts handwritten text using Vision API
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict
from typing import List, Dict, Optional

# Suppress Google Cloud logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Import required libraries
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from pdf2image import convert_from_path
    from PIL import Image
    import io
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("\nInstall required packages:")
    print("pip install google-cloud-vision pdf2image pillow google-auth")
    print("\nFor PDF processing, you also need poppler:")
    print("- Ubuntu/Debian: sudo apt-get install poppler-utils")
    print("- macOS: brew install poppler") 
    print("- Windows: Download poppler binaries and add to PATH")
    sys.exit(1)


class PDFHandwritingOCR:
    """PDF Handwriting OCR processor using Google Cloud Vision API"""

    def __init__(self, credentials_path: str):
        """
        Initialize the OCR processor with credentials

        Args:
            credentials_path: Path to Google Cloud service account JSON file
        """
        self.credentials_path = credentials_path
        self.client = self._setup_vision_client()

    def _setup_vision_client(self):
        """Setup Google Cloud Vision API client"""
        try:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")

            # Load credentials from JSON file
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )

            # Create Vision API client
            client = vision.ImageAnnotatorClient(credentials=credentials)

            print("✅ Google Cloud Vision API client initialized successfully")
            return client

        except Exception as e:
            print(f"❌ Failed to setup Vision API client: {e}")
            sys.exit(1)

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (higher = better quality but slower)

        Returns:
            List of PIL Image objects
        """
        try:
            print(f"📄 Converting PDF to images: {pdf_path}")
            pages = convert_from_path(pdf_path, dpi=dpi)
            print(f"✅ Successfully converted {len(pages)} pages")
            return pages

        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            return []

    def extract_text_from_image(self, image: Image.Image, page_num: int) -> Dict:
        """
        Extract handwritten text from a single image using Vision API

        Args:
            image: PIL Image object
            page_num: Page number for reference

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            print(f"🔍 Processing page {page_num}...")

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create Vision API image object
            vision_image = vision.Image(content=img_byte_arr)

            # Configure for handwriting detection
            image_context = vision.ImageContext(
                language_hints=["en-t-i0-handwrit"]  # English handwriting model
            )

            # Use DOCUMENT_TEXT_DETECTION for better layout understanding
            feature = vision.Feature(
                type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
            )

            # Create request
            request = vision.AnnotateImageRequest(
                image=vision_image,
                features=[feature],
                image_context=image_context
            )

            # Make API call
            response = self.client.annotate_image(request=request)

            # Check for errors
            if response.error.message:
                print(f"⚠️  API error on page {page_num}: {response.error.message}")
                return {"page": page_num, "text": "", "error": response.error.message}

            # Extract text
            full_text = ""
            confidence_scores = []

            if hasattr(response, 'full_text_annotation') and response.full_text_annotation:
                full_text = response.full_text_annotation.text

                # Extract confidence scores
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        confidence_scores.append(block.confidence)

            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            result = {
                "page": page_num,
                "text": full_text.strip(),
                "confidence": avg_confidence,
                "word_count": len(full_text.split()) if full_text else 0
            }

            print(f"✅ Page {page_num}: {result['word_count']} words, confidence: {avg_confidence:.2f}")
            return result

        except Exception as e:
            print(f"❌ Error processing page {page_num}: {e}")
            return {"page": page_num, "text": "", "error": str(e)}

    def process_pdf(self, pdf_path: str, dpi: int = 200) -> Dict:
        """
        Process entire PDF and extract handwritten text

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion

        Returns:
            Dictionary with all extracted text and metadata
        """
        start_time = time.time()

        print("🚀 Starting PDF handwriting OCR process...")
        print(f"📁 Input file: {pdf_path}")

        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return {"error": "File not found"}

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi)
        if not images:
            return {"error": "Failed to convert PDF to images"}

        # Process each page
        all_results = {
            "source_file": pdf_path,
            "total_pages": len(images),
            "pages": [],
            "full_text": "",
            "total_words": 0,
            "avg_confidence": 0.0,
            "processing_time": 0.0
        }

        confidences = []

        for i, image in enumerate(images, 1):
            # Add small delay to avoid rate limiting
            if i > 1:
                time.sleep(0.5)

            page_result = self.extract_text_from_image(image, i)
            all_results["pages"].append(page_result)

            if page_result.get("text"):
                all_results["full_text"] += f"\n--- Page {i} ---\n"
                all_results["full_text"] += page_result["text"] + "\n"
                all_results["total_words"] += page_result.get("word_count", 0)

                if page_result.get("confidence", 0) > 0:
                    confidences.append(page_result["confidence"])

        # Calculate final metrics
        all_results["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        all_results["processing_time"] = time.time() - start_time

        return all_results

    def display_results(self, results: Dict):
        """Display OCR results in terminal"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return

        print("\n" + "="*60)
        print("📊 OCR PROCESSING RESULTS")
        print("="*60)

        print(f"📄 Source File: {results['source_file']}")
        print(f"📑 Total Pages: {results['total_pages']}")
        print(f"📝 Total Words: {results['total_words']}")
        print(f"🎯 Average Confidence: {results['avg_confidence']:.2f}")
        print(f"⏱️  Processing Time: {results['processing_time']:.2f} seconds")

        # Show page-by-page summary
        print(f"\n📋 Page Summary:")
        for page in results['pages']:
            status = "✅" if page.get('text') else "❌"
            confidence = page.get('confidence', 0)
            words = page.get('word_count', 0)
            print(f"  {status} Page {page['page']}: {words} words (confidence: {confidence:.2f})")

        print("\n" + "="*60)
        print("📄 EXTRACTED TEXT")
        print("="*60)

        if results['full_text'].strip():
            print(results['full_text'])
        else:
            print("❌ No text was extracted from the PDF")

        print("="*60)

    def save_results(self, results: Dict, output_file: Optional[str] = None):
        """Save results to text file"""
        if "error" in results:
            return

        if not output_file:
            pdf_name = Path(results['source_file']).stem
            output_file = f"{pdf_name}_extracted_text.txt"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Extracted Text from: {results['source_file']}\n")
                f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Pages: {results['total_pages']}\n")
                f.write(f"Total Words: {results['total_words']}\n")
                f.write(f"Average Confidence: {results['avg_confidence']:.2f}\n")
                f.write("\n" + "="*60 + "\n")
                f.write("EXTRACTED TEXT\n")
                f.write("="*60 + "\n")
                f.write(results['full_text'])

            print(f"💾 Results saved to: {output_file}")

        except Exception as e:
            print(f"⚠️  Could not save results to file: {e}")


def main():
    """Main function to run the PDF OCR script"""
    print("🔤 PDF Handwriting OCR using Google Cloud Vision API")
    print("="*60)

    # Configuration
    CREDENTIALS_FILE = "credentials.json"  # Update this path

    # Check if credentials file exists
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"❌ Credentials file not found: {CREDENTIALS_FILE}")
        print("\nSetup instructions:")
        print("1. Download your service account JSON key from Google Cloud Console")
        print("2. Save it as 'credentials.json' in the same directory as this script")
        print("3. Make sure Cloud Vision API is enabled in your project")
        return

    # Get PDF file path from user input or command line
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("\n📁 Enter the path to your PDF file: ").strip().strip('"')

    if not pdf_path:
        print("❌ No PDF file specified")
        return

    try:
        # Initialize OCR processor
        ocr = PDFHandwritingOCR(CREDENTIALS_FILE)

        # Process PDF
        results = ocr.process_pdf(pdf_path)

        # Display results in terminal
        ocr.display_results(results)

        ocr.save_results(results)

        print("\n🎉 OCR processing completed!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
