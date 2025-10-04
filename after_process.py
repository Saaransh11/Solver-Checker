#!/usr/bin/env python3
"""
JSON OCR Text Splitter - Interactive Processing Only


Focused script for interactive text processing with JSON input/output only.
Splits extracted OCR text into individual answers and saves as JSON.
"""


import re
import json
import sys
from pathlib import Path
from typing import Dict


class JSONAnswerSplitter:
    """
    JSON-focused answer text splitter for interactive processing
    """


    def __init__(self):
        # Answer detection patterns
        self.patterns = [
            r'Ans\s*(\d+)\s*\.',           # "Ans 1.", "Ans 2."
            r'Answer\s*(\d+)\s*\.',        # "Answer 1.", "Answer 2."
            r'Q\s*(\d+)\s*\.',             # "Q 1.", "Q 2."
            r'Question\s*(\d+)\s*\.',      # "Question 1.", "Question 2."
            r'^(\d+)\s*\.',                 # "1.", "2." (start of line)
            r'\((\d+)\)',                   # "(1)", "(2)"
            r'(\d+)\)',                      # "1)", "2)"
        ]


        print("🔧 JSON Answer Splitter initialized")
        print(f"📋 Loaded {len(self.patterns)} detection patterns")


    def detect_markers(self, text: str):
        """Detect answer markers and return sorted positions"""
        markers = []


        for pattern in self.patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                start_pos = match.start()
                answer_num = int(match.group(1))
                marker_text = match.group(0)
                markers.append((start_pos, answer_num, marker_text))


        # Sort by position and remove duplicates
        markers.sort(key=lambda x: x[0])
        unique_markers = []
        for marker in markers:
            if not unique_markers or marker[0] != unique_markers[-1][0]:
                unique_markers.append(marker)


        return unique_markers


    def clean_text(self, text: str) -> str:
        """Clean answer text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())


        # Remove leading/trailing punctuation
        cleaned = re.sub(r'^[\s\-_=]+|[\s\-_=]+$', '', cleaned)


        # Fix common OCR spacing issues
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned)


        return cleaned.strip()


    def split_text(self, text: str) -> Dict[str, str]:
        """Split text into answers and return as JSON-compatible dict"""
        print("🔍 Detecting answer markers...")


        markers = self.detect_markers(text)


        if not markers:
            print("⚠️  No answer markers found - returning full text as answer 1")
            return {"1": self.clean_text(text)}


        print(f"✅ Found {len(markers)} answer markers")


        answers = {}


        for i, (start_pos, answer_num, marker_text) in enumerate(markers):
            # Get end position
            if i + 1 < len(markers):
                end_pos = markers[i + 1][0]
            else:
                end_pos = len(text)


            # Extract and clean answer text
            answer_start = start_pos + len(marker_text)
            answer_text = text[answer_start:end_pos]
            cleaned_text = self.clean_text(answer_text)


            if cleaned_text:
                answers[str(answer_num)] = cleaned_text
                print(f"   📝 Answer {answer_num}: {len(cleaned_text)} characters")


        return answers


    def load_json(self, file_path: str) -> str:
        """Load text from JSON file"""
        print(f"📄 Loading JSON file: {file_path}")


        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)


        # Handle different JSON structures
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if 'text' in data:
                return data['text']
            elif 'content' in data:
                return data['content']
            elif len(data) == 1:
                return list(data.values())[0]
            else:
                # Combine all string values
                text_parts = [str(v) for v in data.values() if isinstance(v, str)]
                return '\n'.join(text_parts)
        elif isinstance(data, list):
            text_parts = [str(item) for item in data if isinstance(item, str)]
            return '\n'.join(text_parts)
        else:
            return str(data)


    def save_json(self, answers: Dict[str, str], output_file: str):
        """Save answers to JSON file"""
        output_path = Path(output_file).with_suffix('.json')


        # Create structured JSON output
        output_data = {
            "total_answers": len(answers),
            "answers": answers,
            "metadata": {
                "processed_by": "JSON Answer Splitter",
                "format": "answer_number: answer_text"
            }
        }


        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


        print(f"💾 Saved JSON: {output_path}")
        return str(output_path)


    def preview_answers(self, answers: Dict[str, str], max_chars: int = 100):
        """Display preview of answers"""
        print("\n📋 SPLIT ANSWERS PREVIEW:")
        print("=" * 50)


        for answer_num in sorted(answers.keys(), key=int):
            answer_text = answers[answer_num]
            preview = answer_text[:max_chars]
            if len(answer_text) > max_chars:
                preview += "..."


            print(f"\n📝 ANSWER {answer_num} ({len(answer_text)} chars):")
            print("-" * 30)
            print(preview)


        print("\n" + "=" * 50)


def main_process(text_path):
    """
    Main interactive function - JSON focused
    """
    print("📝 JSON OCR Text Splitter - Interactive Mode")
    print("=" * 50)
    print("Features:")
    print("• Interactive text processing only")
    print("• JSON input and output")  
    print("• Answer pattern detection and splitting")
    print("=" * 50)


    splitter = JSONAnswerSplitter()

    text = ""


    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {text_path}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return



    if not text.strip():
        print("❌ No text provided")
        return


    print(f"\n📊 Input text loaded: {len(text)} characters")


    # Split the text
    try:
        answers = splitter.split_text(text)
    except Exception as e:
        print(f"❌ Error processing text: {e}")
        return


    if not answers:
        print("❌ No answers extracted")
        return


    # Preview results
    splitter.preview_answers(answers)


    # Get output filename
    output_name = "split_answers"


    try:
        output_file = splitter.save_json(answers, output_name)


        print(f"\n✅ Processing completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        print(f"📊 Total answers extracted: {len(answers)}")


        # Show JSON structure
        print("\n📋 JSON Structure:")
        print("{")
        print('  "total_answers": number,')
        print('  "answers": {')
        for i, answer_num in enumerate(sorted(answers.keys(), key=int)):
            comma = "," if i < len(answers) - 1 else ""
            print(f'    "{answer_num}": "answer text"{comma}')
        print('  },')
        print('  "metadata": {...}')
        print("}")
        return output_file


    except Exception as e:
        print(f"❌ Error saving JSON: {e}")