
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import pymupdf
import numpy as np
import io
import re

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

# Initialize OCR only when a scanned page actually needs it.
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    return _ocr_engine


def extract_page_text(page):
    """Extract text from a normal PDF page in approximate row order."""
    words = page.get_text("words")

    if not words:
        return ""

    words.sort(key=lambda word: (word[1], word[0]))
    lines = []

    for word in words:
        x0, y0, x1, y1, text = word[:5]
        placed = False

        for line in lines:
            if abs(line["y"] - y0) <= 3:
                line["words"].append((x0, text))
                placed = True
                break

        if not placed:
            lines.append({"y": y0, "words": [(x0, text)]})

    lines.sort(key=lambda line: line["y"])
    page_lines = []

    for line in lines:
        line["words"].sort(key=lambda item: item[0])
        page_lines.append(
            " ".join(text for _, text in line["words"])
        )

    return "\n".join(page_lines).strip()


def extract_page_ocr(page):
    """Render a scanned PDF page and recognize its text with RapidOCR."""
    # Render at 2x resolution to improve OCR readability.
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )

    image_array = np.array(image)
    ocr_result, _ = get_ocr_engine()(image_array)

    if not ocr_result:
        return ""

    # RapidOCR returns detected text items with bounding boxes and scores.
    recognized_lines = []

    for item in ocr_result:
        if len(item) >= 2 and item[1]:
            recognized_lines.append(str(item[1]))

    return "\n".join(recognized_lines).strip()


def extract_pdf_text(file_bytes):
    """
    Extract text from normal PDFs and use OCR for scanned or
    mostly image-based pages.
    """
    pages_text = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        if pdf.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(pdf, start=1):
            page_text = extract_page_text(page)

            # Use OCR if native text extraction finds little text.
            if len(page_text.strip()) < 20:
                page_text = extract_page_ocr(page)

            if page_text:
                pages_text.append(
                    f"--- Page {page_number} ---\n{page_text}"
                )

    return "\n\n".join(pages_text)


def parse_lab_results(text):
    """Parse supported lab rows and compare values with printed ranges."""
    results = []

    row_pattern = re.compile(
        r"^(?P<name>.+?)\s+"
        r"(?P<value>-?\d+(?:\.\d+)?)\s+"
        r"(?P<unit>\S+)\s+"
        r"(?P<low>-?\d+(?:\.\d+)?)\s*[-–]\s*"
        r"(?P<high>-?\d+(?:\.\d+)?)"
        r"(?:\s+(?:LOW|L|HIGH|H|NORMAL|N|IN\s+RANGE))?$",
        re.IGNORECASE,
    )

    ignored_names = {
        "test", "result", "unit", "reference", "range", "flag",
        "patient", "sample", "age", "collected", "specimen", "status",
    }

    for line in text.splitlines():
        line = line.strip()
        match = row_pattern.match(line)

        if not match:
            continue

        name = match.group("name").strip()

        if name.lower() in ignored_names:
            continue

        try:
            value = float(match.group("value"))
            low = float(match.group("low"))
            high = float(match.group("high"))
        except ValueError:
            continue

        if low > high:
            continue

        if value < low:
            status = "Low"
        elif value > high:
            status = "High"
        else:
            status = "Within range"

        results.append({
            "name": name,
            "value": match.group("value"),
            "unit": match.group("unit"),
            "low": match.group("low"),
            "high": match.group("high"),
            "status": status,
        })

    return results


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = None
    results = []
    error = None
    filename = None

    if request.method == "POST":
        uploaded_file = request.files.get("report")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a PDF lab report first."
        else:
            filename = secure_filename(uploaded_file.filename)

            if not filename or not filename.lower().endswith(".pdf"):
                error = "Please upload a PDF file."
            else:
                try:
                    file_bytes = uploaded_file.read()

                    if not file_bytes:
                        error = "The selected file is empty."

                    elif not file_bytes.startswith(b"%PDF-"):
                        error = "This file does not appear to be a valid PDF."

                    else:
                        extracted_text = extract_pdf_text(file_bytes)

                        if extracted_text:
                            results = parse_lab_results(extracted_text)
                        else:
                            error = (
                                "No readable text could be extracted. "
                                "Try a clearer scan or a higher-quality PDF."
                            )

                except Exception:
                    app.logger.exception("PDF extraction or OCR failed")
                    extracted_text = None
                    results = []
                    error = (
                        "We couldn't process this PDF. Please try another "
                        "valid PDF or a clearer scan."
                    )

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        results=results,
        error=error,
        filename=filename,
    )


@app.errorhandler(413)
def file_too_large(error):
    return "File too large. Please upload a PDF smaller than 10 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)

