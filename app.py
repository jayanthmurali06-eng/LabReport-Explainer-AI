
import os
import re
import io

import pymupdf
import numpy as np

from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# ---------------------------------------
# FLASK CONFIGURATION
# ---------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}


# ---------------------------------------
# LAB TEST EXPLANATIONS - DAY 3
# ---------------------------------------

LAB_EXPLANATIONS = {
    "hemoglobin": (
        "Hemoglobin is a protein in red blood cells "
        "that carries oxygen throughout the body."
    ),

    "glucose": (
        "Glucose is blood sugar that provides energy "
        "to the body's cells."
    ),

    "creatinine": (
        "Creatinine is a waste product commonly measured "
        "to assess kidney function."
    ),

    "urea": (
        "Urea is a waste product formed when the body "
        "breaks down protein."
    ),

    "cholesterol": (
        "Cholesterol is a fatty substance in the blood "
        "that helps build healthy cells."
    ),

    "platelets": (
        "Platelets are blood components that help blood "
        "clot and stop bleeding."
    ),

    "wbc": (
        "White blood cells help the body fight infections."
    ),

    "rbc": (
        "Red blood cells carry oxygen from the lungs "
        "to different parts of the body."
    ),

    "tsh": (
        "TSH is a hormone that helps regulate "
        "thyroid gland activity."
    ),

    "vitamin d": (
        "Vitamin D helps the body absorb calcium "
        "and maintain healthy bones."
    ),

    "vitamin b12": (
        "Vitamin B12 supports nerve function and "
        "red blood cell production."
    ),

    "calcium": (
        "Calcium is a mineral important for bones, "
        "teeth, muscles and nerve function."
    ),

    "bilirubin": (
        "Bilirubin is a yellow substance produced "
        "when red blood cells break down."
    ),

    "albumin": (
        "Albumin is a protein in the blood that helps "
        "maintain fluid balance."
    ),

    "sodium": (
        "Sodium is an electrolyte that helps regulate "
        "fluid balance and nerve function."
    ),

    "potassium": (
        "Potassium is an electrolyte important for "
        "muscle and heart function."
    ),

    "hba1c": (
        "HbA1c gives an estimate of average blood "
        "glucose levels over approximately three months."
    ),

    "esr": (
        "ESR is a blood test that can help identify "
        "inflammation in the body."
    ),

    "crp": (
        "CRP is a protein that can increase when "
        "there is inflammation in the body."
    ),
    "white blood cells": (
        "White blood cells help the immune system "
        "fight infections and protect the body."
    ),

    "mcv": (
        "MCV measures the average size of red blood cells."
    ),

    "hematocrit": (
        "Hematocrit measures the percentage of blood "
        "volume made up of red blood cells."
    ),
}


def get_explanation(test_name):
    name = test_name.lower().strip()

    for test, explanation in LAB_EXPLANATIONS.items():
        if test in name:
            return explanation

    return (
        "This test measures a specific component of the body. "
        "Consult a healthcare professional for interpretation."
    )


# ---------------------------------------
# DAY 1 - PDF TEXT EXTRACTION
# ---------------------------------------

def extract_pdf_text(pdf_document):

    extracted_pages = []

    for page_number, page in enumerate(pdf_document, start=1):

        words = page.get_text("words")

        # Sort words according to vertical and horizontal position
        words = sorted(words, key=lambda word: (round(word[1], 1), word[0]))

        native_text = " ".join(word[4] for word in words)

        extracted_pages.append(
            f"--- Page {page_number} ---\n{native_text}"
        )

    return "\n\n".join(extracted_pages)


# ---------------------------------------
# DAY 2 - OCR SUPPORT USING RAPIDOCR
# ---------------------------------------

_ocr_engine = None


def get_ocr_engine():

    global _ocr_engine

    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()

    return _ocr_engine


def extract_ocr_text(page):

    # Convert PDF page into an image
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image_bytes = pix.tobytes("png")

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_array = np.asarray(image)

    # Perform OCR
    ocr_engine = get_ocr_engine()

    result, _ = ocr_engine(image_array)

    extracted_words = []

    if result:

        for item in result:

            text = item[1]

            if text:
                extracted_words.append(text)

    return " ".join(extracted_words)


def extract_text_with_ocr(pdf_document):

    extracted_pages = []

    for page_number, page in enumerate(pdf_document, start=1):

        words = page.get_text("words")

        words = sorted(
            words,
            key=lambda word: (round(word[1], 1), word[0])
        )

        words = page.get_text("words", sort=True)

        lines = {}

        for word in words:
            line_key = (word[5], word[6])
            lines.setdefault(line_key, []).append(word)

        native_text = "\n".join(
            " ".join(word[4] for word in line_words)
            for line_words in lines.values()
        ).strip()

        # Use OCR if the page contains very little native text
        if len(native_text) < 20:

            try:
                page_text = extract_ocr_text(page)

            except Exception as e:
                print(f"OCR error on page {page_number}: {e}")
                page_text = native_text

        else:
            page_text = native_text

        extracted_pages.append(
            f"--- Page {page_number} ---\n{page_text}"
        )

    return "\n\n".join(extracted_pages)


# ---------------------------------------
# DAY 3 - LAB RESULT EXTRACTION
# ---------------------------------------


def parse_lab_results_from_pdf(pdf_document):

    results = []

    for page in pdf_document:

        words = page.get_text("words", sort=True)

        if not words:
            continue

        # Group words by their vertical position
        rows = {}

        for word in words:
            x0, y0, x1, y1, text = word[:5]

            row_key = round(y0 / 3)

            rows.setdefault(row_key, []).append(word)

        for row_words in rows.values():

            row_words.sort(key=lambda w: w[0])

            # Identify columns using their horizontal positions
            test_words = []
            value_words = []
            unit_words = []
            range_words = []

            for word in row_words:

                x0 = word[0]
                text = word[4]

                if 80 <= x0 < 220:
                    test_words.append(text)

                elif 220 <= x0 < 290:
                    value_words.append(text)

                elif 290 <= x0 < 350:
                    unit_words.append(text)

                elif 350 <= x0 < 445:
                    range_words.append(text)

            test_name = " ".join(test_words).strip()
            value_text = " ".join(value_words).strip()
            unit = " ".join(unit_words).strip()
            reference = " ".join(range_words).strip()

            # Ignore headings and incomplete rows
            if not test_name or not value_text or not reference:
                continue

            # Check numeric result
            value_match = re.search(
                r"-?\d+(?:\.\d+)?",
                value_text
            )

            # Check reference range
            range_match = re.search(
                r"(-?\d+(?:\.\d+)?)\s*[-–]\s*"
                r"(-?\d+(?:\.\d+)?)",
                reference
            )

            if not value_match or not range_match:
                continue

            value = float(value_match.group(0))
            low = float(range_match.group(1))
            high = float(range_match.group(2))

            if low > high:
                continue

            if value < low:
                status = "Low"

            elif value > high:
                status = "High"

            else:
                status = "Within range"

            results.append({

                "name": test_name,

                "value": value_match.group(0),

                "unit": unit,

                "low": low,

                "high": high,

                "status": status,

                "explanation": get_explanation(test_name)

            })

    return results


# ---------------------------------------
# FILE VALIDATION
# ---------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------------------------------
# MAIN FLASK ROUTE
# ---------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    extracted_text = None
    results = []
    error = None
    filename = None

    if request.method == "POST":

        if "report" not in request.files:

            error = "No file selected."

            return render_template(
                "index.html",
                extracted_text=extracted_text,
                results=results,
                error=error,
                filename=filename
            )

        file = request.files["report"]

        if file.filename == "":

            error = "Please select a PDF file."

        elif not allowed_file(file.filename):

            error = "Only PDF files are allowed."

        else:

            filename = secure_filename(file.filename)

            try:

                # Read PDF into memory
                pdf_bytes = file.read()

                if not pdf_bytes.startswith(b"%PDF-"):
                    raise ValueError("Invalid PDF file.")

                pdf_document = pymupdf.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                )

                try:

                    # Extract text with OCR fallback
                    extracted_text = extract_text_with_ocr(
                        pdf_document
                    )

                    # Extract laboratory results
                    results = parse_lab_results_from_pdf(
                        pdf_document
                    )

                finally:
                    pdf_document.close()

            except Exception as e:

                print("Processing error:", e)

                error = (
                    "Unable to process this PDF. "
                    "Please upload a valid and readable lab report."
                )

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        results=results,
        error=error,
        filename=filename
    )


# ---------------------------------------
# FILE SIZE ERROR HANDLER
# ---------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        extracted_text=None,
        results=[],
        error="File is too large. Maximum upload size is 10 MB.",
        filename=None
    ), 413


# ---------------------------------------
# RUN APPLICATION
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
