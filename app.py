
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pymupdf
import re

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}


def extract_pdf_text(file_bytes):
    """Extract PDF text while preserving approximate visual row order."""
    pages_text = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            words = page.get_text("words")
            if not words:
                continue

            # Group words by their vertical position, then sort left to right.
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

            page_text = "\n".join(page_lines).strip()
            if page_text:
                pages_text.append(f"--- Page {page_number} ---\n{page_text}")

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
                        results = parse_lab_results(extracted_text)

                        if not extracted_text:
                            error = "No readable text was found in this PDF."

                except Exception:
                    error = (
                        "We couldn't read this PDF. Please try another "
                        "valid PDF file."
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
