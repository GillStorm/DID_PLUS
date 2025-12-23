# AI-Powered Decentralized Identity Verification System

## Overview
DID++ is a multi-modal identity verification system combining:
- Face biometrics (MediaPipe)
- Voice biometrics (MFCC)
- Document verification (OCR)
- Weighted score fusion

Designed for Self-Sovereign Identity (SSI) with future blockchain integration.

## Features
- Face recognition using MediaPipe landmarks
- Voice verification using MFCC embeddings
- Document OCR verification using EasyOCR
- Multi-modal score fusion
- Privacy-preserving design (no raw biometric storage)

## Requirements
- **Python 3.11.9**

## Installation

1. **Install Python 3.11.9**
   - Download Python 3.11.9 from the [official Python website](https://www.python.org/downloads/).
   - Follow the instructions for your operating system.
   - Verify the installation:
     ```sh
     python --version
     pip --version
     ```
     Expected output:
     ```
     Python 3.11.9
     ```

2. **Install dependencies**
   - It is recommended to use a virtual environment:
     ```sh
     python -m venv .venv
     # On Windows:
     .venv\Scripts\activate
     # On Linux/macOS:
     source .venv/bin/activate
     ```
   - Install required packages:
     ```sh
     pip install -r requirements.txt
     ```

## Running Tests & Generating Reports

To run all tests:
```sh
pytest tests/
```

To generate a PDF report for the LFW face verification test, set the environment variable `LFW_PDF_REPORT=1` before running tests. You can do this in two ways:

**Recommended:** Create a `.env` file in the project root with:
```
LFW_PDF_REPORT=1
```
The project uses `python-dotenv` to automatically load this variable.

Or, set it in your shell:
- **PowerShell:**
  ```sh
  $env:LFW_PDF_REPORT=1; pytest tests/
  ```
- **Command Prompt:**
  ```sh
  set LFW_PDF_REPORT=1 && pytest tests/
  ```

The PDF report will be saved as `lfw_face_verification_report.pdf` in the project directory.

To see detailed output in the terminal, use:
```sh
pytest tests/ -s
```
The `-s` flag ensures all print output is shown.

## Running the Application

To start the Flask app, set the following environment variables and use the Flask CLI:

**Required:**
- `FLASK_APP=app.py`
- `FLASK_ENV=development` (optional, enables debug mode)

**On Windows (PowerShell):**
```sh
$env:FLASK_APP="app.py"
flask run
```

**On Windows (Command Prompt):**
```sh
set FLASK_APP=app.py
flask run
```

The app will be available at http://127.0.0.1:5000/


## Environment Variables

- `LFW_PDF_REPORT=1` — Enables PDF report generation for LFW face verification tests.
- `FLASK_APP=app.py` — Required for `flask run`.
- `FLASK_ENV=development` — (Optional) Enables debug mode for Flask.

If using a `.env` file, all variables will be loaded automatically if `python-dotenv` is installed (already in requirements).
