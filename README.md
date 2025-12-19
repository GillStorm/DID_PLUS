# DID++: AI-Powered Decentralized Identity Verification System

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
Python 3.9+

## Installation

1. **Install Python 3.9**
   - Download Python 3.9 from the [official website](https://www.python.org/downloads/release/python-390/).
   - Follow the instructions for your operating system to install Python 3.9.
   - Ensure `python` and `pip` point to Python 3.9 by running:
     ```sh
     python --version
     pip --version
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

## Running Tests

- To run all tests:
  ```sh
  pytest tests/
  ```

- To generate a detailed PDF report for the LFW face verification test, set the environment variable before running tests:
  - **PowerShell:**
    ```sh
    $env:LFW_PDF_REPORT=1; pytest tests/
    ```
  - **Command Prompt:**
    ```sh
    set LFW_PDF_REPORT=1 && pytest tests/
    ```

- The PDF report will be saved as `lfw_face_verification_report.pdf` in the project directory.

### Viewing the Detailed Report

To see the detailed report output in your terminal, run:
```sh
pytest tests/ -s
```
The `-s` flag ensures that all print output (including the model's detailed report) is shown.

### Generating a PDF Report

To generate a PDF report for the LFW face verification test, set the environment variable `LFW_PDF_REPORT=1`.

**Recommended:** Create a `.env` file in the project root and add:
```
LFW_PDF_REPORT=1
```
You can also set the variable in your shell as shown below:
  - **PowerShell:**
    ```sh
    $env:LFW_PDF_REPORT=1; pytest tests/
    ```
  - **Command Prompt:**
    ```sh
    set LFW_PDF_REPORT=1 && pytest tests/
    ```
The PDF report will be saved as `lfw_face_verification_report.pdf` in the project directory.
