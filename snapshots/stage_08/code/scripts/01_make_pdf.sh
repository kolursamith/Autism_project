#!/usr/bin/env bash
# Build reports/STAGE_01_REPORT.pdf (report + dataset decisions) with pandoc + LibreOffice.
set -euo pipefail
cd "$(dirname "$0")/../reports/_pdf_build"
{ cat ../STAGE_01_REPORT.md; printf '\n\n<div style="page-break-before: always"></div>\n\n'; cat ../../data_registry/DATASET_DECISIONS.md; } > combined.md
pandoc combined.md -s --metadata title="Stage 1 Report — ASD Hybrid 10-fold" -c style.css --embed-resources -o report.html
libreoffice --headless --convert-to pdf:writer_web_pdf_Export report.html >/dev/null 2>&1
mv report.pdf ../STAGE_01_REPORT.pdf && rm -f combined.md report.html
pdfinfo ../STAGE_01_REPORT.pdf | grep -E "Pages|Page size"
