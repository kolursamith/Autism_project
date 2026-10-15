#!/usr/bin/env bash
# Stage 1 git step: (1) skeleton commit, (2) regenerate the report (it checks the commits), (3) stage commit.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v git >/dev/null || { echo "git is not installed: run 'sudo apt install -y git' first"; exit 1; }
git init -q
git config user.name "Samith Kolur"
git config user.email "santoshkolur@gmail.com"
git add .gitignore results/README.md $(find . -name .gitkeep -not -path './.venv/*')
git commit -qm "Stage 1: project skeleton"
.venv/bin/python scripts/01_write_report.py
./scripts/01_make_pdf.sh
git add -A
git commit -qm "Stage 1: workspace, environment and dataset audit"
echo '$ git log --oneline'; git log --oneline
echo '$ git status --short'; git status --short; echo "(end of status)"
