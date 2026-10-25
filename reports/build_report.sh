#!/usr/bin/env bash
# Build from the repository root so figure/data paths resolve consistently.
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$report_dir/.."
latexmk -quiet -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=reports/build reports/实验报告模板.tex
cp -- reports/build/实验报告模板.pdf reports/实验报告模板.pdf
printf '%s\n' 'Updated reports/实验报告模板.pdf and reports/build/实验报告模板.pdf'
