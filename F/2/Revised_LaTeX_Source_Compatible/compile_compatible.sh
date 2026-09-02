#!/usr/bin/env bash
set -euo pipefail
pdflatex -interaction=nonstopmode -halt-on-error Manuscript.tex
bibtex Manuscript
pdflatex -interaction=nonstopmode -halt-on-error Manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error Manuscript.tex
if command -v gs >/dev/null 2>&1; then
  gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
    -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
    -dDetectDuplicateImages=true -dCompressFonts=true \
    -sOutputFile=Manuscript_Compatible.pdf Manuscript.pdf
else
  cp Manuscript.pdf Manuscript_Compatible.pdf
fi
