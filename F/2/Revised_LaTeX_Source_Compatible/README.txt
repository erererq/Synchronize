Revised blue-marked manuscript source package.

Preservation statement:
1. All original manuscript figure references, figure files, captions, and related body discussion are retained.
2. Reviewer-requested two-dimensional heat maps and the transmitter--receiver workflow are supplementary additions, not replacements.
3. New or modified manuscript text is marked in blue through the \rev{...} command.

Standard compile sequence:
pdflatex -interaction=nonstopmode -halt-on-error Manuscript.tex
bibtex Manuscript
pdflatex -interaction=nonstopmode -halt-on-error Manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error Manuscript.tex

For a compact PDF 1.4 file with broad viewer compatibility, run:
bash compile_compatible.sh

Expected output: Manuscript_Compatible.pdf (21 pages, A4).
