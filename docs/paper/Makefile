# Build the paper. Needs a LaTeX distribution with latexmk (e.g. `apt install latexmk
# texlive-latex-extra`) or tectonic (`TEX=tectonic make`).
TEX ?= latexmk

paper.pdf: paper.tex references.bib generated/numbers.tex $(wildcard figures/*.pdf generated/*.tex)
ifeq ($(TEX),tectonic)
	tectonic paper.tex
else
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
endif

# Regenerate figures, tables and numbers from the newest benchmark run.
generated/numbers.tex:
	cd ../bench && uv run python plot.py

clean:
	latexmk -C 2>/dev/null || true
	rm -f paper.bbl paper.blg

.PHONY: clean
