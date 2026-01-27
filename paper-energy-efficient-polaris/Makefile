# Makefile for LaTeX document compilation

# Document name
DOC = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(DOC).pdf

# Main compilation target
$(DOC).pdf: $(DOC).tex sections/*.tex references/*.bib
	$(LATEX) $(DOC).tex
	$(BIBTEX) $(DOC)
	$(LATEX) $(DOC).tex
	$(LATEX) $(DOC).tex

# Quick compilation without bibliography
quick: $(DOC).tex sections/*.tex
	$(LATEX) $(DOC).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.fdb_latexmk *.fls

# Clean all generated files including PDF
cleanall: clean
	rm -f $(DOC).pdf

# View PDF (Linux)
view: $(DOC).pdf
	xdg-open $(DOC).pdf &

# View PDF (macOS)
view-mac: $(DOC).pdf
	open $(DOC).pdf

# Continuous compilation with latexmk
watch:
	latexmk -pdf -pvc $(DOC).tex

# Check for LaTeX errors
check: $(DOC).tex
	$(LATEX) -halt-on-error $(DOC).tex

# Word count (approximate)
wordcount: $(DOC).tex
	@echo "Approximate word count:"
	@detex $(DOC).tex sections/*.tex | wc -w

# Install required packages (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-lang-spanish

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Compile the full document with bibliography"
	@echo "  quick      - Quick compilation without bibliography"
	@echo "  clean      - Remove auxiliary files"
	@echo "  cleanall   - Remove all generated files including PDF"
	@echo "  view       - Open PDF viewer (Linux)"
	@echo "  view-mac   - Open PDF viewer (macOS)"
	@echo "  watch      - Continuous compilation with latexmk"
	@echo "  check      - Check for LaTeX errors"
	@echo "  wordcount  - Approximate word count"
	@echo "  install-deps - Install required LaTeX packages (Ubuntu/Debian)"
	@echo "  help       - Show this help message"

.PHONY: all quick clean cleanall view view-mac watch check wordcount install-deps help