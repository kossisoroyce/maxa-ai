.PHONY: all clean view wordcount

TEX = pdflatex
BIBTEX = bibtex
DOC = main

# Main targets
all: figures pdf

# Generate PDF with proper bibliography
pdf: $(DOC).pdf

# Rule to compile the main document
$(DOC).pdf: $(DOC).tex
	$(TEX) $<
	if [ -f $(DOC).aux ]; then \
		$(BIBTEX) $(DOC); \
		$(TEX) $<; \
	fi
	$(TEX) $<

# Compile figures
figures: figures/architecture.pdf figures/personalization.pdf

# Rule to compile figures
figures/%.pdf: figures/%.tex
	cd figures && pdflatex -interaction=nonstopmode -output-directory=.. $<

# Clean up
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.lol *.dvi *.fls *.fdb_latexmk *.synctex.gz

cleanall: clean
	rm -f $(DOC).pdf

# View the generated PDF
view: $(DOC).pdf
	open $(DOC).pdf

# Count words in the document
wordcount:
	@echo "Word count (excluding references):"
	@pdftotext $(DOC).pdf - | wc -w

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the document and figures (default)"
	@echo "  pdf       - Build the PDF document"
	@echo "  figures   - Build all figures"
	@echo "  clean     - Remove temporary files"
	@echo "  cleanall  - Remove all generated files"
	@echo "  view      - Open the generated PDF"
	@echo "  wordcount - Count words in the document"
