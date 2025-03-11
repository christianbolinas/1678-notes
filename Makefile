SRC_EXT = md
OUT_EXT = pdf

MD_FILES = $(wildcard *.md)
PDF_FILES = $(MD_FILES:.md=.pdf)

all: $(PDF_FILES)

%.pdf: %.md
	pandoc $< -o $@ 

clean:
	rm -f $(PDF_FILES)