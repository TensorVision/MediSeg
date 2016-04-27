SOURCE = Pflichtenheft

make:
	pdflatex $(SOURCE).tex -output-format=pdf
	bibtex $(SOURCE)
	pdflatex $(SOURCE).tex -output-format=pdf
	pdflatex $(SOURCE).tex -output-format=pdf
	make clean

clean:
	rm -rf  $(TARGET) *.class *.html *.log *.aux *.out *.glo *.glg *.gls *.ist *.xdy *.1 *.toc *.bak *.blg *.glsdefs *.bbl
