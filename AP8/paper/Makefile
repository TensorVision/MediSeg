SOURCE = vorlage

make:
	pdflatex $(SOURCE).tex -output-format=pdf # Referenzen erstellen
	# makeglossaries $(SOURCE)
	bibtex $(SOURCE)
	pdflatex $(SOURCE).tex -output-format=pdf # Referenzen einbinden
	pdflatex $(SOURCE).tex -output-format=pdf # Referenzen einbinden
	make clean

move:
	mv $(SOURCE).pdf pixel-seminar-martin-thoma-`date +'%Y-%m-%d'`.pdf

ebook:
	latexml --dest=$(SOURCE).xml $(SOURCE).tex
	latexmlpost -dest=$(SOURCE).html $(SOURCE).xml
	ebook-convert $(SOURCE).html $(SOURCE).epub --language de --no-default-epub-cover

arxiv:
	rm -f upload.zip
	make clean
	zip -r upload.zip . -x \*.git\* -x Makefile -x *.zip -x *.pdf -x *.md -x *.sublime\* -x vortrag\* -x presentation-images\* -x unused\*

clean:
	rm -rf  $(TARGET) *.class *.html *.aux *.out *.thm *.idx *.toc *.ilg *.glg *.glo *.ist *.xdy *.fdb_latexmk *.bak *.blg *.glsdefs *.acn *.acr *.alg *.nls *.nlo *.bak *.pyg *.lot *.lof *.fls *.run.xml *.bcf
	rm -rf *.log  # Needed to analyze paper generation
	# rm -rf *.bbl *.ind *.gls # Needed for arxiv
