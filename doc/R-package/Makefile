# This is the makefile for compiling Rmarkdown files into the md file with results.
PKGROOT=../../R-package

# ADD The Markdown to be built here, with suffix md
xgboostfromJSON.md: $(PKGROOT)/vignettes/xgboostfromJSON.Rmd
xgboost_introduction.md: $(PKGROOT)/vignettes/xgboost_introduction.Rmd

# General Rules for build rmarkdowns, need knitr
%.md:
	Rscript -e \
	"require(methods);"\
	"require(knitr);"\
	"knitr::opts_knit\$$set(root.dir=\".\");"\
	"knitr::opts_chunk\$$set(fig.path=\"../web-data/xgboost/knitr/$(basename $@)-\");"\
	"knitr::knit(\"$+\")"
