#!/usr/bin/env sh
export tmpfile=$(mktemp --suffix=.qmd)
head -n 11 vignettes/xgboost_introduction.qmd > ${tmpfile}
printf "jupyter: ir\n" >> ${tmpfile}
tail -n +12 vignettes/xgboost_introduction.qmd >> ${tmpfile}
quarto render ${tmpfile} --to md -o xgboost_introduction.md
mv xgboost_introduction.md ../doc/R-package/xgboost_introduction.md
rm ${tmpfile}
