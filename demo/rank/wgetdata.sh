#!/bin/bash
wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2008.rar
unrar x MQ2008.rar
mv -f MQ2008/Fold1/*.txt .
