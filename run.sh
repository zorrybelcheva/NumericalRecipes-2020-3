#!/bin/bash

echo ""
echo "Running NUR first hand-in exercise solutions"
echo "Z. Belcheva, student ID: 2418797"
echo ""

echo "Check if plotting directory exists:"
if [ ! -d "plots" ]; then
  echo "Does not exist; created."
  mkdir plots
fi

echo "Check if output directory exists:"
if [ ! -d "output" ]; then
  echo "Does not exist; created."
  mkdir output
fi

echo "Downloading data for ex. 1"
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m11.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m12.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m13.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m14.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m15.txt
echo "Data downloaded."

echo "Running exercise 1"
python3 -W ignore model-optimisation.py > output/model-optimisation.txt

echo "Running exercise 2"
python3 -W ignore cic-fft.py > output/cic-fft.txt

echo "Generating .pdf of report"
pdflatex report.tex
pdflatex report.tex

echo "Done!"
