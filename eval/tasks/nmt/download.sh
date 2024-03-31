#!/bin/bash
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en'
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de'
for YEAR in 2012 2013 2014; do
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.en"
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.de"
done
cat newstest{2012,2013}.en >dev.en
cat newstest{2012,2013}.de >dev.de
cp newstest2014.en test.en
cp newstest2014.de test.de

cat train.de train.en | subword-nmt learn-bpe -s 32000 > codes
for SET in train dev test; do
  subword-nmt apply-bpe -c codes <${SET}.en >${SET}.en.bpe
  subword-nmt apply-bpe -c codes <${SET}.de >${SET}.de.bpe
done