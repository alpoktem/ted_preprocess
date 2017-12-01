# ted_data_preprocess

Preprocess the prosodically annotated TED corpus. Annotations of talks are prepared using: https://github.com/laic/prosody

## Processing a single talk with tedDataToPickle.py:

Input files: 

1. `.word.txt` (cmd input as `-w`)

2. `.word.txt.norm.align` (cmd input as `-l`)

3. `.aggs.alignword.txt` for fundemental frequency and intensity (cmd input as `-f` and `-i`)

Output file:
1. CSV file with word aligned features (cmd input as `-o`)

## Sample run:

`python tedDataToPickle.py -w sample_ted/txt-sent/0001.word.txt -l sample_ted/txt-sent/0001.word.txt.norm.align  -f sample_ted/derived/segs/f0/0001.aggs.alignword.txt -i sample_ted/derived/segs/i0/0001.aggs.alignword.txt -o 0001.csv`

## Batch process talks:

`./processAllTedData.sh sample_ted sample_ted_eachtalk`

## Obtaining [*punkProse*](https://github.com/alpoktem/punkProse) processable corpus 
To collect samples from talks into one corpus partitioned into training/development/testing sets:

`python corpusMaker.py -i sample_ted_eachtalk -o sample_ted_punc_data -r 0.7 -v 1 -l 50`

(Training and development set are sampled into sequences of size 50 (`-l`). Training set constitutes 0.7 (`-r`) of all data. Word vocabulary is created with minimum word occurence 1 (`-v`).)






