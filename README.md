# A Corpus Reader and Part-of-Speech Taggers for the MULTEXT-EAST corpus based on NLTK and scikit-learn

This repository contains all resources we developed for integrating the MULTEXT-EAST corpus into NLTK.
Additionally some POS-Taggers can be found. For an overview over the capabilities of our work have a look
at the file ```multext_nltk.pdf```

The raw data we generated in our evaluation can be found in the csv-files in the ```results``` directory, whereas in the ```csvtools``` directory the scripts for post-processing the csvs are located.

The file ```mte.py``` is the actual corpus reader, it is integrated into NLTK in exactly the same version. The ```MTEDownloader.py``` is an alternative to the NLTK downloader, which can also be used to retrieve the corpus.
