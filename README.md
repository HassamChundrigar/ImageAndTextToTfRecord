# ImageAndTextToTfRecord
This Program Converts Images and their Corresponding text to TfRecord and Reloads it to program

## Files Descriptions

### Tf-Record.ipynb:

Python Program to convert Images and Corresponding texts to TfRecord, Images and texts should be in the same derectory. It makes tf-Record file with 60% Train, 20% Validation and 20% test set. Change manually to alter the ratio. Text is first converted into dense vector. The 0th index remains left and the indexing starts with 1

### utils.py:
Some useful Methods

### chars.txt:
Your Character set
