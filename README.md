# WORDS N FUN : Semantic analysis module built by agence Data Services

---

## Why ?

The purpose of this project is two folds:
1. To normalize tools and how tos of semantic analysis projects
2. To offer end to end pipelines to speed up the time to market of NLP services


---

## Philosophy of this package

This package used to contain two main parts:
1. Preprocessing : to speed up the application of numerous preprocessing routines to text data
2. Modelling : to speed up the developpement of NLP machine learning models
The later has been refactored into what is now called the NLP framework.

### Preprocessing

One of the main goals of the project was to build a pipeline that could accomodate several data types:
- str
- list
- np.array
- pd.Series
- pd.DataFrame
- csv file (where the supplied data is the path to the actual file)

Whatever the supplied data type, it will kept along the pipeline, meaning that if a csv file is given, a new csv file will be created as the output. If the input is a pd.Series, another pd.Series of the same shape will be returned.
This feature is controled by the utils.data_agnostic function.

#### Main scripts

##### utils.py

Contains utils functions and tools, mainly the data agnostic feature.

##### basic.py

Contains the main data transformation functions.

##### api.py

Contains the main entry point of the package controlling how one could create an end to end transformation pipeline.


---

## Getting started

### Requirements & setup

**Coming soon**: CI pipeline to push this package to pypi

- Python 3.X (tests are ran on python 3.8)
- Set up module dependencies : `pip install -r requirements.txt`
- Set up the actual module : `pip install -e words_n_fun` or `python setup.py develop`
- Download nltk stopwords data:
```python
import nltk
nltk.download('stopwords')
```


### How to use this module

A notebook tutorial is provided in the tutorial directory.
Roughly speaking, her is what a basic use case should look like:

```python
 # Module import
from words_n_fun.preprocessing import api

# Definition of the desired transformation pipeline
pipeline = ['remove_non_string', 'get_true_spaces', 'to_lower', 'remove_punct',
			'remove_numeric', 'remove_stopwords', 'lemmatize', 'remove_accents',
			'trim_string', 'remove_leading_and_ending_spaces']

#### example 1 : csv file ####

# Input file
input_file = "path/to/my/file.csv"
# Column containing the text to preprocess
col = "myText"
# Separator character
sep = ';'
# Instanciation of a preprocessor object
preprocessor = api.get_preprocessor(pipeline=pipeline, prefered_column=col, sep=sep)
# Process data
output_file = preprocessor.transform(input_file)


#### example 2 : list  ####
# Input data
input_list = ["First text to transform", "Second text to pocess"]
# Instanciation of a preprocessor object
preprocessor = api.get_preprocessor(pipeline=pipeline)
# Process data
output_list = preprocessor.transform(input_list)

```

---

## Best Practices & guidelines

Contributors must try their best to follow these mainstream guidelines :
1. Module management https://docs.python.org/fr/2/tutorial/modules.html
2. Python development https://github.com/google/styleguide/blob/gh-pages/pyguide.md, https://www.python.org/dev/peps/pep-0008/
3. All the preprocessing functions must take pd.Series data as input. The built-in utils.data_agnostic decorator allows the user to send different types of data but the actual function carrying out these transformation should be built around pd.Series.

Numerous unit tests have been included in `tests`:
- Please make sure to run them before trying to submit a merge request
- New features must be shipped with their corresponding tests

---
