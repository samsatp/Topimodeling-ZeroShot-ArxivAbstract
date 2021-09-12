# Topic modelling with zero-shot classification (POC)
This is the NLP project in which I'm trying to achieve the text classification task. The main difference from [Topicmodelling v1](https://github.com/samsatp/topicmodeling) is the *method* and *data* used to train the model which will be discussed in detail in the following section. This project is not 100% perfect. *The main purpose of it is to prove the concept of zero-shot learning implementation in Tensorflow (POC)*

## Method
The model will be building by employ *Zero-shot learning* idea. The model architecture can be illustrated in the figure below. The data used to train the model can be found at https://www.kaggle.com/Cornell-University/arxiv It's a repository storing research papers in many fields.

![](./media/model.drawio.png)

## Usage
- `model.py` contains the model's implementation
- `inference.py` is the main file used to predict the task
```bash
usage: inference.py [-h] [-a ABSTRACT] [-c CANDIDATES]

optional arguments:
  -h, --help            show this help message and exit
  -a ABSTRACT, --abstract ABSTRACT
                        A raw query string
  -c CANDIDATES, --candidates CANDIDATES
                        Candidates of categories, separates each candidate with '<sep>'
```
> NOTE: use `\n` to separate each sentence in document specified in `--abstract` 

Example input: `python inference.py --abstract="some document in a long string." --candidates="computer science<sep>economics<sep>geology"`

Example output: `[[0.2603415], [0.6280834], [0.1115749]]`