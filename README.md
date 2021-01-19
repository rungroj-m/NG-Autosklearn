# NG-Autosklearn
NG-Autosklearn: A tool set of software document classification with Auto-sklearn and N-gram IDF

## Requirements

- auto-sklearn>=0.12.0
- pandas>=1.0
- numpy>=1.9.0
- spacy>=2.3.0
- scikit-learn>=0.24.0,<0.25.0
- scipy>=0.14.1
- ngweight (https://github.com/iwnsew/ngweight)

## Install
- Install all library included Auto-sklearn: pip install -r requirements.txt
- Download Ngweight: git clone https://github.com/iwnsew/ngweight
- Download english model for lemmatization: python -m spacy download en_core_web_md

## Usage

See this file: 
- [example/usage.ipynb](example/usage.ipynb)
- [example/usage.py](example/usage.py)

## Reference

- Shirakawa, Masumi, Takahiro Hara, and Shojiro Nishio. "N-gram idf: A global term weighting scheme based on information distance." proceedings of the 24th international conference on World Wide Web. 2015. https://dl.acm.org/doi/abs/10.1145/2736277.2741628
- Feurer, Matthias, et al. "Auto-sklearn: efficient and robust automated machine learning." Automated Machine Learning. Springer, Cham, 2019. 113-134. http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf

## Studies using this tool

- Wattanakriengkrai, Supatsara, et al. "Identifying design and requirement self-admitted technical debt using n-gram idf." 2018 9th International Workshop on Empirical Software Engineering in Practice (IWESEP). IEEE, 2018. https://ieeexplore.ieee.org/abstract/document/8661216
- Maipradit, Rungroj, Hideaki Hata, and Kenichi Matsumoto. "Sentiment Classification Using N-Gram Inverse Document Frequency and Automated Machine Learning." IEEE Software 36.5 (2019): 65-70. https://ieeexplore.ieee.org/abstract/document/8725481
- Maipradit, Rungroj, et al. "Wait for it: identifying “On-Hold” self-admitted technical debt." Empirical Software Engineering 25.5 (2020): 3770-3798. https://doi.org/10.1007/s10664-020-09854-3
- Maipradit, Rungroj, et al. "Automated Identification of On-hold Self-admitted Technical Debt." 2020 IEEE 20th International Working Conference on Source Code Analysis and Manipulation (SCAM). IEEE, 2020. https://ieeexplore.ieee.org/abstract/document/9252045
