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

- Shirakawa, Masumi, Takahiro Hara, and Shojiro Nishio. "N-gram idf: A global term weighting scheme based on information distance." proceedings of the 24th international conference on World Wide Web. 2015.
- Feurer, Matthias, et al. "Auto-sklearn: efficient and robust automated machine learning." Automated Machine Learning. Springer, Cham, 2019. 113-134.

## Studies using this tool

- R. Maipradit, H. Hata and K. Matsumoto, "Sentiment Classification Using N-Gram Inverse Document Frequency and Automated Machine Learning," in IEEE Software, vol. 36, no. 5, pp. 65-70, Sept.-Oct. 2019, doi: 10.1109/MS.2019.2919573.
- R. Maipradit, C. Treude, H. Hata and K. Matsumoto, "Wait for it: identifying “On-Hold” self-admitted technical debt", Empir Software Eng 25, 3770–3798 (2020). https://doi.org/10.1007/s10664-020-09854-3
- R. Maipradit et al., "Automated Identification of On-hold Self-admitted Technical Debt," 2020 IEEE 20th International Working Conference on Source Code Analysis and Manipulation (SCAM), Adelaide, Australia, 2020, pp. 54-64, doi: 10.1109/SCAM51674.2020.00011.
- S. Wattanakriengkrai, R. Maipradit, H. Hata, M. Choetkiertikul, T. Sunetnanta and K. Matsumoto, "Identifying Design and Requirement Self-Admitted Technical Debt Using N-gram IDF," 2018 9th International Workshop on Empirical Software Engineering in Practice (IWESEP), Nara, Japan, 2018, pp. 7-12, doi: 10.1109/IWESEP.2018.00010.
