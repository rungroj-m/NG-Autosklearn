import sys
import pandas as pd
sys.path.insert(0, '../code/')

from preprocessing import Preprocessing
from classification import Classification

if __name__ == "__main__":
        # Example of preprocessing usage
        df = pd.read_csv('dataset/Sample of removed SATD comments - RQ2.csv')
        prep = Preprocessing()

        # remove special character
        df['clean_comment'] = df['SATD comment'].apply(prep.special_character_removal)

        # apply lemmatization
        df['clean_comment'] = df['clean_comment'].apply(prep.lemmatization)
        df['On-hold or not'] = df['On-hold or not'].map(dict(yes=1, no=0))

        # create n-gram using ngweight
        ngweight_folder = 'path_to_ngweight/'
        n_gram = prep.create_n_gram(df['clean_comment'],df['On-hold or not'],ngweight_folder,'dataset/n_gram')

        df = df[['clean_comment','On-hold or not']]
        df.to_csv('dataset/clean_dataset.csv')

        # Example of ten-fold classification usage
        df = pd.read_csv('dataset/clean_dataset.csv')
        classification = Classification('onhold')

        # load n-gram and use it as corpus
        classification.set_n_gram("dataset/n_gram")

        # vectorize comment word frequency
        X = classification.vectorization(df['clean_comment'])
        y = df['On-hold or not']

        # 10-fold classification
        classification.ten_fold(X,y)
