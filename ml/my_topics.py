import re

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,\
    ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import os
import pandas as pd


class Lda_model:
    #SCRIPTS INDEXED SAME AS RATINGS MATRIX

    def __init__(self, n_topics, n_features, file_with_scripts):
        t0 = time()
        print('Reading scripts...')
        self.scripts = pd.read_csv(file_with_scripts, nrows=10000).drop('Unnamed: 0', axis = 1)
        #print('self.scripts', self.scripts)
        print('Scripts read in {}s.'.format(time()-t0))
        self.movieIds = self.scripts['movieId']
        stop_words = self.get_stop_words()
        self.freqs = self.transform_scripts(n_features, stop_words)
        self.model = self.load_model(self.freqs, n_topics, n_features)

    def get_stop_words(self):
        f = open('names.txt')
        a = f.read()
        b = [x.strip() for x in a.split(',')]
        stop_words = ENGLISH_STOP_WORDS.union(b)
        return stop_words


    def print_top_words(self, feature_names, n_top_words):
        '''
        This is a helper function that simply prints out the topics, as represented
        by a list of n_top_words (usually 20) strings.
        '''
        for topic_idx, topic in enumerate(self.model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def print_all_topics(self, n_top_words):
        print('\nTopics in LDA model:')
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words)

    def transform_scripts(self, n_features, stop_words):

        t0 = time()
        print('Filtering data...') # Remove movie titles, they are not docs
        script_data = self.scripts['script']
        print('Data filtered in {0:.3f}s.'.format(time()-t0))

        print('Extracting tf features for LDA... ')
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words=stop_words)
        t0 = time()
        tf = tf_vectorizer.fit_transform(script_data)
        tf.index = self.scripts['movieId']
        print('done in {0:.3f}s.'.format(time() - t0))
        return tf


    def load_model(self, tf, n_topics, n_features):
        '''
        1. First we add given names to the stop words. (line 50)
        2. We then extract only the scripts, not the movie titles.
            The assumption is no script is under 100, no title over 100. (line 54)
        3. Set up vectorizer with appropriate parameters. (line 58)
        4. Use vectorizer to learn vocab, returning document-term matrix. (line 62)
        5. Set up LDA modeller with appropriate parameters. (line 68)
        6. Use LDA modeller to fit LDA model on term-document matrix: (line 73)
            a. First the t-d matrix is decomposed into "document-term" and
                "topic-term" matrices.
            b. Topic-term matrix is stored as components_.
            c. Document-topic matrix can be calculated by running:
                lda.transform(tf)
        7. Print stuff (lines 77-8)
        '''
         # Expand stop words

        print('Fitting LDA models with tf features,', \
              'n_topics={} and n_features={}...'\
              .format(n_topics, n_features))
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        t0 = time()
        lda.fit(tf)
        print('done in {0:.3f}s.'.format(time() - t0))
        return lda

    def get_topic_distribution(self, topic_num = None):
        print('Learning Topic Parameters...')
        t0 = time()
        if topic_num:
            topic_dist = self.model.transform(self.freqs[topic_num])
        else: 
            topic_dist = self.model.transform(self.freqs)
        print('Finished Topic model in {} secs:'.format(time() - t0))
        topic_dist_df = pd.DataFrame(topic_dist, index=self.movieIds)
        return topic_dist_df


































