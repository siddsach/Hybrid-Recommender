from time import time
import pandas as pd
import numpy as np
import numpy.linalg
import my_topics
import random
import pickle

class CollaborativeTopicModel:
    """
    Building hybrid recommender system based on the following paper:

    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.
    Arguments
    ----------
    n_topic: number of topics (hyperparameter)
    n_vica: size of vocabulary
    n_movie: int
        number of movie
    n_user: int
        number of movie
    nullval: number to use for 
    """

    def __init__(self, n_topic, n_voca, nullval, n_iter = 5, e = 1e-100, error_diff = 0.1, params = 3, ratingsfile = 'ratings_100K.csv', scriptsfile = "matched1.csv"):
        
        #User x Item ratings matrix
        print('Cleaning Ratings')
        self.R, self.R_test, self.movienames = self.get_ratings_matrix(scriptsfile, ratingsfile, 0.9)

        self.lda = my_topics.Lda_model(n_topic, n_voca, scriptsfile)
        #THETA MUST HAVE SAME ITEM INDEX AS RATINGS_MATRIX
        self.theta = self.lda.get_topic_distribution()
        #lambda_u = sigma_u^2 / sigma^2
        self.lambda_u = 0.01
        self.lambda_v = 100

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.n_user = len(self.R)
        self.n_item = len(self.R.iloc[0])
        self.threshold = error_diff
        self.nullval = nullval
        self.n_iter = n_iter
        if params == 0:
            self.params = [1/np.nanstd(self.R) ** 2, 0]
        elif params == 1:
            self.params = [1, 0]
        elif params == 2:
            self.params = [1/np.nanstd(self.R) ** 2, 0.1]
        elif params == 3:
            self.params = [1, 0.01]
        elif params == 4:
            self.params = 2 * [1/np.nanstd(self.R) ** 2]
        elif params == 5:
            self.params = [5, 0]
        
        # U = user_topic matrix, n_topic x n_user
        self.U = pd.DataFrame(np.random.multivariate_normal(np.zeros(self.n_user), np.identity(self.n_user) * (1. / self.lambda_u),size=self.n_topic))

        # V = item(doc)_topic matrix, n_topic x n_item
        self.V = pd.DataFrame(np.random.multivariate_normal(np.zeros(self.n_item), np.identity(self.n_item) * (1. / self.lambda_v), size = self.n_topic))


        self.V.columns = self.R.columns
        self.C = self.R.applymap(self.get_c)
        
        self.errors = [None] * self.n_iter
        self.train_error = None
        self.test_error = None

    def binary(self, val):
        if np.isnan(val):
            return 0
        elif val <= 2:
            return 0
        else:
            return 1
    def get_c(self, val):
        if np.isnan(val):
            return self.params[1]
        else:
            return self.params[0]

    @staticmethod
    def get_ratings_matrix(scriptsfile, ratingsfile, hyperparameter):
        print('Combining movie and item indexing...')
        # Reads in movies from the matched file
        df_movies = pd.read_csv(scriptsfile, usecols = ['movieId', 'title'], nrows=10000)
        #print('df movies', df_movies)
        # Drops duplicates
        df_movies = df_movies.drop_duplicates(subset='movieId')
        # Reads in ratings from the movie lens rating file
        df_ratings = pd.read_csv(ratingsfile, usecols=['userId', 'movieId', 'rating'], nrows=30000)
        # Joins ratings and movies from matched final using movieId
        df = df_movies.merge(df_ratings, on='movieId')

        print('Saving Names...')
        # Creates a dictionary mapping movieIds to movie titles
        movieName_dict = df_movies.set_index('movieId')['title'].to_dict()

        train = df.copy()
        test = df.copy()

        # Creates a set numbered from 1:len(rows of df)
        df_rows = set((range(df.shape[0])))

        print('Splitting Train and Test')
        # Creates a random sample that has a hyperparameter 
        test_rows = set(random.sample(df_rows, int(hyperparameter*len(df_rows))))


        # Splits data into training and test
        for i in test_rows:
            test.iat[i, 3] = np.nan

        train_rows = df_rows - test_rows
        for j in train_rows:
            train.iat[j, 3] = np.nan
        
        print('Constructing ratings matrix')
        # Creates the training matrix
        train = train.pivot(index='userId', columns='movieId', values='rating')

        # Takes a percentage of the movies to include in training
        # to test out-of-matrix prediction
        train=train.sample(frac=hyperparameter, axis=1)
        test = test.pivot(index='userId', columns='movieId', values='rating')

        return train, test, movieName_dict



    def fit(self):
        print("Learning U and V...\n")
        t0 = time()
        old_err = 0
        num_iter = 0
        for iteration in range(self.n_iter):
            print('Finished {} iterations'.format(num_iter))
            self.do_e_step()
            err = self.sqr_error()
            self.errors[num_iter] = err
            num_iter += 1
            print(old_err)
            '''
            if abs(old_err - err) < self.threshold:
                print('Error threshold reached!')
                break
            else:
                old_err = err
            '''
        print('Finished training with {} iterations in {} seconds'.format(num_iter, time() - t0))

        self.train_error = err
        self.test_error = self.sqr_error(True)
        print('Achieved mean training error of {}'.format(self.train_error))
        print('Achieved mean test error of {}'.format(self.test_error))

    # reconstructing matrix for prediction
    def predict_item(self, test=False):
        if test:
            missing_ids = self.R_test.columns.difference(self.R.columns)
            add_V = self.theta.ix[missing_ids].T
            old_V = self.V
            new_V = pd.concat([old_V, add_V], axis=1).as_matrix()
            return np.dot(self.U.T.as_matrix(), new_V)
        return np.dot(self.U.T.as_matrix(), self.V.as_matrix())

    # reconstruction error
    def sqr_error(self, test=False):
        ratings_matrix = self.R
        if test:
            ratings_matrix = self.R_test
        err = np.sqrt(np.nanmean((ratings_matrix - self.predict_item(test)) ** 2))
        return err

    def do_e_step(self):
        u = np.matrix(self.U)
        v = np.matrix(self.V)
        self.update_u(u,v)
        self.update_v(u,v)

    #THESE TWO UPDATE STEPS MUST BE CHANGED TO REFLECT NONBINARY RATINGS
    def update_u(self, u, v, hybrid = True):
        t0 = time()
        print('Updating Users')
        new_u = np.array(list(map(self.get_new_uservec, range(self.n_user), self.n_user * [u], self.n_user * [v]))).T
        print('Values in U changed on average by {}'.format(np.nanmean(new_u - self.U)))
        self.U.update(new_u)
        print('Finished Users in {}'.format(t0))

    def get_new_uservec(self, ui, u, v):
        #print('Updating User {}'.format(ui))
        print(self.U.shape)
        c_i = np.matrix(np.diag(self.C.iloc[ui]))
        r_i = np.copy(self.R.iloc[ui])
        r_i[np.isnan(r_i)] = self.nullval
        r_i = np.matrix(r_i).T
            
        left = v * c_i * v.T + (self.lambda_u * np.identity(self.n_topic))
        return np.array(numpy.linalg.solve(left, v * c_i * r_i)).flatten()

    def update_v(self, u, v, hybrid = True):
        t0 = time()
        print(self.V.shape)
        print('Updating Movies')
        new_v = np.array(list(map(self.get_new_movievec, self.R.columns, self.n_item * [u], self.n_item * [v])))
        #print('Values in V changed on average by {}'.format(np.nanmean(new_v - self.V)))
        self.V.update(new_v)
        print('Finished Movies in {}'.format(time() - t0))


    def get_new_movievec(self, vj, u, v):
        #print('Updating Movie: {}'.format(self.movienames[vj]))

        c_j = np.matrix(np.diag(self.R[vj]))
        r_j = np.copy(self.R[vj])
        r_j[np.isnan(r_j)] = self.nullval
        r_j = np.matrix(r_j).T

        theta_j = np.matrix(self.theta.ix[vj]).T
        left = u * c_j * u.T + (self.lambda_v * np.identity(self.n_topic))
        return np.array(np.linalg.solve(left, u * c_j * r_j + (self.lambda_v * theta_j))).flatten()

    #movie_ratings is a dictionarry with movieIDs for keys and ratings for values
    def add_user(self, movie_ratings):

        new_user_id = self.R.index.values.max() + 1
        self.R.loc[new_user_id] = self.nullval
        self.R_test.loc[new_user_id] = self.nullval

        for movieid in movie_ratings.keys():
            if movieid in self.R.columns.tolist():
                if movie_ratings[movieid] != '':
                    self.R.set_value(len(self.R)-1, movieid, movie_ratings[movieid])
                    self.R_test.set_value(len(self.R_test)-1, movieid, movie_ratings[movieid])


        self.C = self.R.applymap(self.get_c)
        print(self.C.shape)
        new_latent_vector = np.random.multivariate_normal(np.zeros(self.n_topic), np.identity(self.n_topic) * (1. / self.lambda_u))
        self.U[new_user_id] = new_latent_vector
        #self.U[:,:-1] = new_latent_vector
        self.n_user += 1
        self.fit()
        predictions_matrix = self.predict_item()
        predictions = pd.Series(predictions_matrix[predictions_matrix.shape[0]-1], index=self.V.columns)
        recs = predictions.sort_values(ascending=False)[:5]

        result = []
        for r in recs.index.values:
            result.append((r,self.movienames[r]))
        print(result)
        return result


# t_zero = time()
#dat_model_doe = CollaborativeTopicModel(n_topic=75, n_voca=10000, nullval=3.5)
#dat_model_doe.fit()
#dat_model_doe.add_user({48385: 3, 69122: 3, 48516: 3, 7318: 2, 1: 5, 364: 1.5, 2571: 3, 48780: '', 85774: 2.3, 527: '', 8464: '', 2710: 4.0, 919: '', 45722: '', 82459: '', 79132: '', 6942: '', 4896: '', 72998: '', 7153: '', 1704: '', 2858: '', 1968: '', 5299: '', 8376: '', 1721: '', 318: '', 296: '', 72641: '', 1732: '', 2502: '', 780: '', 589: '', 593: '', 71379: '', 33493: '', 72407: '', 356: '', 2918: '', 54503: 3, 6377: '', 2028: 3.5, 56174: 5, 111: 2.0, 1265: 5.0, 5618: '', 1270: 3.0, 58559: ''})
# pickle.dump(dat_model_doe, open("dat_model_doe.p", "wb"))

# print(time() - t_zero)
# print('done with small one')

# t_zero = time()
# the_biggest = CollaborativeTopicModel(n_topic=75, n_voca=10000, nullval=0, ratingsfile='data/ml-20m/ratings.csv', scriptsfile='py/final_matched.csv')

# pickle.dump(the_biggest, open("the_biggest.p", "wb"))

# print(time() - t_zero)
# print('done with big one also')
