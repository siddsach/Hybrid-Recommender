# Hybrid-Recommender

This repository contains all the relevant files for my project building a Collaborative Topic Regression recommender system.

Abstract:
Recommendation is a highly multifaceted task. For ideal performance, a recommender system must be able to effectively use multiple kinds of data to deliver high quality recommendations, avoid being biased toward items that have had many ratings, be able to understand new content, and be fast enough that the system is able to deliver recommendations in a reasonable timeframe. In this paper, I explore the application of the novel recommender algorithm Collaborative Topic Regression to the task of recommending movies using both ratings data and the scripts of the movies. This algorithm combines collaborative filtering and content analysis to make significant progress over other recommender algorithms in the objectives described above. With the advent of big data, big compute power, and online media, businesses that effectively apply the power of targeting can form the foundation of startups in virtually every B2C industry, and this algorithm certainly accomplishes that task.


DATA

names.txt --> a list of names to be included in stopwords

ratings-100k.csv --> a dataset of ratings from movielens, file too big for repository, can be found here: https://grouplens.org/datasets/movielens/100k/

matched1.csv --> a dataset of scripts ID'd by the same ids as in movielens, file too big for repository, can be found here:
https://drive.google.com/file/d/0B-Zg2Odn-W_waWRFYkpkNDZHclE/view?usp=sharing


CODE

final_model.py-->actual recommender algorithm

my_topics.py-->applying Latent Dirichlet Allocation Topic Model to the Scripts dataset

trail_recommender1.py-->optimizing hyperparameters and generating some visualizations

scrape_ss.py -->scraping scripts from springfield script dataset

linkage.py -->matching scripts to movie ids in the MovieLens 100k ratings dataset


REPORT

CTR-project-paper.pdf --> report on model and experiments
