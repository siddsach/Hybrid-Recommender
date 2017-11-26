# Hybrid-Recommender

This repository contains all the relevant files for my project building a Collaborative Topic Regression recommender system based on
this paper: http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf

Results
=====
![alt text](/images/rec_results.png?raw=true)

Vanilla PMF --> Standard Probabilistic Matrix Factorization Recommendation Algorithm
CTR-in-Matrix --> predictions for movies we have ratings for
CTR-out-of-Matrix --> predictions for movies we have no ratings for and just use script.
CTR-LDA-in-Matrix --> predictions for movies with ratings where we use LDA vector in place of learned ratings vector. 

Learned Representaitons
=====
![alt text](/images/lda_vis.png?raw=true "Visualization of Movievecs")

The output of LDA (an unsupervised algorithm to learn topic representation) gives word distributions for each topic, and inspecting these topics can give us a sense of what the algorithm is actually learning. Here are a few of the topics clearly signifying some easily interpretable likely meaning as represented by their top 20 words.

* Education → teachers library hawk gentlemen science robinson teaching thank campus study education teach sir university student teacher students college class school

* Film → movies scene stuff feel picture started life camera work big real gonna came things wanted story lot didn kind film

* Christmas → turkey toy bells jingle toys presents mo marley lll vic walt la ho claus year poppy tree lm snow 0000 Music → hear doo stage piano gonna dance night songs guitar record hey playing singing rock play yeah band sing la do

* Law → believe attorney question prison justice murder jury truth state witness years evidence lawyer trial guilty honor didn law judge case

* Military → fight order stand ship soldiers gentlemen enemy aye orders officer soldier thank lieutenant sergeant army colonel general war men captain


Data
=====
names.txt --> a list of names to be included in stopwords

ratings-100k.csv --> a dataset of ratings from movielens, file too big for repository, can be found here: https://grouplens.org/datasets/movielens/100k/

matched1.csv --> a dataset of scripts ID'd by the same ids as in movielens, file too big for repository, can be found here:
https://drive.google.com/file/d/0B-Zg2Odn-W_waWRFYkpkNDZHclE/view?usp=sharing


Code
=====
* final_model.py-->actual recommender algorithm
* my_topics.py-->applying Latent Dirichlet Allocation Topic Model to the Scripts dataset
* trail_recommender1.py-->optimizing hyperparameters and generating some visualizations
* scrape_ss.py -->scraping scripts from springfield script dataset
* linkage.py -->matching scripts to movie ids in the MovieLens 100k ratings dataset
