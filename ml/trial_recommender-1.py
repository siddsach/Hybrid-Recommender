import final_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


#Graphing Lambda U
train_errors = []
test_in_errors = []
test_out_errors = []
lda_in_errors = []

lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000]

for i in range(len(lambdas)):
	model = final_model.CollaborativeTopicModel(lamu = lambdas[i])
	train_errors.append(model.train_error)
	test_in_errors.append(model.test_in_error)
	test_out_errors.append(model.test_out_error)
	lda_in_errors.append(model.lda_error)

plt.figure()
plt.plot(np.log(lambdas), train_errors, color = 'b')
plt.plot(np.log(lambdas), test_in_errors, color = 'r')
plt.plot(np.log(lambdas), test_out_errors, color = 'g')
plt.plot(np.log(lambdas), lda_in_errors, color = 'purple')
plt.xlabel('Lambda U')
plt.xticks(np.log(lambdas), lambdas)
plt.ylabel('RMSE')
plt.title('Lamda U vs. Errors')
plt.legend(['Train', 'Test In-Matrix', 'Test Out-of-Matrix', 'CTR-LDA In-Matrix'])
plt.savefig('lambdautests.png')

#Graphing Lambda V
train_errors = []
test_in_errors = []
test_out_errors = []
lda_in_errors = []

lambdas = [10, 100, 1000, 5000, 10000, 50000, 100000, 500000]

for i in range(len(lambdas)):
	model = final_model.CollaborativeTopicModel(lamv = lambdas[i])
	train_errors.append(model.train_error)
	test_in_errors.append(model.test_in_error)
	test_out_errors.append(model.test_out_error)
	lda_in_errors.append(model.lda_error)

plt.figure()
plt.plot(np.log(lambdas), train_errors, color = 'b')
plt.plot(np.log(lambdas), test_in_errors, color = 'r')
plt.plot(np.log(lambdas), test_out_errors, color = 'g')
plt.plot(np.log(lambdas), lda_in_errors, color = 'p')
plt.xlabel('Lambda V')
plt.xticks(np.log(lambdas), lambdas, fontsize = 'xx-small')
plt.ylabel('RMSE')
plt.title('Lamda V vs. Errors')
plt.legend(['Train', 'Test In-Matrix', 'Test Out-of-Matrix', 'CTR-LDA In-Matrix'])
plt.savefig('lambdavtests.png')


#Graphing nullvals
train_errors = []
test_in_errors = []
test_out_errors = []
lda_in_errors = []

nullvals = np.arange(12)/2

for i in nullvals:
	model = final_model.CollaborativeTopicModel(nullval = i)
	train_errors.append(model.train_error)
	test_in_errors.append(model.test_in_error)
	test_out_errors.append(model.test_out_error)
	lda_in_errors.append(model.lda_error)


ind = np.arange(len(nullvals))
width = 0.2
fig, ax = plt.subplots()
ax.bar(ind, train_errors, width, color = 'b')
ax.bar(ind + width, test_in_errors, width, color = 'r')
ax.bar(ind + 2 * width, test_out_errors, width, color = 'g')
ax.bar(ind + 3 * width, lda_in_errors, width, color = 'p')
ax.set_xlabel('Null Value')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(nullvals)
ax.set_ylabel('RMSE')
ax.set_title('Null Value vs. Errors')
ax.legend(['Train', 'Test In-Matrix', 'Test Out-of-Matrix'])
plt.savefig('Nullvalstests.png')


#Graphing Confidence values
train_errors = []
test_in_errors = []
test_out_errors = []
lda_in_errors = []

for i in range(6):
	model = final_model.CollaborativeTopicModel(params = i)
	train_errors.append(model.train_error)
	test_in_errors.append(model.test_in_error)
	test_out_errors.append(model.test_out_error)
	lda_in_errors.append(model.lda_error)


ind = np.arange(6)
width = 0.2
fig, ax = plt.subplots()
ax.bar(ind, train_errors, width, color = 'b')
ax.bar(ind + width, test_in_errors, width, color = 'r')
ax.bar(ind + 2 * width, test_out_errors, width, color = 'g')
ax.bar(ind + 3 * width, lda_in_errors, width, color = 'p')
ax.set_xticks(ind + width / 2)
pairs = [('1/sigma^2', '0'), ('1', '0'), ('1/sigma^2', '0.01'), ('1', '0.01'), ('1/sigma^2', '1/sigma^2'), ('5', '0')]
ax.set_xticklabels(pairs, size = 'xx-small')
ax.set_ylabel('RMSE')
ax.set_title('Confidence Parameters vs. Errors')
ax.legend(['Train', 'Test In-Matrix', 'Test Out-of-Matrix', 'CTR-LDA In-Matrix'])
plt.savefig('ConfParamsTest.png')


#Graphing Vocab sizes
train_errors = []
test_in_errors = []
test_out_errors = []
lda_in_errors = []
sizes = [100, 500, 1000, 3000, 5000, 10000]
for size in sizes:
	model = CollaborativeTopicModel(saved = False, n_voca = size)
	train_errors.append(model.train_error)
	test_in_errors.append(model.test_in_error)
	test_out_errors.append(model.test_out_error)
	lda_in_errors.append(model.lda_error)

plt.figure()
plt.plot(sizes, train_errors, color = 'b')
plt.plot(sizes, test_in_errors, color = 'r')
plt.plot(sizes, test_out_errors, color = 'g')
plt.plot(sizes, lda_in_errors, color = 'p')
plt.xlabel('Number of Considered Words in model')
plt.xticks(sizes)
plt.ylabel('RMSE')
plt.title('Vocabulary Size vs. Errors')
plt.legend(['Train', 'Test In-Matrix', 'Test Out-of-Matrix', 'CTR-LDA In-Matrix'])
plt.savefig('vocabtests.png')


#Graphing Number of Topics
train_errors = []
test_in_errors = []
test_out_errors = []
lda_in_errors = []
ks = range(5, 105, 5)
for k in ks:
	model = CollaborativeTopicModel(saved = False, n_topic = k)
	train_errors.append(model.train_error)
	test_in_errors.append(model.test_in_error)
	test_out_errors.append(model.test_out_error)
	lda_in_errors.append(model.lda_error)

plt.figure()
plt.plot(sizes, train_errors, color = 'b')
plt.plot(sizes, test_in_errors, color = 'r')
plt.plot(sizes, test_out_errors, color = 'g')
plt.plot(sizes, lda_in_errors, color = 'black')
plt.xlabel('Number of Topics in model')
plt.xticks(sizes)
plt.ylabel('RMSE')
plt.title('N_Topics vs. Errors')
plt.legend(['Train', 'Test In-Matrix', 'Test Out-of-Matrix', 'CTR-LDA in-matrix'])
plt.savefig('topictests.png')

model = CollaborativeTopicModel()

tsne = TSNE()
tsne.fit(model.V.T)
tsneprojected = pd.DataFrame(tsne.transform(model.V.T))

plot.figure()

tsneprojected.columns = ['x', 'y']
tsneprojected.plot(kind='scatter', x='x', y='y')

for i, movieid in enumerate(model.V.columns):
	name = model.movienames[movieid]
	plot.annotate(name, tsneprojected[i], size = 'xx-small'

plot.savefig('latentmovies.png')


