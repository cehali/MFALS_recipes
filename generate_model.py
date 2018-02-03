import numpy as np
import pandas as pd
from numpy.linalg import solve
import pickle


def get_error(q, x, y, w):
    return np.sum((w * (q - np.dot(x, y.T))) ** 2)


def als(latent_vectors, fixed_vecs, ratings, lambda_, type):

    if type == 'user':
        for u, Wu in enumerate(W_train_final):
            latent_vectors[u] = solve(np.dot(fixed_vecs, np.dot(np.diag(Wu), fixed_vecs.T)) + lambda_ * np.eye(n_factors),
                                   np.dot(fixed_vecs, np.dot(np.diag(Wu), ratings[u].T))).T

    elif type == 'item':
        for i, Wi in enumerate(W_train_final.T):
            latent_vectors[i, :] = solve(np.dot(fixed_vecs.T, np.dot(np.diag(Wi), fixed_vecs)) + lambda_ * np.eye(n_factors),
                                   np.dot(fixed_vecs.T, np.dot(np.diag(Wi), ratings[:, i])))

    return latent_vectors

recipes = np.load('recipes_final.npy')
ratings = np.load('amazon_reviews_final.npy')

'''users = ratings[:, 0]
items = ratings[:, 1]

ratings_test = []
for user in users:
    user_train = []
    for rating in ratings:
        if user == rating[0]:
            user_train.append(list(rating))
    if len(user_train) > 1:
        random1 = random.randint(0, len(user_train)-1)        
        ratings_test.append(user_train[random1])
        random2 = random.randint(0, len(user_train)-1)
        ratings_test.append(user_train[random2])

file = open('ratings_test.txt', 'w')
for item in ratings_test:
        file.write('%s\n' % item)
file.close()'''

'''index = range(0, len(ratings))
data = {'user_id': ratings[:, 1], 'recipe_id': ratings[:, 0], 'rating': ratings[:, 3].astype(np.int)}
df = pd.DataFrame(data=data, index=index)

rp = df.pivot_table(values='rating', index=['user_id'], columns=['recipe_id'])
rp = rp.fillna(0)

Q = rp.values

W = Q > 0
W[W == True] = 1
W[W == False] = 0
W = W.astype(np.float64, copy=False)'''

fname_train = 'ratings_train.txt'
with open(fname_train, "rb") as fp_train:
    ratings_train = pickle.load(fp_train)

usr_train = []
rcp_train = []
rat_train = []
for rev_train in ratings_train:
        usr_train.append(rev_train[1])
        rcp_train.append(rev_train[0])
        rat_train.append(int(rev_train[3]))

index_train = range(0, len(ratings_train))
data_train = {'user_id': usr_train, 'recipe_id': rcp_train, 'rating': rat_train}
df_train = pd.DataFrame(data=data_train, index=index_train)

fname_test = 'ratings_test.txt'
with open(fname_test, "rb") as fp_test:
    ratings_test = pickle.load(fp_test)

usr_test_zeros = []
rcp_test_zeros = []
rat_test_zeros = []
for rev_test_zeros in ratings_test:
        usr_test_zeros.append(rev_test_zeros[1])
        rcp_test_zeros.append(rev_test_zeros[0])
        rat_test_zeros.append(0)

index_test = range(0, len(ratings_test))
data_test_zeros = {'user_id': usr_test_zeros, 'recipe_id': rcp_test_zeros, 'rating': rat_test_zeros}
df_test_zeros = pd.DataFrame(data=data_test_zeros, index=index_test)

df_train_final = df_train.append(df_test_zeros, ignore_index=True)
rp_train_final = df_train_final.pivot_table(values=['rating'], index=['user_id'], columns=['recipe_id'], fill_value=0,
                                            dropna=False)
rp_train_final = rp_train_final.fillna(0)

Q_train_final = rp_train_final.values

W_train_final = Q_train_final > 0
W_train_final[W_train_final == True] = 1
W_train_final[W_train_final == False] = 0
W_train_final = W_train_final.astype(np.float64, copy=False)

usr_test = []
rcp_test = []
rat_test = []
for rev_test in ratings_test:
        usr_test.append(rev_test[1])
        rcp_test.append(rev_test[0])
        rat_test.append(int(rev_test[3]))
        # rat_test.append(0)

index_test = range(0, len(ratings_test))
data_test = {'user_id': usr_test, 'recipe_id': rcp_test, 'rating': rat_test}
df_test = pd.DataFrame(data=data_test, index=index_test)

df_test_final = df_train.append(df_test, ignore_index=True)
rp_test_final = df_test_final.pivot_table(values=['rating'], index=['user_id'], columns=['recipe_id'], fill_value=0,
                                            dropna=False)
rp_test_final = rp_test_final.fillna(0)

Q_test_final = rp_test_final.values

W_test_final = Q_test_final > 0
W_test_final[W_test_final == True] = 1
W_test_final[W_test_final == False] = 0
W_test_final = W_test_final.astype(np.float64, copy=False)


for k in range(10, 15):

    n_factors = 10*k
    n_users, n_items = Q_train_final.shape
    n_iterations = 50
    user_reg = 0.001
    item_reg = 0.001

    user_vecs = np.random.random((n_users, n_factors))
    item_vecs = np.random.random((n_items, n_factors))

    errors_train = []
    errors_test = []

    ctr = 1
    while ctr <= n_iterations:
        print '\tcurrent iteration: {}'.format(ctr)
        user_vecs = als(user_vecs, item_vecs.T, Q_train_final, user_reg, type='user')
        item_vecs = als(item_vecs, user_vecs, Q_train_final, item_reg, type='item')
        ctr += 1
        error_train = get_error(Q_train_final, user_vecs, item_vecs, W_train_final)
        print error_train
        errors_train.append(error_train)
        error_test = get_error(Q_test_final, user_vecs, item_vecs, W_test_final)
        print error_test
        errors_test.append(error_test)

    np.save('model_users_' + str(k), user_vecs)
    np.save('model_items_' + str(k), item_vecs)
    np.savetxt('errors_train_0.01' + str(k), errors_train)
    np.savetxt('errors_test_0.01' + str(k), errors_test)


