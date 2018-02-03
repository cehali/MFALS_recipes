import random
import numpy as np
import pandas as pd


ratings = np.load('amazon_reviews_final.npy')
users = ratings[:, 0]
print users

users_train = []
users_test = []

'''for user in users:
    user_train = []
    user_test = []
    for rating in ratings:
        if user == rating[0]:
            user_train.append(rating)
    if len(user_train) > 1:
        random1 = random.randint(0, len(user_train)-1)
        user_test.append(user_train[random1])
        random2 = random.randint(0, len(user_train)-1)
        user_test.append(user_train[random2])

    users_train.append(users_train)
    users_test.append(users_test)'''

'''users_train.append(list(ratings[0]))
users_train.append(list(ratings[1]))
rat = []
for u in users_train:
    rat.append(int(u[3]))
print rat
print users_train'''

file = open('test.txt', 'r')
test = file.read()
print list(test)

