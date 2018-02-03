import numpy as np
import collections
import pandas as pd

#recipes = np.load('/home/cezary/PycharmProjects/build_dataset/recipes_final.npy')
#rating = np.load('/home/cezary/PycharmProjects/build_dataset/amazon_food_list.npy')

#rating = rating[0:10000]

#index = range(0, len(rating))
#data = {'user_id': rating[:, 1], 'recipe_id': rating[:, 0], 'rating': rating[:, 3].astype(np.int)}
#df = pd.DataFrame(data=data, index=index)

#rp = df.pivot_table(values='rating', index=['user_id'], columns=['recipe_id'])

#print df

#print len(set(rating[:, 1].tolist()))
#print len(set(rating[:, 0].tolist()))

'''us = rating[:, 1]

cnt = collections.Counter(us).most_common(1560)

amazon_food_list_final = []
for usr in cnt:
    for rat in rating:
        if rat[1] == usr[0]:
            amazon_food_list_final.append(rat)

np.save('amazon_reviews_final', amazon_food_list_final)'''

'''a_food = food.values.tolist()

amazon_food = []
for fo in a_food:
    for pro in products:
        if fo[0] == pro:
            amazon_food.append(fo)
np.save('amazon_food_list', amazon_food)'''

recipes = np.load('recipes_final.npy')
rating = np.load('amazon_reviews_final.npy')

'''# rating = rating[0:10000]

index = range(0, len(rating))
data = {'user_id': rating[:, 1], 'recipe_id': rating[:, 0], 'rating': rating[:, 3].astype(np.int)}
df = pd.DataFrame(data=data, index=index)

rp = df.pivot_table(0, index=['user_id'], columns=['recipe_id'])
rp = rp.fillna(0)

Q = rp.values

W = Q > 0
W[W == True] = 1
W[W == False] = 0
W = W.astype(np.float64, copy=False)'''

'''rp.to_csv('ratings_full')
np.save('ratings_values', Q)
np.save('mask_with_ratings', W)'''

print len(list(set(rating[:, 1])))