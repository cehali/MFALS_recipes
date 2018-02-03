import heapq
import numpy as np
import pandas as pd


def get_error(q, x, y, w):
    return np.sum((w * (q - np.dot(x, y.T))) ** 2)


def generate_recommendation(user_index):

    ratings_full = pd.read_csv('ratings_full')
    recipes = np.load('recipes_final.npy')
    ratings_values = np.load('ratings_values.npy')
    mask_with_ratings = np.load('mask_with_ratings.npy')

    users = np.load('model_users_3.npy')
    items = np.load('model_items_3.npy')

    rating_predictions = np.dot(users, items.T)
    for x in np.nditer(rating_predictions, op_flags=['readwrite']):
        x[...] = round(x * 2) / 2

    '''rating = np.load('amazon_reviews_final.npy')

    index = range(0, len(rating))
    data = {'user_id': rating[:, 1], 'recipe_id': rating[:, 0], 'rating': rating[:, 3].astype(np.int)}
    df = pd.DataFrame(data=data, index=index)

    rp = df.pivot_table(values='rating', index=['user_id'], columns=['recipe_id'])
    rp = rp.fillna(0)

    Q = rp.values

    error = rating_predictions * mask_with_ratings - Q
    print np.sum(error)'''

    mask_with_predictions = np.ones((len(mask_with_ratings), len(mask_with_ratings[0]))) - mask_with_ratings

    rating_predictions_final = rating_predictions * mask_with_predictions

    recipes_rated_index = heapq.nlargest(20, range(len(ratings_values[user_index])), ratings_values[user_index].take)
    recipes_recomd_index = heapq.nlargest(20, range(len(rating_predictions_final[user_index])),
                                        rating_predictions_final[user_index].take)

    recipes_rated_rate = ratings_values[user_index][[recipes_rated_index]]
    recipes_recom_rate = rating_predictions_final[user_index][[recipes_recomd_index]]

    recipes_rated_index = [x + 1 for x in recipes_rated_index]
    recipes_recomd_index = [x + 1 for x in recipes_recomd_index]

    recipes_rated_id = ratings_full.columns[[recipes_rated_index]]
    recipes_recom_id = ratings_full.columns[[recipes_recomd_index]]

    recipes_rated = []
    recipes_recom = []
    for mov1, mov2 in zip(recipes_rated_id, recipes_recom_id):
        for recipe in recipes:
            if mov1 == recipe.get('amazon_id'):
                recipes_rated.append(recipe)
            if mov2 == recipe.get('amazon_id'):
                recipes_recom.append(recipe)

    print ('Recipes you liked:')
    for i in range(0, len(recipes_rated)):
        print recipes_rated[i].get('title'), recipes_rated_rate[i]

    print ('\nRecipes we recommend:')
    for i in range(0, len(recipes_recom)):
        print recipes_recom[i].get('title'), recipes_recom_rate[i]


generate_recommendation(20)

