from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_similar_recipes(recipe_id):

    recipes = pd.read_csv('/home/cezary/Dataset-recpies+ratings/epi_r_Epicurious.csv')

    recipes = recipes.drop('rating', 1)
    recipes = recipes.drop('calories', 1)
    recipes = recipes.drop('protein', 1)
    recipes = recipes.drop('fat', 1)
    recipes = recipes.drop('sodium', 1)
    recipes.set_index('title', inplace=True)

    recipes = recipes.iloc[0:15000]
    recipes_values = recipes.values

    recipes_similarities = cosine_similarity(recipes_values)
    recipes_similarities = pd.DataFrame(recipes_similarities, index=recipes.T.columns, columns=recipes.T.columns)

    print recipes_similarities.T[recipes_similarities.columns[recipe_id]].nlargest(6).index.tolist()

    # visualize similarity of recipes
    '''standard_scalar = StandardScaler()

    x_std = standard_scalar.fit_transform(recipes_values)

    tsne = TSNE(n_components=2, random_state=0)
    x_test = tsne.fit_transform(x_std)

    plt.scatter(x_test[:, 0], x_test[:, 1])
    plt.show()

    model = manifold.TSNE()
    Y = model.fit_transform(recipes_values)

    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()'''


get_similar_recipes(2)
