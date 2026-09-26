"""Product recommendation logic using scikit-learn.

Steps:
1. create_data()    -> makes a product list and purchases of 1,000 customers
2. build_dataset()  -> converts purchases into X (inputs) and y (answers)
3. compare_models() -> tunes 5 models with GridSearchCV and picks the best one
4. recommend()      -> suggests products using the best model

Run `python recommender.py` to see the model comparison in the terminal.
"""

import random

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------
# 1. Creating the data
# ---------------------------------------------------------------

# category -> products in that category
catalog = {
    "Cricket": ["Cricket Bat", "Cricket Ball", "Stumps", "Cricket Helmet", "Batting Gloves",
                "Batting Pads", "Cricket Kit Bag"],
    "Football": ["Football", "Football Boots", "Shin Guards", "Goalkeeper Gloves",
                 "Football Jersey", "Ball Pump"],
    "Badminton": ["Badminton Racket", "Shuttlecocks", "Badminton Shoes", "Badminton Net",
                  "Grip Tape"],
    "Fitness": ["Dumbbells", "Yoga Mat", "Skipping Rope", "Resistance Bands", "Gym Gloves",
                "Protein Powder", "Shaker Bottle"],
    "Running": ["Running Shoes", "Running Shorts", "Fitness Band", "Water Bottle", "Sports Socks"],
    "Mobile": ["Smartphone", "Phone Case", "Screen Guard", "Earphones", "Power Bank", "Charger"],
    "Laptop": ["Laptop", "Laptop Bag", "Mouse", "Keyboard", "Laptop Stand", "Pen Drive"],
    "Kitchen": ["Frying Pan", "Pressure Cooker", "Knife Set", "Chopping Board", "Mixer Grinder",
                "Storage Containers"],
    "Stationery": ["Notebook", "Pens", "Highlighters", "Geometry Box", "Calculator", "School Bag"],
    "Camping": ["Tent", "Sleeping Bag", "Torch", "Trekking Backpack", "Camping Stove"],
}

# category -> another category its customers often buy from
related = {
    "Cricket": "Fitness", "Football": "Running", "Badminton": "Fitness", "Fitness": "Running",
    "Running": "Fitness", "Mobile": "Laptop", "Laptop": "Mobile", "Kitchen": "Stationery",
    "Stationery": "Laptop", "Camping": "Running",
}


# ---------------------------------------------------------------------------
# Step 1: Create the dataset
# ---------------------------------------------------------------------------


def create_data():
    random.seed(7)  # same data every time the program runs

    # Product table: product, category, price
    product_rows = []
    for category in catalog:
        for product in catalog[category]:
            price = random.randint(2, 59) * 100
            product_rows.append([product, category, price])

    products = pd.DataFrame(product_rows, columns=["product", "category", "price"])

    # Purchase table: customer_id, product
    purchase_rows = []
    for customer_id in range(1, 1001):
        # every customer mainly shops in one category
        main_category = random.choice(list(catalog))
        count = random.randint(2, 5)
        bought = random.sample(catalog[main_category], count)

        # half of the customers also buy from a related category
        if random.random() < 0.5:
            other_category = related[main_category]
            count = random.randint(1, 2)
            bought = bought + random.sample(catalog[other_category], count)

        for product in bought:
            purchase_rows.append([customer_id, product])

    purchases = pd.DataFrame(purchase_rows, columns=["customer_id", "product"])

    return products, purchases


# ---------------------------------------------------------------
# 2. Building X and y
# ---------------------------------------------------------------

def build_dataset(products, purchases):
    """For every product a customer bought, hide it and use the rest as input.

    X -> one column per product, 1 = bought, 0 = not bought
    y -> the hidden product the model has to predict
    """
    all_products = list(products["product"])

    X = []
    y = []

    for customer_id in purchases["customer_id"].unique():
        basket = list(purchases[purchases["customer_id"] == customer_id]["product"])

        for hidden_product in basket:
            row = []
            for product in all_products:
                if product in basket and product != hidden_product:
                    row.append(1)
                else:
                    row.append(0)

            X.append(row)
            y.append(hidden_product)

    X = pd.DataFrame(X, columns=all_products)
    y = pd.Series(y)

    return X, y


# ---------------------------------------------------------------
# 3. Comparing models
# ---------------------------------------------------------------

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": BernoulliNB(),
}

# hyperparameter values that GridSearchCV will try for each model
param_grids = {
    "K-Nearest Neighbors": {"n_neighbors": [5, 15, 30], "metric": ["cosine", "euclidean"]},
    "Decision Tree": {"max_depth": [10, 20, None], "min_samples_leaf": [1, 5]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [10, None]},
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
}


def top5_score(model, X, y):
    """Top-5 accuracy: is the hidden product among the model's 5 best guesses?"""
    probabilities = model.predict_proba(X)
    return top_k_accuracy_score(y, probabilities, k=5, labels=model.classes_)


def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    best_model = None
    best_score = 0

    for name in models:
        # try every combination of hyperparameters with 3-fold cross validation
        grid = GridSearchCV(models[name], param_grids[name], cv=3, scoring=top5_score)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        # check the tuned model on the test data
        top5 = top5_score(model, X_test, y_test)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        results.append([name, str(grid.best_params_), round(top5, 3), round(accuracy, 3)])

        if top5 > best_score:
            best_score = top5
            best_model = model

    results = pd.DataFrame(results, columns=["Model", "Best parameters", "Top-5 accuracy", "Accuracy"])
    results = results.sort_values(by="Top-5 accuracy", ascending=False)
    results = results.reset_index(drop=True)

    return results, best_model


# ---------------------------------------------------------------
# 4. Recommending products
# ---------------------------------------------------------------

def recommend(model, all_products, bought, k=5):
    # convert the user's products into the same 0/1 format as X
    row = []
    for product in all_products:
        if product in bought:
            row.append(1)
        else:
            row.append(0)

    row = pd.DataFrame([row], columns=all_products)

    # probability of each product being the next purchase
    probabilities = model.predict_proba(row)[0]
    scores = pd.DataFrame({"product": model.classes_, "score": probabilities})

    # remove products the user already has, keep the top k
    scores = scores[scores["product"].isin(bought) == False]
    scores = scores.sort_values(by="score", ascending=False)

    return scores.head(k)


if __name__ == "__main__":
    products, purchases = create_data()
    X, y = build_dataset(products, purchases)
    print("Products:", len(products))
    print("Purchases:", len(purchases))
    print("Training rows:", len(X))
    print()

    results, best_model = compare_models(X, y)
    print(results.to_string(index=False))
    print()
    print("Best model:", results["Model"][0])