"""Product recommendation logic (scikit-learn).

1. Data: a small product catalog and the purchase history of 1,000 customers,
   generated with a fixed random seed. Each customer shops in one or two
   related categories.
2. Recommendation as classification: for every customer, one purchased product
   is hidden. The model sees the other products they bought (as 0/1 columns)
   and has to predict the hidden one.
3. Five models are tuned with GridSearchCV and compared on top-5 accuracy.
   The best one is used to recommend products.

Run `python recommender.py` to see the model comparison in the terminal.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

related = {
    "Cricket": "Fitness", "Football": "Running", "Badminton": "Fitness", "Fitness": "Running",
    "Running": "Fitness", "Mobile": "Laptop", "Laptop": "Mobile", "Kitchen": "Stationery",
    "Stationery": "Laptop", "Camping": "Running",
}


def create_data(n_customers=1000, seed=42):
    """Return the product table and the purchase history."""
    rng = np.random.default_rng(seed)
    products = pd.DataFrame(
        [(name, cat) for cat, names in catalog.items() for name in names],
        columns=["product", "category"],
    )
    products["price"] = rng.integers(2, 60, len(products)) * 100

    rows = []
    for customer in range(1, n_customers + 1):
        category = rng.choice(list(catalog))
        items = list(rng.choice(catalog[category], rng.integers(2, 6), replace=False))
        if rng.random() < 0.5:  # some customers also shop in a related category
            other = related[category]
            items += list(rng.choice(catalog[other], rng.integers(1, 3), replace=False))
        for item in items:
            rows.append((customer, item))

    purchases = pd.DataFrame(rows, columns=["customer_id", "product"])
    return products, purchases


def build_dataset(products, purchases):
    """One row per (customer, hidden product): X = other products bought, y = hidden product."""
    all_products = list(products["product"])
    X, y = [], []
    for _, basket in purchases.groupby("customer_id")["product"]:
        basket = list(basket)
        for hidden in basket:
            X.append([1 if (p in basket and p != hidden) else 0 for p in all_products])
            y.append(hidden)
    return pd.DataFrame(X, columns=all_products), pd.Series(y)


# model -> hyperparameters to try
MODELS = {
    "K-Nearest Neighbors": (KNeighborsClassifier(),
                            {"n_neighbors": [5, 15, 30], "metric": ["cosine", "euclidean"]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {"max_depth": [10, 20, None], "min_samples_leaf": [1, 5]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"n_estimators": [50, 100], "max_depth": [10, None]}),
    "Logistic Regression": (LogisticRegression(max_iter=1000),
                            {"C": [0.1, 1, 10]}),
    "Naive Bayes": (BernoulliNB(),
                    {"alpha": [0.1, 0.5, 1.0]}),
}


def top5_score(model, X, y):
    """Top-5 accuracy: is the hidden product among the model's 5 best guesses?"""
    return top_k_accuracy_score(y, model.predict_proba(X), k=5, labels=model.classes_)


def compare_models(X, y):
    """Tune every model, test it, and return a results table and the best model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results, best_model, best_score = [], None, -1
    for name, (model, params) in MODELS.items():
        search = GridSearchCV(model, params, cv=3, scoring=top5_score)
        search.fit(X_train, y_train)

        model = search.best_estimator_
        probs = model.predict_proba(X_test)
        top5 = top_k_accuracy_score(y_test, probs, k=5, labels=model.classes_)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append({"Model": name, "Best parameters": str(search.best_params_),
                        "Top-5 accuracy": round(top5, 3), "Accuracy": round(acc, 3)})

        if top5 > best_score:
            best_score, best_model = top5, model

    results = pd.DataFrame(results).sort_values("Top-5 accuracy", ascending=False)
    return results.reset_index(drop=True), best_model


def recommend(model, all_products, bought, k=5):
    row = pd.DataFrame([[1 if p in bought else 0 for p in all_products]], columns=all_products)
    probs = model.predict_proba(row)[0]
    ranking = pd.DataFrame({"product": model.classes_, "score": probs})
    ranking = ranking[~ranking["product"].isin(bought)]
    return ranking.sort_values("score", ascending=False).head(k)


if __name__ == "__main__":
    products, purchases = create_data()
    X, y = build_dataset(products, purchases)
    print(f"Products: {len(products)}, purchases: {len(purchases)}, training rows: {len(X)}\n")
    results, best_model = compare_models(X, y)
    print(results.to_string(index=False))
    print(f"\nBest model: {results.loc[0, 'Model']}")