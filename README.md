# Product Recommendation System

Select the products a customer has bought, and the app suggests 5 other products to buy (e.g. Cricket Bat + Cricket Ball → Stumps, Cricket Helmet, Cricket Kit Bag ...)

Built with scikit-learn and Streamlit.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

`python recommender.py` prints the model comparison in the terminal.

## How it works

1. **Dataset:** 59 products in 10 categories and the purchases of 1,000 customers, generated in `recommender.py` with a fixed random seed.
2. **Recommendation as classification:** for each customer, one purchased product is hidden. The model gets the other products they bought (as 0/1 columns) and has to predict the hidden product.
3. **Models compared:** K-Nearest Neighbors, Decision Tree, Random Forest, Logistic Regression, Naive Bayes. Each one is tuned with `GridSearchCV`.
4. **Metric:** top-5 accuracy, i.e. how often the hidden product is in the model's top 5 guesses. The model with the highest score is used in the app.
5. **Recommending:** the best model gives a probability for every product; products the user already has are removed and the top 5 are shown.

## Files

- `recommender.py`: dataset, model comparison, recommendations
- `app.py`: Streamlit app