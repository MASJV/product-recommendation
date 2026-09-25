"""Streamlit app. Run: streamlit run app.py"""

import streamlit as st

from recommender import build_dataset, compare_models, create_data, recommend

st.set_page_config(page_title="Product Recommendation", page_icon="🛒")


@st.cache_resource(show_spinner="Training models, please wait...")
def get_model():
    products, purchases = create_data()
    X, y = build_dataset(products, purchases)
    results, best_model = compare_models(X, y)
    return products, results, best_model


products, results, model = get_model()
all_products = list(products["product"])

st.title("🛒 Product Recommendation System")
st.write("Select the products you have bought and get suggestions for what to buy next.")

bought = st.multiselect("Products you have purchased", sorted(all_products))

if st.button("Suggest Recommendations", type="primary"):
    if not bought:
        st.warning("Please select at least one product.")
    else:
        recs = recommend(model, all_products, bought, k=5)
        recs = recs.merge(products, on="product")
        st.subheader("Recommended for you")
        for i, row in enumerate(recs.itertuples(), start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {row.product}**  \n{row.category} · ₹{row.price}")

with st.expander("Model comparison"):
    st.write("Each model was tuned with GridSearchCV. The best one is used for recommendations.")
    st.dataframe(results, hide_index=True)
    st.write(f"**Selected model:** {results.loc[0, 'Model']}")