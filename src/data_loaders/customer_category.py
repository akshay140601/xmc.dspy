import os
from collections import Counter, defaultdict
import datasets
import pandas as pd
import json

def _load_customer_category():
    base_dir = "./data"
    customer_dir = os.path.join(base_dir, "customer_category")

    # get ontology
    customer_terms = [
        term.strip("\n")
        for term in open(os.path.join(customer_dir, "customer_category.txt")).readlines()
    ]

    # load the dataset
    df = pd.read_excel(os.path.join(customer_dir, "data_train_test.xlsx"))
    validation_df = df.iloc[:80]
    test_df = df.iloc[80:]

    # get prior counts, normalized, from the train set
    all_train_cats = validation_df["label"]
    all_train_cats = [ls.split(", ") for ls in all_train_cats]
    all_train_cats = [x for ls in all_train_cats for x in ls]

    customer_category_priors = Counter(all_train_cats)
    customer_category_priors = defaultdict(
        lambda: 0.0,
        {k: v / len(all_train_cats) for k, v in customer_category_priors.items()},
    )
    # save prior
    with open(os.path.join(customer_dir, "customer_category_priors.json"), "w") as fp:
        json.dump(customer_category_priors, fp)
        
    return validation_df, test_df, customer_terms, None, customer_category_priors      
    