import json, pickle, dill
from pathlib import Path

ART_DIR = Path("artifacts")
PRE_PATH = ART_DIR / "preprocessor.pkl"
MODEL_PATH = ART_DIR / "model.pkl"

def main():
    with open(PRE_PATH, "rb") as f:
        pre = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = dill.load(f)

    # Extract column groups from ColumnTransformer
    num_cols, cat_cols = [], []
    for name, trans, cols in pre.transformers:
        if name == "num_pipeline":
            num_cols = list(cols)
        elif name == "cat_pipeline":
            cat_cols = list(cols)

    # Get inner transformers
    num_pipe = pre.named_transformers_["num_pipeline"]
    cat_pipe = pre.named_transformers_["cat_pipeline"]
    scaler_num = num_pipe.named_steps.get("scaler")
    ohe = cat_pipe.named_steps.get("one_hot_encoder")
    scaler_cat = cat_pipe.named_steps.get("scaler")

    # Categories from OneHotEncoder
    cat_categories = None
    if ohe is not None and hasattr(ohe, "categories_"):
        cat_categories = [list(c) for c in ohe.categories_]

    # Feature names after transform
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        # Fallback: numeric names + OHE names
        ohe_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else []
        feat_names = list(num_cols) + ohe_names

    # Numeric scaler stats (if fitted)
    num_means = getattr(scaler_num, "mean_", None)
    num_scales = getattr(scaler_num, "scale_", None)

    # Categorical scaler note (with_mean=False → no means)
    cat_with_mean = getattr(scaler_cat, "with_mean", None)
    cat_with_std = getattr(scaler_cat, "with_std", None)

    # Model details (supports LinearRegression; prints generic for others)
    model_info = {
        "class": type(model).__name__,
    }
    if hasattr(model, "coef_"):
        coefs = list(map(float, model.coef_.ravel()))
        pairs = list(zip(feat_names, coefs))
        model_info.update({
            "intercept": float(getattr(model, "intercept_", 0.0)),
            "n_features": len(feat_names),
            "coefficients": pairs,  # [(feature, weight), ...]
        })

    report = {
        "preprocessor": {
            "type": type(pre).__name__,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "ohe_categories_per_column": dict(zip(cat_cols, cat_categories or [[]]*len(cat_cols))),
            "feature_names_after_transform": feat_names,
            "numeric_scaler": {
                "mean_": list(map(float, num_means)) if num_means is not None else None,
                "scale_": list(map(float, num_scales)) if num_scales is not None else None,
            },
            "categorical_scaler": {
                "with_mean": cat_with_mean,
                "with_std": cat_with_std,
            },
        },
        "model": model_info,
    }

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

"""Here’s what you’re looking at and how it all ties together:

### Where the “transformer” comes from
- It’s from scikit-learn. Specifically:
  - `Pipeline` from `sklearn.pipeline`
  - `ColumnTransformer` from `sklearn.compose`
  - `SimpleImputer`, `StandardScaler`, `OneHotEncoder` from `sklearn.impute` and `sklearn.preprocessing`
- You installed these when you installed scikit-learn. Your preprocessor was built in `src/components/data_transformation.py` and saved to `artifacts/preprocessor.pkl`.

### What `pre.transformers` is
- When you load the preprocessor (`pre`), it’s a fitted `ColumnTransformer`.
- `pre.transformers` is a list of its configured sub-transformers before fitting, shaped like:
  - `[("num_pipeline", <Pipeline object>, numerical_columns), ("cat_pipeline", <Pipeline object>, categorical_columns)]`
- After fitting, you typically use `pre.named_transformers_["num_pipeline"]` or `pre.named_transformers_["cat_pipeline"]` to access the fitted pipelines and their inner steps (like imputer, encoder, scaler).

### Line-by-line walkthrough of your describe script
Explained by logical blocks (each line does exactly this):

- Imports
  - `import json, pickle, dill`: load and serialize objects; `pickle` for the preprocessor, `dill` for the model.
  - `from pathlib import Path`: convenient path handling.

- Paths
  - Define `ART_DIR = Path("artifacts")`.
  - `PRE_PATH = ART_DIR / "preprocessor.pkl"`, `MODEL_PATH = ART_DIR / "model.pkl"`.

- Load artifacts
  - Open and `pickle.load(PRE_PATH)` → gives you the fitted `ColumnTransformer` (your “preprocessor”).
  - Open and `dill.load(MODEL_PATH)` → gives you the trained model (you saved via `dill.dump`).

- Extract column groups
  - Iterate `for name, trans, cols in pre.transformers:`
    - If `name == "num_pipeline"`, set `num_cols = list(cols)`.
    - If `name == "cat_pipeline"`, set `cat_cols = list(cols)`.
  - This reads the original columns assigned to each sub-pipeline.

- Access inner steps of each pipeline
  - `num_pipe = pre.named_transformers_["num_pipeline"]`
  - `cat_pipe = pre.named_transformers_["cat_pipeline"]`
  - Grab steps:
    - `scaler_num = num_pipe.named_steps.get("scaler")`
    - `ohe = cat_pipe.named_steps.get("one_hot_encoder")`
    - `scaler_cat = cat_pipe.named_steps.get("scaler")`

- One-hot categories
  - If `ohe` exists and has `categories_`, collect lists of categories per categorical column.

- Feature names after full transform
  - Try `pre.get_feature_names_out()` (newer sklearn).
  - If that fails, fall back to `ohe.get_feature_names_out(cat_cols)` and prepend `num_cols`.

- Numeric scaler stats
  - If `scaler_num` is a `StandardScaler`, read `mean_` and `scale_` (vectors for each numeric column).

- Categorical scaler flags
  - `with_mean` and `with_std` from the categorical scaler (you set `with_mean=False` to keep sparse matrices valid).

- Model details
  - `type(model).__name__` for the class name.
  - If it’s linear-like (has `coef_`), extract `intercept_` and pair each coefficient with its feature name.

- Assemble report and print
  - Build a dict with:
    - numeric/categorical columns
    - OHE categories per column
    - final feature names
    - numeric scaler stats
    - model summary (and coefficients if available)
  - `print(json.dumps(report, indent=2))` to make it human-readable.

### Quick mental model
- ColumnTransformer is your “split → transform → recombine” engine:
  - Numeric columns: impute median → scale.
  - Categorical columns: impute mode → one-hot encode → scale without centering.
- `pre.transformers` tells you “what was configured.”
- `pre.named_transformers_` gives you the fitted inner pieces and what they learned (like category vocabularies).
- The script just loads these objects and pretty-prints a summary so you can “read” them."""