
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "model" / "quickcart_regression_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
DATA_PATH = ROOT / "data" / "quick_commerce_survey_responses_cleaned.csv"

st.set_page_config(
    page_title="QuickCart India | ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL UI THEME
# ============================================================

st.html("""
<style>

/* Main application background */
.stApp {
    background: #F4F7FC;
    color: #172033;
}

[data-testid="stAppViewContainer"] {
    background: #F4F7FC;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Force readable text throughout main application */
.stApp p,
.stApp li,
.stApp label,
.stApp span,
.stApp .stMarkdown,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"] {
    color: #243247;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #172554 !important;
    font-weight: 750;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] > div {
    background: #111827;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #E5E7EB !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}

/* Sidebar dataset metric cards: white background with dark, readable text */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE5F2 !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] * {
    color: #172554 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] label,
section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #475569 !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #172554 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #374151;
}

/* Sidebar expanders */
section[data-testid="stSidebar"] details {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 10px;
    margin-bottom: 10px;
}

section[data-testid="stSidebar"] details summary {
    color: #FFFFFF !important;
}

/* Hero section */
.hero {
    background: linear-gradient(115deg, #172554, #5B21B6, #087E8B);
    padding: 35px;
    border-radius: 20px;
    color: white;
    box-shadow: 0 12px 35px rgba(30, 41, 90, 0.15);
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 2.7rem;
    margin: 0;
}

.hero p {
    color: #F0F9FF !important;
    font-size: 1.08rem;
    margin-top: 12px;
    margin-bottom: 0;
}

/* Description cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    padding: 22px;
    border-radius: 15px;
    min-height: 150px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.info-card h4 {
    color: #4338CA !important;
    font-size: 0.95rem;
    margin: 0 0 12px 0;
}

.info-card h2 {
    color: #172554 !important;
    font-size: 2rem;
    margin: 0 0 8px 0;
}

.info-card p {
    color: #475569 !important;
    font-size: 0.9rem;
    margin: 0;
}

/* General cards */
.panel {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    padding: 22px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.panel h3 {
    color: #312E81 !important;
}

.panel p,
.panel li {
    color: #334155 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE5F2;
    border-radius: 13px;
    padding: 15px;
}

div[data-testid="stMetric"] label {
    color: #475569 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #172554 !important;
}

/* Input fields */
/* Input fields - high contrast */

div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stMultiSelect"] [data-baseweb="select"] * {
    color: #172554 !important;
}

div[data-testid="stSelectbox"] input,
div[data-testid="stMultiSelect"] input {
    color: #172554 !important;
}

div[data-testid="stSlider"] {
    color: #312E81 !important;
}

/* Dropdown options */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #FFFFFF !important;
}

/* Force white controls and dark text in every selectbox, including
   Shopping preferences and the Regression model selector. */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [role="combobox"],
div[data-testid="stMultiSelect"] [role="combobox"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border-color: #CBD5E1 !important;
}

/* Selected value, placeholder, and multi-select text */
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stMultiSelect"] [data-baseweb="select"] *,
div[data-testid="stSelectbox"] input,
div[data-testid="stMultiSelect"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Open dropdown menu and each option */
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
[role="listbox"] *,
[role="option"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Keep hover and keyboard-selected options clearly visible */
[role="option"]:hover,
[aria-selected="true"][role="option"] {
    background-color: #E8EEF9 !important;
    color: #111827 !important;
}

/* Preserve visible dropdown arrows */
div[data-testid="stSelectbox"] [data-baseweb="select"] svg,
div[data-testid="stMultiSelect"] [data-baseweb="select"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}
/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background: #4338CA;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 650;
    min-height: 45px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #312E81;
    color: white !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE5F2;
    border-radius: 12px;
}

/* Alerts */
[data-testid="stAlert"] p {
    color: inherit !important;
}

/* High-contrast explanatory and result notices */
.notice-box {
    color: #FFFFFF !important;
    font-size: 1.08rem !important;
    font-weight: 600;
    line-height: 1.75;
    padding: 18px 22px;
    border-radius: 12px;
    margin: 12px 0 18px 0;
}

.notice-box,
.notice-box p,
.notice-box strong,
.notice-box span {
    color: #FFFFFF !important;
    font-size: 1.08rem !important;
    opacity: 1 !important;
}

.notice-info {
    background: #1E293B;
    border-left: 5px solid #60A5FA;
}

.notice-success {
    background: #14532D;
    border-left: 5px solid #4ADE80;
}

.notice-warning {
    background: #78350F;
    border-left: 5px solid #FBBF24;
}

.prediction-result {
    background: #172554;
    border: 1px solid #334A83;
    padding: 24px;
    border-radius: 15px;
    margin: 18px 0;
    color: #FFFFFF !important;
}

.prediction-result h3,
.prediction-result h1,
.prediction-result p {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.prediction-result h3 {
    font-size: 1.35rem !important;
    margin: 0 0 12px 0;
}

.prediction-result h1 {
    font-size: 2.15rem !important;
    margin: 0 0 14px 0;
}

.prediction-result p {
    font-size: 1.08rem !important;
    line-height: 1.7;
    margin: 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #334155 !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    line-height: 1.9;
    padding: 26px 20px;
    border-top: 1px solid #CBD5E1;
    margin-top: 30px;
}

.footer,
.footer * {
    color: #334155 !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
}

.footer strong {
    display: inline-block;
    color: #172554 !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    margin-bottom: 4px;
}

.small-muted {
    color: #64748B !important;
    font-size: 0.87rem;
}

</style>
""")


# ============================================================
# 3. LOAD PROJECT ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file is missing. Expected: "
            "model/quickcart_regression_model.joblib"
        )

    artifact = joblib.load(MODEL_PATH)

    metrics = None

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return artifact, metrics


@st.cache_data
def load_dataset():

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return pd.DataFrame()


try:
    artifact, metrics = load_artifacts()
    dataset = load_dataset()

except Exception as error:
    st.error(f"Unable to initialize application: {error}")
    st.stop()


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

models = artifact["all_models"]

selected_default = artifact.get(
    "selected_model_name",
    "Mean baseline"
)

feature_columns = artifact["feature_columns"]

category_options = artifact.get(
    "category_options",
    []
)

categorical_features = artifact.get(
    "categorical_features",
    []
)

numerical_features = artifact.get(
    "numerical_features",
    []
)

binary_features = artifact.get(
    "binary_features",
    []
)

labels = {
    1: "Never",
    2: "Rarely",
    3: "Sometimes",
    4: "Frequently",
    5: "Almost Always"
}


# ============================================================
# 5. DATASET COLUMN DESCRIPTIONS
# ============================================================

dataset_columns = [
    (
        "Most_Used_App",
        "Categorical",
        "The quick-commerce application most frequently used by the respondent."
    ),
    (
        "Ordering_Frequency_Change",
        "Categorical",
        "How the respondent's ordering frequency changed after using quick-commerce apps."
    ),
    (
        "Average_Order_Value",
        "Categorical",
        "The average spending band for an order, such as Under Rs.250 or Rs.500-1000."
    ),
    (
        "Reliance_On_Traditional_Bulk_Shopping",
        "Categorical",
        "Whether the respondent reduced, replaced or continued traditional monthly bulk grocery shopping."
    ),
    (
        "Shifted_Categories",
        "Multi-select",
        "Grocery categories that the respondent started purchasing through quick-commerce apps."
    ),
    (
        "Impulse_Buying_Frequency",
        "Ordinal",
        "The respondent's reported impulse-buying frequency: Never, Rarely, Sometimes, Frequently or Almost Always."
    ),
    (
        "Impulse_Triggers",
        "Multi-select",
        "Reported reasons behind impulse purchases, such as discounts, notifications and sudden cravings. Excluded from the predictive model."
    ),
    (
        "No_Home_Inventory_Agreement",
        "Ordinal",
        "Agreement with the survey statement about home inventory and grocery shopping, recorded on a 1–5 scale."
    ),
    (
        "Monthly_Budget_Percentage_On_Apps",
        "Categorical",
        "The percentage band of monthly grocery budget spent through quick-commerce apps."
    ),
    (
        "Overall_Grocery_Expenditure_Impact",
        "Categorical",
        "The respondent's reported change in overall grocery expenditure after adopting quick-commerce apps."
    )
]


# ============================================================
# 6. SIDEBAR - COMPLETE PROJECT DOCUMENTATION
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="text-align:center;padding:10px 0 20px 0;">
            <div style="font-size:2.7rem;">🛒</div>
            <h2 style="color:white!important;margin:0;">
                QuickCart India
            </h2>
            <p style="color:#A5B4FC!important;font-size:0.85rem;">
                Machine Learning Project
            </p>
        </div>
        """)

    st.divider()

    st.markdown("### 📌 Project overview")

    st.write(
        "An exploratory machine-learning project that studies "
        "customer-reported quick-commerce grocery shopping behaviour "
        "and evaluates regression models."
    )

    st.markdown("### 📊 Dataset information")

    if metrics:

        st.metric(
            "Total survey responses",
            metrics["dataset"]["rows"]
        )

        st.metric(
            "Original columns",
            metrics["dataset"]["original_columns"]
        )

        st.metric(
            "Cleaned columns",
            metrics["dataset"]["cleaned_columns"]
        )

    st.caption(
        "Source: Mendeley Data | DOI: 10.17632/wc2yz9485m.1"
    )

    with st.expander("📋 All 10 original dataset columns", expanded=False):

        for number, (column, dtype, description) in enumerate(
            dataset_columns, start=1
        ):

            st.html(f"""
                **{number}. {column}**

                <span style="color:#A5B4FC!important;font-size:0.78rem;">
                {dtype}
                </span>

                {description}
                """)

            st.divider()

    with st.expander("⚙️ Engineered columns and features"):

        st.markdown("""
        **Two additional columns are created during cleaning:**

        1. `Impulse_Buying_Frequency_Score`

           Converts the target answer into a numeric score from 1 to 5.

        2. `No_Home_Inventory_Agreement_Score`

           Converts the home-inventory agreement response into a numeric score from 1 to 5.

        **Model inputs:**

        - 6 categorical features
        - 1 numerical feature
        - 7 binary category indicators

        **Total: 14 input features before categorical encoding.**
        """)

    st.markdown("### 🎯 What are we predicting?")

    st.write(
        "The target is the reported impulse-buying frequency score."
    )

    for score, meaning in labels.items():

        st.markdown(
            f"**{score} — {meaning}**"
        )

    st.caption(
        "This is an ordered survey score, not an exact number "
        "of purchases or a future order prediction."
    )

    with st.expander("🧠 Algorithms used", expanded=False):

        st.markdown("""
        **1. Mean Baseline**

        Predicts the average target value from training data. It is
        the reference model used to check whether regression adds value.

        **2. Linear Regression**

        Models a linear relationship between prepared input features
        and the target score.

        **3. Polynomial Regression — Degree 2**

        Uses squared and interaction terms to model more complex
        relationships between features.

        **4. Ridge Polynomial Regression — Degree 2**

        Combines polynomial features with L2 regularization to reduce
        excessively large model coefficients and control overfitting.

        """)

        st.markdown("**Preprocessing techniques**")

        st.write("""
        - One-Hot Encoding
        - Standard Scaling
        - Polynomial Feature Expansion
        - Scikit-learn Pipelines
        """)

    with st.expander("🔬 How prediction works"):

        st.markdown("""
        **Step 1:** Collect survey-style answers.

        **Step 2:** Convert answers into the same feature structure
        used during model training.

        **Step 3:** Apply the saved preprocessing pipeline.

        **Step 4:** Pass the prepared input to the selected model.

        **Step 5:** Obtain a numerical prediction.

        **Step 6:** Round it to the nearest valid survey score
        between 1 and 5.

        The saved model is loaded from the Joblib artifact.
        The application does not retrain it for every visitor.
        """)

    with st.expander("🧪 Training and testing methodology"):

        st.markdown("""
        - Dataset: 800 responses
        - Training: 640 responses (80%)
        - Testing: 160 responses (20%)
        - Split: Stratified
        - Random state: 42
        - Cross-validation: 5-fold
        - Target: Impulse-buying frequency score
        """)

    with st.expander("⚠️ Dataset limitations"):

        st.markdown("""
        - Self-reported survey data.
        - Each row represents a respondent, not a purchase.
        - No exact transaction amounts.
        - No customer IDs or location details.
        - No age or income information.
        - Cannot establish cause and effect.
        - Results apply to this dataset and experiment only.
        """)

    st.divider()

    st.caption(
        "Developed using Python, Pandas, NumPy, "
        "Scikit-learn, Joblib and Streamlit."
    )


# ============================================================
# 7. MAIN HERO SECTION
# ============================================================

st.html(
    """
    <div class="hero">

        <h1>🛒 QuickCart India</h1>

        <p>
            Understanding quick-commerce customer behaviour
            through survey analysis and machine learning.
        </p>

        <p style="margin-top:15px;font-size:0.9rem;">
            DATA ANALYSIS &nbsp; | &nbsp;
            REGRESSION MODELLING &nbsp; | &nbsp;
            CUSTOMER BEHAVIOUR
        </p>

    </div>
    """)


st.html("""
This interactive dashboard explores how respondents report using
quick-commerce applications for grocery shopping. It demonstrates
data analysis, feature engineering, regression model comparison
and survey-score estimation.
""")


# ============================================================
# 8. DATASET SUMMARY CARDS
# ============================================================

rows = metrics["dataset"]["rows"] if metrics else "—"

train_rows = (
    metrics["split"]["training_rows"]
    if metrics else "—"
)

test_rows = (
    metrics["split"]["testing_rows"]
    if metrics else "—"
)

original_columns = (
    metrics["dataset"]["original_columns"]
    if metrics else "—"
)

c1, c2, c3, c4, c5 = st.columns(5)

cards = [
    (c1, "Survey responses", rows, "Published survey records"),
    (c2, "Original columns", original_columns, "Source dataset fields"),
    (c3, "Training records", train_rows, "80% training split"),
    (c4, "Testing records", test_rows, "20% testing split"),
    (c5, "Target scale", "1–5", "Ordered survey score")
]

for col, title, value, description in cards:

    with col:

        st.html(
            f"""
            <div class="info-card">

                <h4>{title}</h4>

                <h2>{value}</h2>

                <p>{description}</p>

            </div>
            """)


st.write("")


# ============================================================
# 9. MAIN APPLICATION TABS
# ============================================================

tab_predict, tab_results, tab_dataset, tab_about = st.tabs(
    [
        "🔎 Try an estimate",
        "📊 Model results",
        "📋 Dataset explorer",
        "📘 Project methodology"
    ]
)


# ============================================================
# 10. PREDICTION TAB
# ============================================================

with tab_predict:

    st.header("Customer behaviour estimation")

    st.write(
        "Enter survey-style answers to estimate the respondent's "
        "reported impulse-buying frequency score."
    )

    st.html("""
        <div class="notice-box notice-info">
            These are illustrative model inputs. The result is a
            survey-score estimate, not a guaranteed future purchase prediction.
        </div>
    """)

    # Dynamically get categorical options from the cleaned dataset.
    def get_options(column, fallback):

        if not dataset.empty and column in dataset.columns:

            values = (
                dataset[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if values:
                return sorted(values)

        return fallback


    app_options = get_options(
        "Most_Used_App",
        ["Blinkit", "Swiggy Instamart", "Zepto",
         "FlipKart Minutes", "Not specified"]
    )

    frequency_options = get_options(
        "Ordering_Frequency_Change",
        [
            "Increased moderately (1–3 orders per week)",
            "Increased drastically (4+ orders per week / almost daily)",
            "Remained the same",
            "Decreased significantly"
        ]
    )

    aov_options = get_options(
        "Average_Order_Value",
        [
            "Under Rs.250",
            "Rs.250 to 500",
            "Rs.500-1000",
            "Above Rs.1000"
        ]
    )

    bulk_options = get_options(
        "Reliance_On_Traditional_Bulk_Shopping",
        [
            "Significantly reduced (I do smaller bulk trips, relying more on apps)",
            "Slightly reduced (I still do bulk monthly shopping for main staples)",
            "Completely replaced (I no longer do monthly bulk trips)",
            "Unchanged (I only use quick apps for rare emergencies)"
        ]
    )

    budget_options = get_options(
        "Monthly_Budget_Percentage_On_Apps",
        [
            "Less than 10%",
            "10% to 30%",
            "30% to 60%",
            "More than 60%"
        ]
    )

    impact_options = get_options(
        "Overall_Grocery_Expenditure_Impact",
        [
            "Yes, slightly",
            "Yes, significantly (due to frequent small orders, delivery fees, and impulse buys)",
            "No, my spending has remained the same",
            "No, it actually decreased (less food wastage from over-buying in bulk)"
        ]
    )

    with st.form("quickcart_prediction_form"):

        st.subheader("1. Shopping preferences")

        left, right = st.columns(2)

        with left:

            app_value = st.selectbox(
                "Most-used quick-commerce app",
                app_options
            )

            freq_value = st.selectbox(
                "Change in ordering frequency",
                frequency_options
            )

            aov_value = st.selectbox(
                "Average order-value band",
                aov_options
            )

        with right:

            budget_value = st.selectbox(
                "Monthly grocery budget spent on apps",
                budget_options
            )

            bulk_value = st.selectbox(
                "Reliance on traditional bulk shopping",
                bulk_options
            )

            impact_value = st.selectbox(
                "Overall grocery expenditure impact",
                impact_options
            )

        st.divider()

        st.subheader("2. Home inventory behaviour")

        inventory_score = st.slider(
            "Agreement with the home-inventory statement",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Strongly Disagree; 5 = Strongly Agree"
        )

        st.caption(
            "This numeric value represents the agreement score "
            "used in the trained model."
        )

        st.divider()

        st.subheader("3. Grocery categories shifted to quick commerce")

        st.write(
            "Select all grocery categories that apply."
        )

        chosen_categories = st.multiselect(
            "Grocery categories",
            options=category_options
        )

        st.divider()

        st.subheader("4. Select the model")

        model_names = list(models.keys())

        default_index = (
            model_names.index(selected_default)
            if selected_default in model_names
            else 0
        )

        model_name = st.selectbox(
            "Regression model / baseline",
            model_names,
            index=default_index
        )

        submitted = st.form_submit_button(
            "🔎 Estimate reported frequency",
            use_container_width=True
        )


    # --------------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------------

    if submitted:

        row = {
            "Most_Used_App": app_value,
            "Ordering_Frequency_Change": freq_value,
            "Average_Order_Value": aov_value,
            "Reliance_On_Traditional_Bulk_Shopping": bulk_value,
            "Monthly_Budget_Percentage_On_Apps": budget_value,
            "Overall_Grocery_Expenditure_Impact": impact_value,
            "No_Home_Inventory_Agreement_Score": inventory_score
        }

        # Recreate the exact binary features used in training.
        for i, option in enumerate(category_options, start=1):

            row[f"Shifted_Category_{i:02d}"] = int(
                option in chosen_categories
            )

        input_frame = pd.DataFrame([row])

        # Ensure exact model feature order.
        input_frame = input_frame.reindex(
            columns=feature_columns,
            fill_value=0
        )

        try:

            raw_prediction = float(
                models[model_name].predict(input_frame)[0]
            )

            display_score = int(
                np.clip(
                    np.floor(raw_prediction + 0.5),
                    1,
                    5
                )
            )

            st.divider()

            st.html("""
                <div class="notice-box notice-success">
                    Prediction generated successfully.
                </div>
            """)

            st.subheader("Prediction result")

            a, b, c = st.columns(3)

            a.metric(
                "Raw model estimate",
                f"{raw_prediction:.3f}"
            )

            b.metric(
                "Nearest survey score",
                f"{display_score} / 5"
            )

            c.metric(
                "Selected model",
                model_name
            )

            st.html(f"""
                <div class="prediction-result">

                    <h3>Estimated reported frequency</h3>

                    <h1>
                        {display_score} — {labels[display_score]}
                    </h1>

                    <p>
                        The selected model estimates that the
                        survey-style response corresponds most closely
                        to the above frequency category.
                    </p>

                </div>
                """)

            st.html("""
                <div class="notice-box notice-warning">
                    This is an educational model estimate based on the
                    training survey. It is not a verified measurement,
                    a prediction of an exact number of future purchases,
                    or financial advice.
                </div>
            """)

        except Exception as error:

            st.error(
                f"Prediction could not be generated: {error}"
            )


# ============================================================
# 11. MODEL RESULTS TAB
# ============================================================

with tab_results:

    st.header("Model performance evaluation")

    st.write(
        "The following results compare the baseline and regression "
        "approaches using the held-out test dataset."
    )

    if metrics:

        selected = metrics.get("selected_model", "Not available")

        best_regression = metrics.get(
            "best_regression_by_test_mae",
            "Not available"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Selected model",
            selected
        )

        b.metric(
            "Best regression model",
            best_regression
        )

        c.metric(
            "Test records",
            metrics["split"]["testing_rows"]
        )

        st.divider()

        st.subheader("Comparison of all algorithms")

        results = pd.DataFrame(
            metrics["models"]
        ).T

        st.dataframe(
            results.round(4),
            use_container_width=True
        )

        st.caption(
            "MAE and RMSE: lower is better. "
            "R²: higher is generally better."
        )

        st.subheader("Test MAE comparison")

        mae_data = pd.DataFrame({
            "Model": list(metrics["models"].keys()),
            "Test MAE": [
                values["Test MAE"]
                for values in metrics["models"].values()
            ]
        }).sort_values("Test MAE")

        st.bar_chart(
            mae_data.set_index("Model"),
            horizontal=True
        )

        st.subheader("Important experimental finding")

        if metrics.get("regression_beats_baseline_by_test_mae"):

            st.success(
                "At least one regression model achieved a lower "
                "test MAE than the mean baseline."
            )

        else:

            st.html("""
                <div class="notice-box notice-warning">
                    None of the regression models achieved a lower
                    test MAE than the mean baseline. The simpler
                    baseline performed better on this test split.
                </div>
            """)

        st.html(f"""
            **Experiment interpretation:**

            {metrics.get("interpretation", "No interpretation available.")}
            """
        )

        st.divider()

        st.subheader("Cross-validation results")

        if metrics.get("cross_validation"):

            cv_df = pd.DataFrame(
                metrics["cross_validation"]
            )

            st.dataframe(
                cv_df.round(4),
                use_container_width=True
            )

        st.subheader("Understanding the evaluation measures")

        with st.expander("Mean Absolute Error (MAE)"):

            st.write(
                "The average absolute difference between actual "
                "and predicted scores. Lower values indicate "
                "smaller average errors."
            )

        with st.expander("Mean Squared Error (MSE)"):

            st.write(
                "The average squared prediction error. "
                "Larger errors receive greater penalties."
            )

        with st.expander("Root Mean Squared Error (RMSE)"):

            st.write(
                "The square root of MSE. It expresses the error "
                "on the same numerical scale as the target."
            )

        with st.expander("R-squared (R²)"):

            st.write(
                "Measures performance relative to a mean-based "
                "reference. A negative test R² indicates that "
                "the model performed worse than predicting "
                "the test-set mean."
            )

        st.download_button(
            "⬇️ Download model metrics",
            data=METRICS_PATH.read_bytes(),
            file_name="quickcart_metrics.json",
            mime="application/json"
        )

    else:

        st.info(
            "The reports/metrics.json file is not available."
        )


# ============================================================
# 12. DATASET EXPLORER TAB
# ============================================================

with tab_dataset:

    st.header("Explore the survey dataset")

    if not dataset.empty:

        st.write(
            "This is the cleaned dataset used in the project."
        )

        a, b, c = st.columns(3)

        a.metric("Rows", dataset.shape[0])
        b.metric("Columns", dataset.shape[1])
        c.metric(
            "Missing values",
            int(dataset.isna().sum().sum())
        )

        st.subheader("Column information")

        column_info = pd.DataFrame(
            dataset_columns,
            columns=[
                "Column name",
                "Data type",
                "Description"
            ]
        )

        column_info = pd.concat(
            [
                column_info,
                pd.DataFrame([
                    [
                        "Impulse_Buying_Frequency_Score",
                        "Engineered target",
                        "Numeric target created from the impulse-buying frequency answer."
                    ],
                    [
                        "No_Home_Inventory_Agreement_Score",
                        "Engineered feature",
                        "Numeric 1–5 agreement score extracted from the original response."
                    ]
                ], columns=column_info.columns)
            ],
            ignore_index=True
        )

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Cleaned dataset preview")

        st.dataframe(
            dataset.head(20),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download cleaned dataset",
            data=DATA_PATH.read_bytes(),
            file_name="quick_commerce_survey_responses_cleaned.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Cleaned dataset CSV was not found in the data folder."
        )


# ============================================================
# 13. PROJECT METHODOLOGY TAB
# ============================================================

with tab_about:

    st.header("Understanding the complete project")

    st.subheader("1. Problem statement")

    st.write("""
    Quick-commerce platforms have changed grocery shopping habits
    through fast delivery, instant ordering, discounts and convenient
    access to products.

    This project studies how respondents report these changes and
    examines whether their survey answers can estimate their
    self-reported impulse-buying frequency score.
    """)

    st.divider()

    st.subheader("2. Data collection and cleaning")

    st.markdown("""
    - Dataset collected from the published Mendeley Data source.
    - Original dataset contains 800 survey responses.
    - Original dataset contains 10 columns.
    - Cleaned dataset contains 12 columns after feature creation.
    - Missing application responses were handled as `Not specified`.
    - Multi-select grocery categories were separated into binary features.
    - Survey answers were converted into numerical scores where required.
    """)

    st.divider()

    st.subheader("3. Feature engineering")

    st.write("""
    The model uses 14 input features before encoding:

    - 6 categorical features.
    - 1 numerical agreement score.
    - 7 binary grocery-category indicators.

    Categorical variables are transformed using One-Hot Encoding.
    The numerical feature is standardized. The preprocessing steps
    are included in Scikit-learn pipelines to avoid fitting
    transformations on the test data.
    """)

    st.divider()

    st.subheader("4. Algorithms")

    st.markdown("""
    **Mean Baseline**

    Establishes a simple reference by predicting the training-set
    mean.

    **Linear Regression**

    Learns linear relationships between the prepared inputs
    and the target.

    **Polynomial Regression (Degree 2)**

    Adds squared and interaction features to model more complex
    relationships.

    **Ridge Polynomial Regression (Degree 2)**

    Applies L2 regularization to the polynomial model to control
    excessively large coefficients.
    """)

    st.divider()

    st.subheader("5. Model evaluation")

    st.write("""
    The dataset is split into 80% training and 20% testing records
    using stratification by target score and random state 42.

    Models are compared using MAE, MSE, RMSE, R², median absolute
    error and explained variance. Five-fold cross-validation
    is also performed on the training data.
    """)

    st.divider()

    st.subheader("6. Important limitation")

    st.html("""
        <div class="notice-box notice-warning">
            <p>
                The target is an ordinal survey response from 1 to 5.
                Regression treats the differences between the scores as
                approximately equal. This is an assumption, not a guarantee.
            </p>
            <p>
                The dataset does not contain actual order transactions,
                exact spending amounts or future purchase records.
            </p>
            <p>
                Therefore, this project estimates reported survey scores
                rather than actual future shopping activity.
            </p>
        </div>
    """)

    st.divider()

    st.subheader("7. Technology stack")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - Joblib
    - Streamlit
    - Google Colab
    - Git and GitHub
    """)


# ============================================================
# 14. FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <strong>🛒 QuickCart India</strong>

        <br><br>

        Customer Behaviour Analysis Using Regression

        <br>

        Dataset: Mendeley Data |
        DOI: 10.17632/wc2yz9485m.1

        <br><br>

        Built with Python, Pandas, Scikit-learn and Streamlit.

        <br>

        Educational Machine Learning Project

    </div>
    """)

