# QuickCart India — Quick-Commerce Customer Behaviour Analysis

**A complete student machine-learning project exploring quick-commerce survey responses and testing Linear Regression and Polynomial Regression.**

🌐 **Streamlit application:** https://quickcart-india-polynomial-regression-vishnu.streamlit.app/

## About the project

Quick-commerce apps have changed the way some people buy groceries. Instead of relying only on planned monthly shopping, customers can order a few items whenever they need them. This project studies survey responses about that behaviour and checks whether selected answers can estimate a person's reported impulse-buying frequency score.

I built the workflow from the beginning: collecting the data, checking it, cleaning it, exploring it through graphs, preparing features, training different regression approaches, evaluating their results and connecting a saved model to a Streamlit application.

The main goal is not to show a high score at any cost. I want to understand what the dataset actually supports and whether Polynomial Regression improves on simpler approaches when tested on responses it has not seen during training.

## Dataset

The project uses the published Mendeley Data survey **“The Shift from Planned Monthly Groceries to Instant Impulse Purchasing Due to Quick Commerce Platforms.”**

- Source: [Mendeley Data](https://data.mendeley.com/datasets/wc2yz9485m/1)
- DOI: `10.17632/wc2yz9485m.1`
- Dataset size in the source snapshot: 800 survey responses
- Original fields: 10
- Each row represents one survey response, not one order or invoice.

The repository includes the raw source snapshot and a cleaned dataset copy under `data/`.

## What is being estimated?

The target is `Impulse_Buying_Frequency_Score`, created from the leading number in the `Impulse_Buying_Frequency` survey answer.

| Score | Survey answer |
|---:|---|
| 1 | Never |
| 2 | Rarely |
| 3 | Sometimes |
| 4 | Frequently |
| 5 | Almost Always |

This is an ordered survey response. Regression treats the differences between these scores as approximately equal, which is an assumption made for this exercise. The prediction is an estimate of the reported survey score; it is not an exact count of future impulse purchases, an order amount, or a rupee-valued forecast.

`Average_Order_Value` is recorded as spending bands, not exact amounts. It is therefore treated as a category and is not used as the numerical target.

## Project workflow

The complete analysis is available in one notebook: `notebooks/quickcart_complete_analysis.ipynb`.

### 1. Collect and inspect the data

The notebook downloads the public source file using Python's `urllib` module and reads it from memory using `io.BytesIO`. This avoids manually uploading a raw CSV into Google Colab. It checks the expected columns, number of rows, data types, missing values, repeated rows and unique answers.

### 2. Clean the responses

- Missing app responses are retained as `Not specified` rather than guessed or deleted.
- Unnecessary spaces around text answers are removed.
- The two ordered 1–5 survey responses are converted into numeric score columns while the original text fields are preserved.
- The raw data is not overwritten.

### 3. Exploratory data analysis

The notebook includes graphs for missing values, target-score distribution, most-used app, ordering-frequency change, average order-value bands, monthly budget ranges, home-inventory agreement, multi-select grocery categories and impulse triggers. It also compares score distributions across selected groups and checks the relationship between the ordered survey scores.

Every graph is followed by an explanation of why it is used, what its axes represent and what can reasonably be concluded from it. These are descriptive findings from the survey sample and do not establish cause and effect.

### 4. Feature preparation

The model inputs include app choice, ordering-frequency change, AOV band, traditional bulk-shopping reliance, app-budget range, reported expenditure impact, home-inventory agreement score and binary indicators for grocery categories.

The `Impulse_Triggers` question is excluded from the main model because it is closely related to the target and could make the evaluation less meaningful.

Text categories are converted using one-hot encoding. The numerical agreement score is standardised. Binary category indicators remain as 0/1 values. The preprocessing is kept inside Scikit-learn pipelines so it is fitted on training data only.

### 5. Train and compare models

The notebook uses a stratified 80/20 split with `random_state=42`. It compares:

1. Mean baseline
2. Linear Regression
3. Polynomial Regression (degree 2)
4. Ridge Polynomial Regression (degree 2)

Polynomial Regression adds squared and interaction terms. That can represent more complicated patterns, but can also create a large number of features and overfit. Ridge adds a penalty to large model weights and is included as a regularised comparison.

### 6. Evaluation and error analysis

The models are compared using MAE, MSE, RMSE, R², median absolute error and explained variance. The notebook includes test-error comparisons, actual-versus-predicted plots, residual plots, residual distribution, error by target score, rounded-score comparison and five-fold cross-validation on the training data.

The final model choice follows held-out test results, with the mean baseline used as the reference. A more complex model is not labelled better unless the results support that conclusion.

## Repository structure

```text
quickcart-india-polynomial-regression/
├── data/
│   ├── quick_commerce_survey_responses_raw.csv
│   └── quick_commerce_survey_responses_cleaned.csv
├── model/
│   └── quickcart_regression_model.joblib
├── notebooks/
│   └── quickcart_complete_analysis.ipynb
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── test_predictions.csv
├── app.py
├── requirements.txt
└── README.md
```

The model and reports are generated when the notebook is executed. The notebook downloads the cleaned CSV, Joblib model, metrics JSON and supporting CSV reports to the local Downloads area in Google Colab. These files are then placed into the folders above.

## Run the project locally

### 1. Clone the repository

```bash
git clone https://github.com/mvishnuvardhan2229-dotcom/quickcart-india-polynomial-regression.git
cd quickcart-india-polynomial-regression
```

### 2. Create an environment and install packages

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

Open the local address printed in the terminal, normally `http://localhost:8501`.

The app expects the trained artifact at `model/quickcart_regression_model.joblib`. It reads `reports/metrics.json` to display the experiment results. Run the notebook first if these files are not available.

## Run the notebook in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/quickcart_complete_analysis.ipynb` using **File → Upload notebook**.
3. Run the notebook from top to bottom.
4. The raw dataset is downloaded by the notebook using `urllib` and read in memory using `io.BytesIO`.
5. Download the output files and place them in the folders shown in the notebook.

## Streamlit application

The app provides:

- A form for entering survey-style responses.
- An estimate on the 1–5 reported frequency scale.
- Selection between the saved model pipelines.
- A view of the recorded evaluation metrics.
- Explanations of the target, workflow and model limitations.

The application uses the saved model for inference. It does not retrain the model every time a visitor submits the form. User-entered values are not stored in a project database.

## What I learned

Through this project I practised:

- Collecting a public dataset with `urllib` and reading it with `io.BytesIO`.
- Inspecting and cleaning survey data using Pandas.
- Understanding missing values, categorical fields and multi-select answers.
- Creating numeric scores and binary features.
- Exploratory data analysis and choosing suitable graphs.
- Separating target values from input features.
- Train/test splitting and stratification.
- One-hot encoding, scaling, pipelines and avoiding preprocessing leakage.
- Comparing a baseline with Linear Regression and Polynomial Regression.
- Understanding overfitting, regularisation, residuals and regression metrics.
- Saving model pipelines with Joblib.
- Building and deploying an interactive Streamlit application.

## Limitations

- The responses are self-reported and cross-sectional.
- The target is an ordinal answer from 1 to 5, not a continuous measurement.
- The dataset is not transaction-level data and does not contain exact orders or rupee amounts.
- It has no customer ID, purchase date, location, age or income fields.
- Findings describe this survey sample and should not be generalised to all Indian customers without representative external data.
- Correlation and group differences do not prove causation.
- Test performance on this dataset does not guarantee real-world predictive performance.

This is an educational machine-learning project and should not be used for financial, commercial or personal decisions.

## License and dataset acknowledgement

Dataset: [Mendeley Data, DOI 10.17632/wc2yz9485m.1](https://data.mendeley.com/datasets/wc2yz9485m/1). Please check the source record for the current licence and attribution requirements before redistributing the dataset independently.
