import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

st.set_page_config(
    page_title="LoanWise",
    page_icon="",
    layout="centered"
)


def calculate_loan_metrics(principal, annual_interest_rate, months, additional_fees):
    r = annual_interest_rate / 12.0

    if r == 0:
        monthly_payment = principal / months
    else:
        monthly_payment = principal * (
            r * (1 + r) ** months
        ) / ((1 + r) ** months - 1)

    total_repayment = monthly_payment * months + additional_fees
    total_interest = total_repayment - principal - additional_fees

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_repayment": round(total_repayment, 2),
        "total_interest": round(total_interest, 2)
    }

def cash_flow_warnings(monthly_income, disposable_income, monthly_payment, existing_debt):
    leftover = disposable_income - monthly_payment
    dti = (existing_debt + monthly_payment) / monthly_income

    warnings = []

    if leftover < 0:
        warnings.append(
            "Critical Warning: Your monthly cash flow is negative after loan repayment."
        )
    elif leftover < monthly_income * 0.15:
        warnings.append(
            "Caution: Your remaining monthly cash is below 15% of income."
        )

    if dti > 0.5:
        warnings.append(
            f"Heavy Debt Load: Your debt-to-income ratio is {dti * 100:.1f}%."
        )

    return warnings, leftover, dti


@st.cache_data
def load_data():
    data = pd.read_csv("Data/credit_risk_dataset.csv")

    data = data.drop_duplicates()

    data = data[data["person_age"] <= 100]
    data = data[data["person_emp_length"].isna() | (data["person_emp_length"] <= 60)]

    return data

@st.cache_resource
def train_models(data):
    X = data.drop(columns=["loan_status", "loan_grade"], errors="ignore")
    y = data["loan_status"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_columns = X.select_dtypes(include="number").columns
    categorical_columns = X.select_dtypes(exclude="number").columns

    fill_values = {}

    for column in numeric_columns:
        fill_values[column] = X_train[column].median()

    for column in categorical_columns:
        mode_values = X_train[column].mode()
        fill_values[column] = mode_values.iloc[0] if not mode_values.empty else "Unknown"

    X_train = pd.get_dummies(X_train.fillna(fill_values))
    columns = X_train.columns.tolist()

    X_valid = pd.get_dummies(X_valid.fillna(fill_values))
    X_valid = X_valid.reindex(columns=columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    scores = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_valid_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)

        scores[name] = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

    best_model_name = max(scores, key=scores.get)

    X = pd.get_dummies(X.fillna(fill_values))
    X = X.reindex(columns=columns, fill_value=0)

    if best_model_name == "Logistic Regression":
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)

        final_model = LogisticRegression(max_iter=1000)
        final_model.fit(X_scaled, y)
    elif best_model_name == "Decision Tree":
        final_scaler = None
        final_model = DecisionTreeClassifier(random_state=42)
        final_model.fit(X, y)
    else:
        final_scaler = None
        final_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        final_model.fit(X, y)

    return final_model, best_model_name, final_scaler, columns, fill_values, scores

data = load_data()

required_columns = [
    "loan_status",
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

missing_columns = [column for column in required_columns if column not in data.columns]

if missing_columns:
    st.error(f"Your CSV file is missing these columns: {', '.join(missing_columns)}")
    st.stop()

model, model_name, scaler, columns, fill_values, scores = train_models(data)

st.title("LoanWise")
st.write("A simple loan risk prediction and financial planning application.")
st.info("This project is a prototype and is not financial advice.")
st.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Loan Prediction", "About"]
)

if page == "Home":
    st.header("Welcome to LoanWise")

    st.write(
        "LoanWise uses machine learning to predict whether a loan applicant "
        "may be at risk of default."
    )

    st.write(
        "It also calculates loan repayments, total interest, disposable income, "
        "and remaining monthly cash flow."
    )

    st.write("Go to the Loan Prediction page to try it.")

elif page == "Loan Prediction":
    st.header("Loan Risk Prediction")
    st.write("Enter the applicant's information below.")

    with st.form("loan_form"):
        st.subheader("Applicant")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25
        )

        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=50000
        )

        home = st.selectbox(
            "Home Ownership",
            sorted(data["person_home_ownership"].dropna().unique())
        )

        employment = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=60.0,
            value=2.0
        )

        credit_history = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            max_value=100,
            value=5
        )

        previous_default = st.selectbox(
            "Previous Credit Default",
            sorted(data["cb_person_default_on_file"].dropna().unique())
        )

        st.subheader("Monthly Budget")

        expenses = st.number_input(
            "Monthly Living Expenses",
            min_value=0,
            value=1500,
            step=100
        )

        existing_debt = st.number_input(
            "Existing Monthly Debt/Mortgage Payments",
            min_value=0,
            value=300,
            step=50
        )

        st.subheader("Proposed Loan")

        intent = st.selectbox(
            "Loan Intent",
            sorted(data["loan_intent"].dropna().unique())
        )

        amount = st.number_input(
            "Loan Amount",
            min_value=1,
            value=10000
        )

        interest = st.number_input(
            "Loan Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

        months = st.selectbox(
            "Repayment Period (Months)",
            options=[12, 24, 36, 48, 60, 120],
            index=2
        )

        fees = st.number_input(
            "Additional Fees/Insurance",
            min_value=0,
            value=0,
            step=50
        )

        submitted = st.form_submit_button("Predict Loan Risk")

    if submitted:
        if income <= 0:
            st.error("Annual income must be greater than zero.")
        else:
            applicant = pd.DataFrame([{
                "person_age": age,
                "person_income": income,
                "person_home_ownership": home,
                "person_emp_length": employment,
                "loan_intent": intent,
                "loan_amnt": amount,
                "loan_int_rate": interest,
                "loan_percent_income": amount / income,
                "cb_person_default_on_file": previous_default,
                "cb_person_cred_hist_length": credit_history
            }])

            applicant = applicant.fillna(fill_values)
            applicant = pd.get_dummies(applicant)
            applicant = applicant.reindex(
                columns=columns,
                fill_value=0
            )

            if scaler is not None:
                applicant_input = scaler.transform(applicant)
            else:
                applicant_input = applicant

            prediction = model.predict(applicant_input)[0]
            probability = model.predict_proba(applicant_input)[0][1]

            monthly_income = income / 12.0

            metrics = calculate_loan_metrics(
                amount,
                interest / 100.0,
                months,
                fees
            )

            disposable_income = monthly_income - expenses - existing_debt

            warnings, leftover, dti = cash_flow_warnings(
                monthly_income,
                disposable_income,
                metrics["monthly_payment"],
                existing_debt
            )

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Higher Risk of Loan Default")
            else:
                st.success("Lower Risk of Loan Default")

            # No model is ever completely certain, so very low and very high
            # values are shown as "less than 1%" and "more than 99%"
            if probability < 0.01:
                probability_text = "less than 1%"
            elif probability > 0.99:
                probability_text = "more than 99%"
            else:
                probability_text = f"{probability * 100:.2f}%"

            st.write(
                f"Estimated default probability: {probability_text}"
            )

            st.divider()
            st.subheader("Loan Cost Breakdown")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Monthly Repayment",
                f"${metrics['monthly_payment']:,.2f}"
            )

            c2.metric(
                "Total Interest",
                f"${metrics['total_interest']:,.2f}"
            )

            c3.metric(
                "Total Repayment",
                f"${metrics['total_repayment']:,.2f}"
            )

            st.subheader("Monthly Cash Flow")

            d1, d2, d3 = st.columns(3)

            d1.metric(
                "Monthly Income",
                f"${monthly_income:,.2f}"
            )

            d2.metric(
                "Disposable Income",
                f"${disposable_income:,.2f}"
            )

            d3.metric(
                "Left After Repayment",
                f"${leftover:,.2f}"
            )

            st.caption(
                "Disposable income = monthly income − living expenses − existing debt payments. "
                f"Debt-to-income including this loan: {dti * 100:.1f}%."
            )

            for warning in warnings:
                st.warning(warning)

elif page == "About":
    st.header("About LoanWise")

    st.write(
        "LoanWise is a student machine learning project that combines "
        "credit risk prediction with loan cost calculations and cash flow analysis."
    )

    st.subheader("Machine Learning Models")

    st.write("The application compares three classification algorithms:")

    for name in [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]:
        st.write(f"- {name}")

    st.subheader("Model Evaluation")

    st.write(
        "The models are compared using the F1-score on a validation dataset. "
        "The application automatically selects the model with the highest score."
    )

    for name, score in scores.items():
        st.write(f"**{name}:** {score:.3f}")

    st.success(f"Best Model: {model_name}")

    st.caption(
        "The F1-scores shown above are validation results, not independent test results."
    )

    st.subheader("Other Features")

    st.write(
        "LoanWise also calculates monthly loan repayments, total interest, "
        "total repayment, disposable income, and remaining cash after repayment."
    )

    st.write(
        "It uses simple financial rules to warn users about negative cash flow, "
        "low remaining income, and high debt-to-income ratios."
    )