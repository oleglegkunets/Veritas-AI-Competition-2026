# Veritas-AI-Competition-2026
Project of Oleg Legkunets, Zhang DonChao and Jindi Li.

LoanWise

Disclaimer: This project is a prototype and is not financial advice.

Track: AI + Finance
Team: QuantGreats
The Problem

Taking out a loan can seem like a straightforward decision, especially when the monthly repayments look manageable. However, people don’t always consider how a new loan might affect their overall financial situation.
We wanted to create something that would help people make more informed borrowing decisions. Rather than simply telling users whether they might default on a loan, we wanted to help them understand why a loan could be risky and whether they could realistically afford it.
That’s how we came up with LoanWise.

What We Built

LoanWise is an interactive web application developed using Python and Streamlit. It helps users assess their loan default risk while also giving them a clearer picture of how borrowing could affect their monthly finances. Users enter information about their income, employment, existing debts, personal circumstances, and the loan they’re considering. LoanWise then uses this information to provide a risk prediction and a breakdown of the financial implications.
The application has three main features:
1. Loan Default Prediction: Uses machine learning to estimate the likelihood of a borrower defaulting on a loan.
2. Loan Repayment Calculator: Calculates monthly repayments, total interest, and the overall cost of the loan.
3. Monthly Cash Flow Analysis: Shows how much money a user would have left each month after paying their living expenses, existing debts, and the new loan repayment.

What Makes LoanWise Different

When we started developing LoanWise, we realised that predicting loan defaults was only one part of the problem.
A machine learning model might tell someone that they have a relatively low risk of defaulting, but that doesn’t necessarily mean taking out the loan is a good financial decision.
Instead of relying entirely on a prediction, we combined machine learning with practical financial calculations. This allows users to consider both their estimated default risk and the potential impact of the loan on their everyday budget.
How It Works
1. Data Collection and Preparation
We used the Credit Risk Dataset from Kaggle, published by Lao Tse.
Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
The dataset contains 32,581 records and 12 columns, including information about borrowers’ income, age, employment history, home ownership, loan amounts, interest rates, loan purposes, and credit histories.
Our target variable was loan_status, which indicates whether a borrower defaulted on their loan:
* 0 = Non-default
* 1 = Default
Before training our models, we needed to prepare the dataset.
Some values were missing, so we replaced missing numerical values with the median and categorical values with the most common entry. We also applied one-hot encoding to categorical variables so that our machine learning models could process them. We also did some basic data cleaning: removed duplicate records and impossible data. Finally, excluded loan_grade from the training data as it is not something users would normally know
2. Training the Models
Rather than choosing one machine learning algorithm from the beginning, we decided to compare three: Logistic Regression, Decision Tree, and Random Forest.

We divided the dataset into an 80% training set and a 20% validation set.
To evaluate the models, we focused on the F1-score, which takes both precision and recall into account. We chose this metric because the dataset contains more non-default cases than default cases, meaning accuracy alone might give a misleading impression of how well a model performs.
After evaluating the three models, LoanWise selects the one with the highest validation F1-score. We then retrained the selected model on the complete dataset before integrating it into the application.
3. Financial Calculations
We also wanted LoanWise to provide information beyond the machine learning prediction.
The repayment calculator uses a standard loan repayment formula to calculate how much a user would need to pay each month, how much interest they would pay over the loan period, and the total repayment amount.
Our Results
After testing all three models, we obtained the following validation F1-scores:
Model:				F1-score:
Logistic Regression		0.570
Decision Tree			0.736
Random Forest		0.798

Random Forest performed the best, achieving an F1-score of 0.798. Decision Tree followed with 0.736, while Logistic Regression scored 0.570.
Based on these results, we chose Random Forest as the final prediction model for LoanWise.
Although we were pleased with its performance, we also recognised that the model has limitations.
An F1-score of 0.798 doesn’t mean that every prediction will be correct. The model learns patterns from historical data, and those patterns may not always reflect an individual user’s circumstances.
There is also a difference between estimating someone’s default risk and knowing whether they will actually repay a loan. Financial situations can change unexpectedly, and no prediction model can account for every possible factor.

Challenges and What We Learned
Training a model to predict loan defaults was a useful starting point, but we felt that a prediction alone wasn’t enough to help someone understand the risks of borrowing.
This led us to develop the repayment calculator and monthly cash flow analysis alongside the machine learning model.
By comparing F1-scores, we developed a better understanding of why precision, recall, and class imbalance matter, particularly in financial prediction tasks.
Finally, this project helped us see that building a useful AI application isn’t just about developing an accurate model. It’s also about understanding the problem, recognising the model’s limitations, and presenting the results in a way that people can use responsibly.
What We’d Do Next

If we had more time to develop LoanWise, there are several areas we would like to explore.
1. Improving the prediction model
We would experiment with hyperparameter tuning, cross-validation, and additional datasets to see whether we could improve the reliability of our predictions. We would also like to evaluate how well the model performs on different groups of borrowers.
2. Making the financial analysis more personalised
Our current application focuses on income, expenses, existing debts, and loan repayments. In the future, we would like to include factors such as savings, emergency funds, and unexpected expenses.
We would also like to introduce different financial scenarios, allowing users to explore what might happen if their income decreased or their expenses increased.
3. Improving the user experience
We would like to make the interface more intuitive, especially for users with limited financial knowledge. One possibility would be to provide clearer explanations of the factors influencing a risk prediction, rather than presenting only the final result.
4. Making LoanWise more reliable and responsible
We would also like to test how well the model’s predicted probabilities reflect actual default rates and investigate potential biases in the dataset.
Because borrowing decisions can have serious consequences, we believe that transparency and reliability should be just as important as predictive performance.

Sources and Tools
* Dataset: Kaggle – Credit Risk Dataset by Lao Tse
    https://www.kaggle.com/datasets/laotse/credit-risk-dataset
* Programming language: Python
    https://www.python.org/
* Web application framework: Streamlit
    https://streamlit.io/
* Machine learning library: Scikit-learn
    https://scikit-learn.org/
* Data processing library: Pandas
    https://pandas.pydata.org/
* AI assistance: ChatGPT, used for guidance, explanations, debugging, and development support and Claude, mainly helped us clean the dataset from impossible values and repeated loans.

