# Imports 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import io

###############################################################################

# Load the dataset
file_path_1 = r'project_dataset/extract - fees - data analyst - .csv'
file_path_2 = r'project_dataset/extract - cash request - data analyst.csv'
data1 = pd.read_csv(file_path_1)
data2 = pd.read_csv(file_path_2)
# making local copies of original dataset 
fees = data1.copy() 
cash = data2.copy()

###############################################################################

# EDA
st.title("EDA on Datasets")

# .head() method
st.markdown("<h3 style='font-size:24px;'>Display the first few rows:</h3>", unsafe_allow_html=True)
st.markdown("""```python\nfees.head()\n```""")
st.dataframe(fees.head())  # or st.table()
st.markdown("""```python\ncash.head()\n```""")
st.dataframe(cash.head())  # or st.table()

st.write("")

# .describe() method
st.markdown("<h3 style='font-size:24px;'>Summary statistics of numerical values:</h3>", unsafe_allow_html=True)
st.markdown("""```python\nfees.describe().T\n```""")
st.dataframe(fees.describe().T)
st.markdown("""```python\ncash.describe().T\n```""")
st.dataframe(cash.describe().T)

st.write("")

# .info() method (column counts, column labels, column data types, the number of non-null values in each column, ...)
st.markdown("<h3 style='font-size:24px;'>Information about the dataset:</h3>", unsafe_allow_html=True)

# Capture and display fees.info()
buffer_fees = io.StringIO()
fees.info(buf=buffer_fees)
st.markdown("""```python\nfees.info()\n```""")
st.code(buffer_fees.getvalue(), language="text")

st.write("")

# Capture and display cash.info()
buffer_cash = io.StringIO()
cash.info(buf=buffer_cash)
st.markdown("""```python\ncash.info()\n```""")
st.code(buffer_cash.getvalue(), language="text")

st.write("")

###############################################################################

# Data quality analysis
# Checking for null values in the dataset
st.markdown("<h3 style='font-size:24px;'>Data quality analysis (finding null values):</h3>", unsafe_allow_html=True)

st.write("Checking for null values in fees:")
st.markdown("""```python\nfees.isnull().sum()\n```""")
st.code(fees.isnull().sum(), language="text")
st.write("There are only 4 null values in `cash_request_id`, which can be removed without affecting the dataset. The cleaned data will be ready for joining.")

st.write("")

st.write("Checking for null values in cash requests:")
st.markdown("""```python\ncash.isnull().sum()\n```""")
st.code(cash.isnull().sum(), language="text")
st.write("There are 2103 null values for `user_id`. They need to be discarded in frequency analysis.")

cash.dropna(subset=["user_id"], inplace=True)
st.markdown("```python\ncash.dropna(subset=['user_id'], inplace=True)\n```")

st.write("")

###############################################################################

# Cohort Analysis

st.title("Cohort Analysis")
st.markdown("<h3 style='font-size:24px;'>Create a column with YYYY-MM-DD format:</h3>", unsafe_allow_html=True)
st.write("Create a new column with YYYY-MM-DD format for `activity_month`. The activity month is the month in which the cash request was made:")
cash["activity_month"] = cash["created_at"].str[:7] + "-01"
st.markdown("""```python\ncash["activity_month"] = cash["created_at"].str[:7] + "-01"\n```""")

st.write("")

# Selecting 3 columns from cash
st.write("""Select 3 significant columns `["id", "created_at", "activity_month"]` from `cash`:""")
st.markdown("""```python\ncash[["id", "created_at", "activity_month"]])\n```""")
st.dataframe(cash[["id", "created_at", "activity_month"]])

st.write("")

# Apply .dtypes attribute
st.write("Check the data types:")
st.markdown("""```python\ncash[["id", "created_at", "activity_month"]].dtypes\n```""")
st.code(cash[["id", "created_at", "activity_month"]].dtypes)

st.write("")

# Making sure that activity month is a datetime object
st.write("Make sure that `activity_month` is a datetime object:")
st.markdown("""```python\ncash["activity_month"] = cash["activity_month"].astype("datetime64[s]")\n```""")
cash["activity_month"] = cash["activity_month"].astype("datetime64[s]")

st.write("")

# Double check
st.write("Observe the change:")
st.markdown("""```python\ncash[["id", "created_at", "activity_month"]].dtypes\n```""")
st.code(cash[["id", "created_at", "activity_month"]].dtypes)

st.write("")

# Create cohorts
st.markdown("<h3 style='font-size:24px;'>Create Cohorts:</h3>", unsafe_allow_html=True)
st.write("Users are assigned to the cohort of their first activity:")
st.markdown("""```python\ncash["cohort_month"] = cash["user_id"].map(cash.groupby("user_id")["activity_month"].min())\n```""")
cash["cohort_month"] = cash["user_id"].map(cash.groupby("user_id")["activity_month"].min())

st.write("")

# Add a new column cohort_index
st.write("Add a new column `cohort_index` with cohort as an index. Then, group the activities based on the month following the start of the cohort:")
st.markdown("""```python\ncash["cohort_index"] = (cash["activity_month"].dt.to_period("M").astype(int) - cash["cohort_month"].dt.to_period("M").astype(int))\n```""")
cash["cohort_index"] = (cash["activity_month"].dt.to_period("M").astype(int) - cash["cohort_month"].dt.to_period("M").astype(int))
st.code(cash["cohort_index"])

st.write("")

# Select and display the significant columns
st.write("Select and display the significant columns:")
st.markdown("""```python\ncash[["id", "created_at", "user_id", "activity_month", "cohort_month", "cohort_index"]]\n```""")
st.dataframe(cash[["id", "created_at", "user_id", "activity_month", "cohort_month", "cohort_index"]])

st.write("")

# Find the user with the maximum frequency of cash requests
st.write("Find the user with the maximum frequency of cash requests:")
st.markdown("""```python\ncash["user_id"].value_counts()\n```""")
st.code(cash["user_id"].value_counts())

st.write("")

# Create a dataframe for the user with the maximum cash requests
st.write("Create a dataframe for the user with the maximum cash requests:")
st.markdown("""```python\nuser_max = cash[cash["user_id"] == 3377][["id", "created_at", "user_id", "activity_month", "cohort_month", "cohort_index"]]\n```""")
user_max = cash[cash["user_id"] == 3377.][["id", "created_at", "user_id", "activity_month", "cohort_month", "cohort_index"]]
st.dataframe(user_max)

st.write("")

# Observe user_max sorted by activity_month in descending order
st.write("Observe user_max sorted by `activity_month` in descending order:")
st.markdown("""```python\nuser_max.sort_values(by='activity_month', ascending=False)\n```""")
st.dataframe(user_max.sort_values(by='activity_month', ascending=False))

st.write("")

st.write("As we can see, the longer the activity period is, the greater the cohort index will be.")

st.write("")

# Observe that index also works with more than year-long window
st.write("Observe that index also works with more than year-long window:")
st.markdown("""```python\nuser_max2 = cash[cash["user_id"] == 526][["id", "created_at", "user_id", "activity_month", "cohort_month", "cohort_index"]]\n```""")
user_max2 = cash[cash["user_id"] == 526][["id", "created_at", "user_id", "activity_month", "cohort_month", "cohort_index"]]
st.markdown("""```python\nuser_max2 = user_max2.sort_values(by='activity_month', ascending=False)\n```""")
user_max2 = user_max2.sort_values(by='activity_month', ascending=False)
st.markdown("""```python\nuser_max2\n```""")
st.dataframe(user_max2)

st.write("")

# Merging fees and cash
st.markdown("<h3 style='font-size:24px;'>Merging fees and cash:</h3>", unsafe_allow_html=True)
st.write("Counting the ids from cash request that match with cash_request_ids from fees:")
st.markdown("""```python\nMatching IDs count:\n```""")
st.code(cash[cash['id'].isin(fees['cash_request_id'])].shape[0])
st.write("")
st.markdown("""```python\nNon_matching IDs count:\n```""")
st.code(cash[~cash['id'].isin(fees['cash_request_id'])].shape[0])

st.write("")

st.write("The reason for more `id` in cash compared to `cash_request_id` in fees might lie in the fact that a large number of cash requests were rejected or have not been paid back yet (no fees).")

st.markdown("""```python\ncash["status"].value_counts()\n```""")
st.code(cash["status"].value_counts())

st.write("")

# Revenue generated by the Cohort:
st.markdown("<h3 style='font-size:24px;'>Revenue generated by each cohort over months:</h3>", unsafe_allow_html=True)
st.write("Calculate the total fees (revenue) for all accepted payments, grouped by `cash_request_id`:")
st.markdown("""```python\nfees_agg_revenue = fees[fees["status"] == 'accepted'].groupby("cash_request_id")[["total_amount"]].sum()\n```""")
fees_agg_revenue = fees[fees["status"] == 'accepted'].groupby("cash_request_id")[["total_amount"]].sum()
st.markdown("""```python\nfees_agg_revenue.reset_index()\n```""")
fees_agg_revenue.reset_index()
st.dataframe(fees_agg_revenue)

st.write("")

# Merging fees_agg_revenue and cash
st.markdown("<h3 style='font-size:24px;'>Merging fees_agg_revenue and cash:</h3>", unsafe_allow_html=True)
st.write("Processed data sets cash and fees_agg_revenue are ready to be merged:")
st.markdown("""```python\ndf = cash.merge(fees_agg_revenue, how ='left', left_on='id', right_on='cash_request_id')\n```""")
df = cash.merge(fees_agg_revenue, how ='left', left_on='id', right_on='cash_request_id')
st.markdown("""```python\ndf\n```""")
st.dataframe(df)

st.write("")

###############################################################################

# Metrics

st.title("Metrics to Analyze")

# Frequency of Service Usage
st.markdown("<h3 style='font-size:24px;'>Frequency of Service Usage:</h3>", unsafe_allow_html=True)
st.write("To understand how often users from each cohort utilize IronHack Payments' cash advance services over time.")
st.markdown("""```python\nfreq_usage = df.groupby(["cohort_month", "activity_month"])["user_id"].count().reset_index()\n```""")
freq_usage = df.groupby(["cohort_month", "activity_month"])["user_id"].count().reset_index()
freq_usage.rename(columns={"user_id": "cash_request_attempt"}, inplace=True)
st.markdown("""```python\nfreq_usage\n```""")
st.dataframe(freq_usage)

st.write("")

# Incident Rate
st.markdown("<h3 style='font-size:24px;'>Incident Rate:</h3>", unsafe_allow_html=True)
st.write("""Calculate Incident Rate (assuming "rejected" marks payment issues):""")
st.markdown("""```python\nincident_df = df[df["status"] == "rejected"]\n```""")
incident_df = df[df["status"] == "rejected"]  # Assuming "rejected" marks payment issues
st.markdown("""```python\nincident_rate = (
    incident_df.groupby(["cohort_month", "activity_month"])["user_id"]
    .count()
    .reset_index()
    .rename(columns={"user_id": "incident_count"})
)\n```""")
incident_rate = incident_df.groupby(["cohort_month", "activity_month"])["user_id"].count().reset_index().rename(columns={"user_id": "incident_count"})

st.markdown("""```python\nincident_rate\n```""")
st.dataframe(incident_rate)

st.write("")

# Merge incident rate with frequency data
st.markdown("<h3 style='font-size:24px;'>Merge incident rate with frequency data:</h3>", unsafe_allow_html=True)

st.markdown("""
```python\ncohort_analysis = pd.merge(freq_usage, incident_rate, on=["cohort_month", "activity_month"], how="left")\n 
cohort_analysis["incident_count"] = cohort_analysis["incident_count"].fillna(0)\n
cohort_analysis["incident_rate"] = cohort_analysis["incident_count"] / cohort_analysis["cash_request_attempt"]""")

cohort_analysis = pd.merge(freq_usage, incident_rate, on=["cohort_month", "activity_month"], how="left")
cohort_analysis["incident_count"] = cohort_analysis["incident_count"].fillna(0)
cohort_analysis["incident_rate"] = cohort_analysis["incident_count"] / cohort_analysis["cash_request_attempt"]

st.write("")

# New Metric: User Retention Rate
st.markdown("<h3 style='font-size:24px;'>New Metric: User Retention Rate:</h3>", unsafe_allow_html=True)

st.markdown("""
```python\ncohort_analysis = df.groupby(["cohort_month", "activity_month"])["user_id"].nunique().reset_index()\n 
unique_users_per_month.rename(columns={"user_id": "unique_users"}, inplace=True)\n""")

unique_users_per_month = df.groupby(["cohort_month", "activity_month"])["user_id"].nunique().reset_index()
unique_users_per_month.rename(columns={"user_id": "unique_users"}, inplace=True)

st.write("")

# Calculate User Retention Rate
st.markdown("<h3 style='font-size:24px;'>Calculate User Retention Rate:</h3>", unsafe_allow_html=True)

st.markdown("""
```python\nfirst_month_users = df.groupby("cohort_month")["user_id"].nunique().reset_index()\n
first_month_users.rename(columns={"user_id": "initial_users"}, inplace=True)\n
cohort_analysis = cohort_analysis.merge(first_month_users, on="cohort_month", how="left")\n
cohort_analysis["initial_users"] = cohort_analysis["initial_users"].fillna(1)\n""")

first_month_users = df.groupby("cohort_month")["user_id"].nunique().reset_index()
first_month_users.rename(columns={"user_id": "initial_users"}, inplace=True)
cohort_analysis = cohort_analysis.merge(first_month_users, on="cohort_month", how="left")
cohort_analysis["initial_users"] = cohort_analysis["initial_users"].fillna(1)  # Avoid division errors

st.write("")

# Merge unique_users_per_month to ensure it's present
st.markdown("<h3 style='font-size:24px;'>Merge unique_users_per_month to ensure it's present:</h3>", unsafe_allow_html=True)

st.markdown("""
```python\ncohort_analysis = cohort_analysis.merge(unique_users_per_month, on=["cohort_month", "activity_month"], how="left", validate="one_to_one")\n 
cohort_analysis["unique_users"] = cohort_analysis["unique_users"].fillna(0)\n""")

cohort_analysis = cohort_analysis.merge(unique_users_per_month, on=["cohort_month", "activity_month"], how="left", validate="one_to_one")
cohort_analysis["unique_users"] = cohort_analysis["unique_users"].fillna(0)

st.write("")

# Calculate retention rate
st.markdown("<h3 style='font-size:24px;'>Calculate Retention Rate:</h3>", unsafe_allow_html=True)

st.markdown("""
```python\ncohort_analysis["retention_rate"] = cohort_analysis["unique_users"] / cohort_analysis["initial_users"]\n 
print(cohort_analysis["retention_rate"])\n""")

cohort_analysis["retention_rate"] = cohort_analysis["unique_users"] / cohort_analysis["initial_users"]
st.dataframe(cohort_analysis["retention_rate"])

st.write("")

# Merge all metrics
st.markdown("<h3 style='font-size:24px;'>Merge All Metrics:</h3>", unsafe_allow_html=True)

st.markdown("""
```python\ncohort_analysis = cohort_analysis.merge(fees_agg_revenue, on=["cohort_month", "activity_month"], how="left")\n""")

cohort_analysis = cohort_analysis.merge(df, on=["cohort_month", "activity_month"], how="left")
# st.dataframe(cohort_analysis)
# st.dataframe(df)

st.write()

###############################################################################

# Visualization

st.title("Visualization")
# st.markdown("<h3 style='font-size:24px;'></h3>", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filters")
cohort_selected = st.sidebar.multiselect("Select Cohort Month", cohort_analysis["cohort_month"].unique())
activity_selected = st.sidebar.date_input("Select Date Range", [])

# Apply filters
if cohort_selected:
    cohort_analysis = cohort_analysis[cohort_analysis["cohort_month"].isin(cohort_selected)]

# Ensure 'activity_month' is in datetime format
cohort_analysis["activity_month"] = pd.to_datetime(cohort_analysis["activity_month"])

# Apply date filter if the user selects a range
if activity_selected and len(activity_selected) == 2:
    start_date = pd.to_datetime(activity_selected[0])
    end_date = pd.to_datetime(activity_selected[1])
    cohort_analysis = cohort_analysis[(cohort_analysis["activity_month"] >= start_date) & (cohort_analysis["activity_month"] <= end_date)]

# Display dataframe
st.write("### Cohort Analysis Data")
st.dataframe(cohort_analysis)

# Line chart for Service Usage
st.write("## Service Usage Over Time")
st.line_chart(cohort_analysis.set_index("activity_month")["cash_request_attempt"])

# Incident Rate Plot
st.write("## Incident Rate by Cohort")
fig, ax = plt.subplots()
sns.lineplot(data=cohort_analysis, x="activity_month", y="incident_rate", palette="tab10", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
# hue="cohort_month"

# Revenue Plot
st.write("## Revenue Generated")
fig, ax = plt.subplots()
sns.lineplot(data=cohort_analysis, x="activity_month", y="amount", palette="tab10", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
# hue="cohort_month"

# Retention Rate Plot
st.write("## User Retention Rate")
fig, ax = plt.subplots()
sns.lineplot(data=cohort_analysis, x="activity_month", y="retention_rate", palette="tab10", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
# hue="cohort_month"

# Bar Plot: Total Transactions per Cohort
st.subheader("Bar Plot: Total Transactions per Cohort")
plt.figure(figsize=(12, 6))
sns.barplot(data=cohort_analysis, x="cohort_month", y="cash_request_attempt", estimator=sum, palette="viridis")
plt.title("Total Transactions per Cohort")
plt.xlabel("Cohort Month")
plt.ylabel("Total Transactions")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Histogram: Distribution of Revenue
st.subheader("Histogram: Revenue Distribution")
plt.figure(figsize=(12, 6))
sns.histplot(cohort_analysis["amount"], bins=30, kde=True, color="blue")
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
st.pyplot(plt.gcf())