import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, accuracy_score, precision_score, recall_score, f1_score)
import joblib


def load_data(path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at {path}.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df

def summarize_dataframe(df):
    print('\n=== Dataframe Head ===')
    print(df.head())
    print('\n=== Missing values per column ===')
    print(df.isna().sum())

def compute_kpis(df):
    kpis = {}
    total = len(df)
    kpis['total_employees'] = total
    if 'Attrition' in df.columns:
        kpis['attrition_count'] = int((df['Attrition'] == 'Yes').sum())
        kpis['attrition_rate'] = kpis['attrition_count'] / total
    if 'Age' in df.columns:
        kpis['avg_age'] = df['Age'].mean()
    if 'MonthlyIncome' in df.columns:
        kpis['avg_monthly_income'] = df['MonthlyIncome'].mean()
    return kpis

def preprocess_for_model(df, drop_columns=None, target='Attrition'):
    df_proc = df.copy()
    default_drops = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    if drop_columns is None:
        drop_columns = default_drops
    else:
        drop_columns = list(set(default_drops + drop_columns))
    for c in drop_columns:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])
    if target not in df_proc.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    y = (df_proc[target] == 'Yes').astype(int)
    X = df_proc.drop(columns=[target])
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return X, y, numerical_cols, categorical_cols

def build_preprocessor(numerical_cols, categorical_cols):
    num_transformer = StandardScaler()
    skl_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
    if skl_ver >= (1, 2):
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ])
    return preprocessor

def train_and_evaluate(X, y, preprocessor, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    pipe_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state))
    ])
    pipe_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=random_state))
    ])
    print('\nTraining Logistic Regression...')
    pipe_lr.fit(X_train, y_train)
    print('Training Random Forest...')
    pipe_rf.fit(X_train, y_train)
    results = {}
    for name, model in [('LogisticRegression', pipe_lr), ('RandomForest', pipe_rf)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            'model': model,
            'roc_auc': auc,
        }
        print(f"\n=== {name} === ROC AUC: {auc:.4f}")
    best_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    print(f"\nBest model: {best_name}")
    return results, best_name

def save_model(model, filename='best_attrition_model.pkl'):
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def load_model(filename='best_attrition_model.pkl'):
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except Exception:
            return None
    return None

def run_analysis(csv_path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    df = load_data(csv_path)
    summarize_dataframe(df)
    kpis = compute_kpis(df)
    print('\nKPIs:', kpis)
    X, y, numerical_cols, categorical_cols = preprocess_for_model(df)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    results, best_name = train_and_evaluate(X, y, preprocessor)
    best_model = results[best_name]['model']
    save_model(best_model)
    print('\nAnalysis complete.')

def run_dashboard(csv_path='WA_Fn-UseC_-HR-Employee-Attrition.csv', model_path='best_attrition_model.pkl'):
    try:
        import streamlit as st
    except Exception:
        print('Streamlit is required.')
        return

    st.set_page_config(page_title='HR Attrition Dashboard', layout='wide')
    st.title('Employee Attrition — HR Dashboard')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error(f'CSV not found at {csv_path}.')
        return

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    attrition_count = int((df['Attrition'] == 'Yes').sum()) if 'Attrition' in df.columns else None
    attrition_rate = (attrition_count / total) if attrition_count is not None else None

    with col1:
        st.metric('Total employees', total)
    with col2:
        st.metric('Attritions (count)', attrition_count)
    with col3:
        st.metric('Attrition rate', f"{attrition_rate:.2%}" if attrition_rate is not None else 'N/A')
    with col4:
        if 'MonthlyIncome' in df.columns:
            st.metric('Avg Monthly Income', f"{df['MonthlyIncome'].mean():.0f}")

    st.markdown('---')

    if 'Department' in df.columns and 'Attrition' in df.columns:
        dept = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        dept_prop = dept.div(dept.sum(axis=1), axis=0)
        st.subheader('Attrition proportion by Department')
        st.bar_chart(dept_prop)

    model = load_model(model_path)

    if model is None:
        st.info('Training model on first run — please wait...')
        X, y, numerical_cols, categorical_cols = preprocess_for_model(df)
        preprocessor = build_preprocessor(numerical_cols, categorical_cols)
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42))
        ])
        pipe.fit(X, y)
        save_model(pipe, model_path)
        model = pipe
        st.success('Model trained and ready!')
    else:
        st.success('Loaded trained model successfully!')

    features_df = df.copy()
    if 'Attrition' in features_df.columns:
        features_df = features_df.drop(columns=['Attrition'])

    try:
        probs = model.predict_proba(features_df)[:, 1]
        df['Attrition_Prob'] = probs

        st.subheader('Top 10 employees by predicted attrition risk')
        if 'EmployeeNumber' in df.columns:
            top10 = df.sort_values('Attrition_Prob', ascending=False).head(10)[
                ['EmployeeNumber', 'Attrition_Prob', 'Department', 'JobRole', 'Age', 'MonthlyIncome']
            ].fillna('')
        else:
            top10 = df.sort_values('Attrition_Prob', ascending=False).head(10)
        st.dataframe(top10)

        st.subheader('Attrition probability distribution')
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(probs, bins=30)
        ax.set_xlabel('Predicted attrition probability')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    except Exception as e:
        st.error(f'Could not run predictions: {e}')

    st.markdown('---')
    st.subheader('Predict for one employee')

    with st.form('employee_form'):
        age = st.number_input('Age', min_value=18, max_value=70, value=30)
        monthly_income = st.number_input('MonthlyIncome', min_value=0, value=5000)
        if 'JobRole' in df.columns:
            job_role = st.selectbox('JobRole', options=sorted(df['JobRole'].unique()))
        else:
            job_role = st.text_input('JobRole')
        if 'Department' in df.columns:
            department = st.selectbox('Department', options=sorted(df['Department'].unique()))
        else:
            department = st.text_input('Department')
        submit = st.form_submit_button('Predict')

        if submit:
            sample = df.drop(columns=['Attrition']) if 'Attrition' in df.columns else df.copy()
            sample = sample.head(1).copy()
            if 'Age' in sample.columns:
                sample.loc[:, 'Age'] = age
            if 'MonthlyIncome' in sample.columns:
                sample.loc[:, 'MonthlyIncome'] = monthly_income
            if 'JobRole' in sample.columns:
                sample.loc[:, 'JobRole'] = job_role
            if 'Department' in sample.columns:
                sample.loc[:, 'Department'] = department
            try:
                prob = model.predict_proba(sample)[:, 1][0]
                st.write(f'Predicted probability of attrition: {prob:.2%}')
                st.write('Predicted class:', 'Yes' if prob > 0.5 else 'No')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    st.markdown('---')
    st.caption('Dashboard generated from employee_attrition_analysis_and_dashboard.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-analysis', action='store_true')
    parser.add_argument('--csv', type=str, default='WA_Fn-UseC_-HR-Employee-Attrition.csv')
    parser.add_argument('--model', type=str, default='best_attrition_model.pkl')
    args = parser.parse_args()

    if args.run_analysis:
        run_analysis(csv_path=args.csv)
    else:
        run_dashboard(csv_path=args.csv, model_path=args.model)
