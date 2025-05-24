```python
# app.py - Streamlit application for Financial Product Recommender (simplified UI)
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import requests
from pyexcel_xls import get_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier, XGBRegressor

# Suppress warnings for a clean UI
warnings.filterwarnings('ignore')

# --- DATA LOADING (cached) ---
@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/aporrini/Financial-Product-Recommender/"
        "release-1.0/Dataset2_Needs.xls"
    )
    xls_bytes = requests.get(url).content
    workbook = get_data(io.BytesIO(xls_bytes))
    needs = pd.DataFrame(workbook['Needs'][1:], columns=workbook['Needs'][0])
    products = pd.DataFrame(workbook['Products'][1:], columns=workbook['Products'][0])
    # Map product names
    name_map = {1:"Balanced Mutual Fund",2:"Income Conservative UL",3:"Fixed Income MF",
                4:"High Dividend MF",5:"Balanced MF",6:"Defensive UL",7:"Aggressive UL",
                8:"Balanced UL",9:"Cautious Segregated",10:"Fixed Segregated",11:"Total Return Segregated"}
    products['ProductName'] = products['IDProduct'].map(name_map)
    return needs, products

needs_df, products_df = load_data()

# --- FEATURE ENGINEERING ---
def feature_engineering(df):
    X = df.copy()
    X['Wealth_log'] = np.log1p(X['Wealth'])
    X['Income_log'] = np.log1p(X['Income '])
    ratio = X['Income '] / X['Wealth'].replace(0, np.nan)
    X['Income_Wealth_Ratio_log'] = np.log1p(ratio.fillna(X['Income '].max()))
    X['Is_Single'] = (X['FamilyMembers']==1).astype(int)
    X['Is_Senior'] = (X['Age']>65).astype(int)
    X['Has_Education'] = (X['FinancialEducation']>0).astype(int)
    X['Risk_Age_Interaction'] = X['RiskPropensity'] * X['Age']
    base = ['Age','Gender','FamilyMembers','FinancialEducation']
    engineered = base + ['Wealth_log','Income_log','Income_Wealth_Ratio_log','Is_Single','Is_Senior','Has_Education','Risk_Age_Interaction']
    scaler = MinMaxScaler()
    X_base = pd.DataFrame(scaler.fit_transform(X[base]), columns=base)
    X_eng = pd.DataFrame(scaler.fit_transform(X[engineered]), columns=engineered)
    return X_base, X_eng

# --- MODEL TRAINING (cached) ---
@st.cache_resource
def train_models():
    # Prepare labels
    X_base, X_eng = feature_engineering(needs_df)
    y_i = needs_df['IncomeInvestment']
    y_a = needs_df['AccumulationInvestment']
    # Split defaults: use all for training (models cached)
    # Risk regressor
    risk_model = XGBRegressor(use_label_encoder=False, eval_metric='logloss', random_state=42)
    risk_model.fit(X_base, needs_df['RiskPropensity'])
    # Stacked classifiers
    def build_stack(y):
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        xgb_c = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        stack = StackingClassifier(
            estimators=[('rf',rf),('xgb',xgb_c)],
            final_estimator=RandomForestClassifier(), cv=5
        )
        stack.fit(X_eng, y)
        return stack
    stack_i = build_stack(y_i)
    stack_a = build_stack(y_a)
    # Thresholds at 0.5
    t_i, t_a = 0.5, 0.5
    return risk_model, stack_i, stack_a, t_i, t_a

risk_model, stack_inc, stack_acc, thr_i, thr_a = train_models()

# --- RECOMMENDATION FUNCTION ---
def recommend(user_input, epsilon=0.05):
    df = pd.DataFrame([user_input])
    Xb, Xe = feature_engineering(df)
    p_i = stack_inc.predict_proba(Xe)[:,1][0]
    p_a = stack_acc.predict_proba(Xe)[:,1][0]
    r_i = int(p_i>=thr_i)
    r_a = int(p_a>=thr_a)
    recs = []
    for typ,pred,prob in [('Income',r_i,p_i),('Accumulation',r_a,p_a)]:
        if pred:
            pool = products_df[products_df['Type']==(0 if typ=='Income' else 1)]
            sr = user_input['RiskPropensity']
            cand = pool[pool['Risk']<=sr+epsilon]
            if not cand.empty:
                best = cand.loc[cand['Risk'].idxmax()]
                recs.append({'Type':typ,'ID':int(best['IDProduct']),'Name':best['ProductName'],'Risk':best['Risk'],'Prob':round(prob,3)})
    if not recs:
        recs.append({'Type':'None','ID':0,'Name':'No product','Risk':np.nan,'Prob':np.nan})
    return pd.DataFrame(recs)

# --- QUESTIONNAIRES ---
financial_qs = [
    {'q':'What is your highest finance education level?',
     'opts':{'None':0,'High School':0.02,'Bachelor':0.05,'Master':0.1,'PhD':0.15}},
    {'q':'Have you worked in finance?',
     'opts':{'No':0,'Yes':0.1}}
]
risk_qs = [
    {'q':'If you lose 10%, what do you do?',
     'opts':{'Panic-sell':0,'Hold':0.1,'Buy more':0.2}},
    {'q':'Your portfolio drops 50%. Action?',
     'opts':{'Sell all':0,'Keep':0.05,'Average down':0.15}}
]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Product Recommender", layout="centered")
st.title("💼 Financial Product Recommender")

st.markdown("### Personal Details")
age = st.slider("Age", 18, 100, 35)

st.markdown("---")
st.subheader("Financial Literacy Questionnaire")
fl_answers = {}
for item in financial_qs:
    fl_answers[item['q']] = st.radio(item['q'], list(item['opts'].keys()))

st.markdown("---")
st.subheader("Risk Propensity Questionnaire")
rp_answers = {}
for item in risk_qs:
    rp_answers[item['q']] = st.radio(item['q'], list(item['opts'].keys()))

if st.button("Get Recommendation"):
    # compute financial education score
    fin_vals = np.array([financial_qs[i]['opts'][ans] for i,ans in enumerate(fl_answers.values())])
    fin_score = fin_vals.mean()
    # compute risk from survey
    rp_vals = np.array([risk_qs[i]['opts'][ans] for i,ans in enumerate(rp_answers.values())])
    rp_score = rp_vals.mean()
    # model risk
    default = needs_df.copy().median()
    user = {'Age':age,'Gender':0,'FamilyMembers':1,'FinancialEducation':fin_score,
            'Income ': default['Income '],'Wealth':default['Wealth'],'RiskPropensity':rp_score}
    model_r = risk_model.predict(feature_engineering(pd.DataFrame([user]))[0])[0]
    # combine
    comb_r = 0.7*model_r + 0.3*rp_score
    # get recommendation
    rec_df = recommend({**user,'RiskPropensity':comb_r})
    st.markdown("### Results")
    st.write(f"- Financial Literacy score: {fin_score:.3f}")
    st.write(f"- Survey risk score: {rp_score:.3f}")
    st.write(f"- Model risk score: {model_r:.3f}")
    st.write(f"- Combined risk score: {comb_r:.3f}")
    st.table(rec_df)
```

