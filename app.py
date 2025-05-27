# app.py – Streamlit Financial Product Recommender with proper feature scaling
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os, io, requests, joblib
from xgboost import XGBRegressor, XGBClassifier  # for unpickling
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import MinMaxScaler
from pyexcel_xls import get_data

warnings.filterwarnings('ignore')
# Must be first Streamlit command
st.set_page_config(page_title="Financial Product Recommender", layout="centered")

# ------------------
# 1) LOAD MODELS
# ------------------
@st.cache_resource
def load_models():
    mdl_dir = os.path.join(os.path.dirname(__file__), 'models_trained')
    risk_model   = joblib.load(os.path.join(mdl_dir, 'risk_model.pkl'))
    stack_income = joblib.load(os.path.join(mdl_dir, 'stack_income.pkl'))
    stack_accum  = joblib.load(os.path.join(mdl_dir, 'stack_accum.pkl'))
    return risk_model, stack_income, stack_accum, 0.5, 0.5

risk_model, stack_income, stack_accum, thr_i, thr_a = load_models()

# ------------------
# 2) LOAD & PREP DATA
# ------------------
@st.cache_data
def load_data():
    # Load Needs and Products
    url = (
      'https://raw.githubusercontent.com/aporrini/Financial-Product-Recommender/'
      'release-1.0/Dataset2_Needs.xls'
    )
    content = requests.get(url).content
    xls = get_data(io.BytesIO(content))
    needs = pd.DataFrame(xls['Needs'][1:], columns=xls['Needs'][0])
    products = pd.DataFrame(xls['Products'][1:], columns=xls['Products'][0])
    # standardize Income column
    if 'Income ' in needs.columns:
        needs = needs.rename(columns={'Income ': 'Income'})

    # Append new products and clean
    extra = pd.DataFrame({
      'IDProduct':[12,13,14,15,16],
      'Type':[0,0,1,1,0],
      'Risk':[0.55,0.70,0.70,0.15,0.85]
    })
    products = pd.concat([products, extra], ignore_index=True)
    products = products[products['Risk'] != 0.12]

    # Map names
    name_map = {
      1:'Balanced Mutual Fund',2:'Income Conservative Unit-Linked (Life Insurance)',
      3:'Fixed Income Mutual Fund',4:'Balanced High Dividend Mutual Fund',
      5:'Balanced Mutual Fund',6:'Defensive Flexible Allocation Unit-Linked (Life Insurance)',
      7:'Aggressive Flexible Allocation Unit-Linked (Life Insurance)',
      8:'Balanced Flexible Allocation Unit-Linked (Life Insurance)',
      9:'Cautious Allocation Segregated Account',10:'Fixed Income Segregated Account',
      11:'Total Return Aggressive Allocation Segregated Account',
      12:'Global Diversified Income Fund',13:'Emerging Markets High Yield Bond Fund',
      14:'Sustainable Growth Equity Portfolio',
      15:'Short-Term Government Bond Accumulation Fund',16:'Tranche Equity CDO'
    }
    products['ProductName'] = products['IDProduct'].astype(int).map(name_map)
    return needs, products

needs_df, products_df = load_data()

# ------------------
# 3) SETUP SCALERS
# ------------------
# Define columns
base_cols = ['Age','Gender','FamilyMembers','FinancialEducation','Income','Wealth']
eng_cols = ['Age','Gender','FamilyMembers','FinancialEducation',
            'Wealth_log','Income_log','Income_Wealth_Ratio_log',
            'Is_Single','Is_Senior','Has_Education','Risk_Age_Interaction']

# Create training features to fit scalers
df = needs_df.copy()
# rename Income column if needed
df['Wealth_log'] = np.log1p(df['Wealth'])
df['Income_log'] = np.log1p(df['Income'])
ratio = df['Income'] / df['Wealth'].replace(0, np.nan)
df['Income_Wealth_Ratio_log'] = np.log1p(ratio.fillna(df['Income'].max()))
df['Is_Single'] = (df['FamilyMembers'] == 1).astype(int)
df['Is_Senior'] = (df['Age'] > 65).astype(int)
df['Has_Education'] = (df['FinancialEducation'] > 0).astype(int)
df['Risk_Age_Interaction'] = df['RiskPropensity'] * df['Age']

base_scaler = MinMaxScaler().fit(df[base_cols])
eng_scaler  = MinMaxScaler().fit(df[eng_cols])

# ------------------
# 4) FEATURE ENGINEERING
# ------------------
def make_features(d: dict):
    # Build raw features DataFrame
    raw = {
        'Age': d['Age'],
        'Gender': d['Gender'],
        'FamilyMembers': d['FamilyMembers'],
        'FinancialEducation': d['FinancialEducation'],
        'Income': d['Income'],
        'Wealth': d['Wealth']
    }
    X_raw = pd.DataFrame([raw])
    # Engineered columns
    X_raw['Wealth_log'] = np.log1p(X_raw['Wealth'])
    X_raw['Income_log'] = np.log1p(X_raw['Income'])
    ratio = X_raw['Income'] / X_raw['Wealth'].replace(0, np.nan)
    X_raw['Income_Wealth_Ratio_log'] = np.log1p(ratio.fillna(X_raw['Income'].max()))
    X_raw['Is_Single'] = (X_raw['FamilyMembers'] == 1).astype(int)
    X_raw['Is_Senior'] = (X_raw['Age'] > 65).astype(int)
    X_raw['Has_Education'] = (X_raw['FinancialEducation'] > 0).astype(int)
    X_raw['Risk_Age_Interaction'] = d['RiskPropensity'] * d['Age']
    # Scale base and engineered
    Xb_int = pd.DataFrame(base_scaler.transform(X_raw[base_cols]), columns=base_cols)
    Xe     = pd.DataFrame(eng_scaler.transform(X_raw[eng_cols]), columns=eng_cols)
    # Rename for risk_model (trained on 'Income ')
    Xb = Xb_int.rename(columns={'Income': 'Income '})
    return Xb, Xe

# ------------------
# 5) RECOMMEND
# ------------------
def recommend_products(d: dict, eps=0.05):
    Xb, Xe = make_features(d)
    pi = stack_income.predict_proba(Xe)[0,1]
    pa = stack_accum.predict_proba(Xe)[0,1]
    recs=[]
    if pi>=thr_i:
        pool = products_df[(products_df['Type']==0)&(products_df['Risk']<=d['RiskPropensity']+eps)]
        if not pool.empty:
            b = pool.loc[pool['Risk'].idxmax()]
            recs.append({'Type':'Income','ID':int(b.IDProduct),'Name':b.ProductName,'Risk':b.Risk,'Prob':round(pi,3)})
    if pa>=thr_a:
        pool = products_df[(products_df['Type']==1)&(products_df['Risk']<=d['RiskPropensity']+eps)]
        if not pool.empty:
            b = pool.loc[pool['Risk'].idxmax()]
            recs.append({'Type':'Accumulation','ID':int(b.IDProduct),'Name':b.ProductName,'Risk':b.Risk,'Prob':round(pa,3)})
    if not recs:
        recs.append({'Type':'None','ID':0,'Name':'No Investment Needed','Risk':'-','Prob':'-'})
    return pd.DataFrame(recs)

# ------------------
# 6) QUESTIONNAIRES (verbatim)
# ------------------
financial_lit = [
  {'question':'What is your education title?','options':{'No':0.0,'High School Diploma':0.015,'Bachelor Degree':0.025,'Bachelor Degree in economic/financial subjects':0.075,'Master Degree':0.05,'Master Degree in economic/financial subjects':0.1}},
  {'question':'Have you worked in the financial industry?','options':{'Yes':0.1,'No':0.0}},
  {'question':'Flag the most risky financial instruments in which you have invested','options':{'Equity':0.04,'Mutual funds/Sicav/ETFs':0.015,'Bonds':0.02,'Government Bonds':0.015,'Structured Bonds (equity linked, reverse floater, reverse convertible)':0.06,'Insurance Products':0.008,'Covered Warrants/Warrants/Investment Certificates':0.06,'Portfolio Management':0.04,'Financial Derivatives (e.g. Options/Swaps/leveraged instruments)':0.1}},
  {'question':'With what frequency did you invest in financial products in the last 5 years?','options':{'More than 10 times a year':0.1,'Between 5 and 10':0.05,'Less than 5':0.0}},
  {'question':'The rating is a score expressed by an independent third party entity that measures?','options':{'The solidity of an enterprise':0.1,'The productivity rate of an enterprise':0.015,'The revenues of a company':0.0}},
  {'question':'What is an option?','options':{'It is a financial contract whose value depends on the movements of an underlying asset':0.1,'An investment contract similar to equity and/or Bonds':0.06,'An instrument with guaranteed capital':0.0}},
  {'question':'What happens to the owners of subordinated bonds in insolvency of the issuer?','options':{'They never get reimbursed':0.05,'They get reimbursed just after the owners of non-subordinated bonds':0.1,'They get reimbursed with stocks':0.0}},
  {'question':'What is a FX Swap?','options':{'A swap on interest rates':0.01,'A product combining a spot and a forward currency contract':0.1,'Do not know':0.0}},
  {'question':'What is the frequency of publication of the NAV of Alternative funds?','options':{'At least twice a year':0.1,'Daily':0.03,'Do not know':0.0}},
  {'question':'In a Credit Linked Note (CLN), what is the reimbursement of the capital tied to?','options':{'The risk of default of the issuer':0.03,'The risk of default of the issuer and the reference entity':0.1,'The risk of default of the reference entity only':0.0}}
]

risk_qs = [
  {'question':'How would you react to a loss of 10% on your investment portfolio?','options':{'I would sell everything':0.0,'I would wait and see what happens':0.12,'I would buy more':0.25}},
  {'question':'What is your investment goal on a 5 year horizon?','options':{'Low returns but minimal risk of loss (gain 1%, loss 1%)':0.04,'Normal returns with limited loss (gain 5%, loss 5%)':0.1,'High return with high risk (gain 50%, loss 50%)':0.25}},
  {'question':'Which investment strategy aligns with your goals?','options':{'Liquidity: protect capital (≤1 year horizon)':0.0,'Short term: protect capital with modest growth (≤3 years)':0.09,'Savings: high protection with growth (≤5 years)':0.15,'Long-medium term: significant growth (>5 years)':0.2,'Speculative':0.25}},
  {'question':'If a diversified portfolio showed -25% tech equities, -15% high-yield bonds, +5% commodities, what would you do?','options':{'Rebalance towards defensive assets':0.04,'Buy more at lower prices':0.25,'Exit the markets':0.0,'Maintain original strategy':0.15}}
]

# ------------------
# 7) STREAMLIT UI
# ------------------
st.header("Financial Product Recommender")
st.subheader("MiFID Questionnaire")
col1, col2 = st.columns(2)
with col1:
    age    = st.slider("Age",18,100,35)
    gender = st.radio("Gender",["Male","Female"])
    family = st.slider("Family Members",1,10,2)
with col2:
    income = st.number_input("Income (€)",0,200000,50000,1000)
    wealth = st.number_input("Wealth (€)",0,500000,100000,1000)

st.divider()

fl_ans = {q['question']: st.selectbox(q['question'], list(q['options'].keys())) for q in financial_lit}
st.subheader("Risk Propensity Questionnaire")
rp_ans = {q['question']: st.selectbox(q['question'], list(q['options'].keys())) for q in risk_qs}

if st.button("Get Recommendation"):
    # compute questionnaire scores
    lit_score = sum(q['options'][fl_ans[q['question']]] for q in financial_lit)
    rp_score  = sum(q['options'][rp_ans[q['question']]] for q in risk_qs)
    # build user dict
    user = {
        'Age': age,
        'Gender': 1 if gender == 'Female' else 0,
        'FamilyMembers': family,
        'FinancialEducation': lit_score,
        'Income': income,
        'Wealth': wealth,
        'RiskPropensity': rp_score
    }
    # feature engineering and model predictions
    Xb, Xe = make_features(user)
    model_risk = risk_model.predict(Xb)[0]
    combined   = 0.7 * model_risk + 0.3 * rp_score

    # Convert to discrete MiFID II levels
    edu_lvl  = min(max(int(np.ceil(lit_score * 6)), 1), 6)
    risk_lvl = min(max(int(np.ceil(combined * 4)), 1), 4)

    # Display scores in a cleaner UI
    st.subheader("📊 Scores")
    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Literacy Score", f"{lit_score:.2f}", delta=None)
    scol2.metric("Survey Risk", f"{rp_score:.2f}", delta=None)
    scol3.metric("Model Risk", f"{model_risk:.2f}", delta=None)
    scol4.metric("Combined Risk", f"{combined:.2f}", delta=None)

    st.subheader("💡 MiFID II Levels")
    lcol1, lcol2 = st.columns(2)
    lcol1.metric("Financial Literacy Level", f"{edu_lvl} / 6")
    lcol2.metric("Risk Propensity Level", f"{risk_lvl} / 4")

    # Show recommendations table
    st.subheader("📈 Recommendations")
    rec_df = recommend_products({**user, 'RiskPropensity': combined})
    st.table(rec_df)
