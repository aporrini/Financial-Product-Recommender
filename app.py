# app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import requests
import joblib

# Ensure unpickling works by importing all needed classes:
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from pyexcel_xls import get_data

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Financial Product Recommender", layout="centered")

# --- 1) LOAD MODELS ---
@st.cache_resource
def load_models():
    base_url = "https://raw.githubusercontent.com/aporrini/Financial-Product-Recommender/main/models/"
    def _ld(name):
        b = requests.get(base_url + name).content
        return joblib.load(io.BytesIO(b))
    risk   = _ld("risk_model.pkl")
    inc    = _ld("stack_income.pkl")
    accum  = _ld("stack_accum.pkl")
    return risk, inc, accum, 0.5, 0.5

risk_model, stack_income, stack_accum, thr_i, thr_a = load_models()

# --- 2) LOAD & AUGMENT DATA ---
@st.cache_data
def load_data():
    url = (
      "https://raw.githubusercontent.com/aporrini/Financial-Product-Recommender/"
      "release-1.0/Dataset2_Needs.xls"
    )
    xls = get_data(io.BytesIO(requests.get(url).content))
    needs    = pd.DataFrame(xls["Needs"][1:],    columns=xls["Needs"][0])
    products = pd.DataFrame(xls["Products"][1:], columns=xls["Products"][0])

    # add your 5 new products
    new = pd.DataFrame({
      "IDProduct":[12,13,14,15,16],
      "Type":     [0,  0,  1,  1,  0],
      "Risk":     [0.55,0.70,0.70,0.15,0.85]
    })
    products = pd.concat([products, new], ignore_index=True)
    products = products[products["Risk"] != 0.12]

    # name map including new IDs
    name_map = {
      1:"Balanced Mutual Fund",
      2:"Income Conservative Unit-Linked (Life Insurance)",
      3:"Fixed Income Mutual Fund",
      4:"Balanced High Dividend Mutual Fund",
      5:"Balanced Mutual Fund",
      6:"Defensive Flexible Allocation Unit-Linked (Life Insurance)",
      7:"Aggressive Flexible Allocation Unit-Linked (Life Insurance)",
      8:"Balanced Flexible Allocation Unit-Linked (Life Insurance)",
      9:"Cautious Allocation Segregated Account",
      10:"Fixed Income Segregated Account",
      11:"Total Return Aggressive Allocation Segregated Account",
      12:"Global Diversified Income Fund",
      13:"Emerging Markets High Yield Bond Fund",
      14:"Sustainable Growth Equity Portfolio",
      15:"Short-Term Government Bond Accumulation Fund",
      16:"Tranche Equity CDO"
    }
    products["ProductName"] = products["IDProduct"].astype(int).map(name_map)
    return needs, products

needs_df, products_df = load_data()

# --- 3) FEATURE ENGINEERING ---
def feature_engineering(d: dict):
    a,g,f = d["Age"], d["Gender"], d["FamilyMembers"]
    edu,inc,wel,rp = d["FinancialEducation"], d["Income"], d["Wealth"], d["RiskPropensity"]
    return pd.DataFrame([{
      "Age":a, "Gender":g, "FamilyMembers":f, "FinancialEducation":edu, "RiskPropensity":rp,
      "Wealth_log":np.log1p(wel), "Income_log":np.log1p(inc),
      "Income_Wealth_Ratio_log": np.log1p(inc/wel) if wel>0 else np.log1p(inc),
      "Is_Single":int(f==1), "Is_Senior":int(a>65), "Has_Education":int(edu>0.1),
      "Risk_Age_Interaction":rp*a
    }])

# --- 4) RECOMMENDATION FUNCTION ---
def recommend_products(d: dict, eps=0.05):
    df = feature_engineering(d)
    pi = stack_income.predict_proba(df)[0,1]
    pa = stack_accum.predict_proba(df)[0,1]
    recs = []
    if pi>=thr_i:
        pool = products_df[(products_df["Type"]==0)&(products_df["Risk"]<=d["RiskPropensity"]+eps)]
        if not pool.empty:
            b = pool.loc[pool["Risk"].idxmax()]
            recs.append({"Type":"Income","ID":int(b.IDProduct),"Name":b.ProductName,"Risk":b.Risk,"Prob":round(pi,3)})
    if pa>=thr_a:
        pool = products_df[(products_df["Type"]==1)&(products_df["Risk"]<=d["RiskPropensity"]+eps)]
        if not pool.empty:
            b = pool.loc[pool["Risk"].idxmax()]
            recs.append({"Type":"Accumulation","ID":int(b.IDProduct),"Name":b.ProductName,"Risk":b.Risk,"Prob":round(pa,3)})
    if not recs:
        recs.append({"Type":"None","ID":0,"Name":"No Investment Needed","Risk":"-","Prob":"-"})
    return pd.DataFrame(recs)

# --- 5) YOUR QUESTIONS (verbatim) ---
financial_lit = [
  {'question':'What is your education title?',
   'options':{'No':0.0,'High School Diploma':0.015,'Bachelor Degree':0.025,
              'Bachelor Degree in economic/financial subjects':0.075,
              'Master Degree':0.05,'Master Degree in economic/financial subjects':0.1}},
  {'question':'Have you worked in the financial industry?','options':{'Yes':0.1,'No':0.0}},
  {'question':'Flag the most risky financial instruments in which you have invested',
   'options':{'Equity':0.04,'Mutual funds/Sicav/ETFs':0.015,'Bonds':0.02,
              'Government Bonds':0.015,
              'Structured Bonds (equity linked, reverse floater, reverse convertible)':0.06,
              'Insurance Products':0.008,'Covered Warrants/Warrants/Investment Certificates':0.06,
              'Portfolio Management':0.04,
              'Financial Derivatives (e.g. Options/Swaps/leveraged instruments)':0.1}},
  {'question':'With what frequency did you invest in financial products in the last 5 years?',
   'options':{'More than 10 times a year':0.1,'Between 5 and 10':0.05,'Less than 5':0.0}},
  {'question':'The rating is a score expressed by an independent third party entity that measures?',
   'options':{'The solidity of an enterprise':0.1,'The productivity rate of an enterprise':0.015,
              'The revenues of a company':0.0}},
  {'question':'What is an option?',
   'options':{'It is a financial contract whose value depends on the movements of an underlying asset':0.1,
              'An investment contract similar to equity and/or Bonds':0.06,
              'An instrument with guaranteed capital':0.0}},
  {'question':'What happens to the owners of subordinated bonds in insolvency of the issuer?',
   'options':{'They never get reimbursed':0.05,
              'They get reimbursed just after the owners of non-subordinated bonds':0.1,
              'They get reimbursed with stocks':0.0}},
  {'question':'What is a FX Swap?',
   'options':{'A swap on interest rates':0.01,
              'A product combining a spot and a forward currency contract':0.1,
              'Do not know':0.0}},
  {'question':'What is the frequency of publication of the NAV of Alternative funds?',
   'options':{'At least twice a year':0.1,'Daily':0.03,'Do not know':0.0}},
  {'question':'In a Credit Linked Note (CLN), what is the reimbursement of the capital tied to?',
   'options':{'The risk of default of the issuer':0.03,
              'The risk of default of the issuer and the reference entity':0.1,
              'The risk of default of the reference entity only':0.0}}
]

risk_qs = [
  {'question':'How would you react to a loss of 10% on your investment portfolio?',
   'options':{'I would sell everything':0.0,'I would wait and see what happens':0.12,
              'I would buy more':0.25}},
  {'question':'What is your investment goal on a 5 year horizon?',
   'options':{'Low returns but minimal risk of loss (gain 1%, loss 1%)':0.04,
              'Normal returns with limited loss (gain 5%, loss 5%)':0.1,
              'High return with high risk (gain 50%, loss 50%)':0.25}},
  {'question':'Which investment strategy aligns with your goals?',
   'options':{'Liquidity: protect capital (≤1 year horizon)':0.0,
              'Short term: protect capital with modest growth (≤3 years)':0.09,
              'Savings: high protection with growth (≤5 years)':0.15,
              'Long-medium term: significant growth (>5 years)':0.2,
              'Speculative':0.25}},
  {'question':'If a diversified portfolio showed -25% tech equities, -15% high-yield bonds, +5% commodities, what would you do?',
   'options':{'Rebalance towards defensive assets':0.04,'Buy more at lower prices':0.25,
              'Exit the markets':0.0,'Maintain original strategy':0.15}}
]

# --- 6) STREAMLIT UI ---
st.markdown("---")
st.header("🔹 Personal Profile")
age    = st.slider("Age",18,100,35)
gender = st.radio("Gender",["Male","Female"])
family = st.slider("Family Members",1,10,2)
income = st.number_input("Income (€)",0,200000,50000,1000)
wealth = st.number_input("Wealth (€)",0,500000,100000,1000)

st.markdown("---")
st.header("🔹 Financial Literacy Questionnaire")
fl_ans = {}
for q in financial_lit:
    fl_ans[q['question']] = st.selectbox(q['question'], list(q['options'].keys()))

st.markdown("---")
st.header("🔹 Risk Propensity Questionnaire")
rp_ans = {}
for q in risk_qs:
    rp_ans[q['question']] = st.selectbox(q['question'], list(q['options'].keys()))

if st.button("Get Recommendation"):
    lit_vals = np.array([q['options'][fl_ans[q['question']]] for q in financial_lit])
    lit_score = lit_vals.sum()
    rp_vals  = np.array([q['options'][rp_ans[q['question']]] for q in risk_qs])
    rp_score = rp_vals.sum()
    user = {
      "Age":age,"Gender":1 if gender=="Female" else 0,"FamilyMembers":family,
      "FinancialEducation":lit_score,"Income":income,"Wealth":wealth,"RiskPropensity":rp_score
    }
    model_r = risk_model.predict(feature_engineering(user))[0]
    comb_r   = 0.7*model_r + 0.3*rp_score
    st.subheader("📊 Scores")
    st.write(f"• Financial Literacy: {lit_score:.3f}")
    st.write(f"• Survey Risk      : {rp_score:.3f}")
    st.write(f"• Model Risk       : {model_r:.3f}")
    st.write(f"• Combined Risk    : {comb_r:.3f}")
    st.subheader("📈 Recommendations")
    st.table(recommend_products(user, epsilon=0.05))
