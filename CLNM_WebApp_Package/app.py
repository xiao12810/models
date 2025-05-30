import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 路径初始化
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "best_lr_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "minmax_scaler.pkl"))
selected_features = joblib.load(os.path.join(BASE_DIR, "models", "selected_features.pkl"))
reference_columns = joblib.load(os.path.join(BASE_DIR, "models", "reference_columns.pkl"))
original_df = pd.read_excel(os.path.join(BASE_DIR, "data", "final_data_11.xlsx"))

# Streamlit 配置
st.set_page_config(page_title="CLNM 风险预测", layout="centered")

# 美化标题
st.markdown(
    """
    <h3 style='text-align: center; font-family: Arial;'>甲状腺癌中央区淋巴结转移风险预测</h3>
    """,
    unsafe_allow_html=True
)

# 用户输入
sex = st.selectbox("性别", ["Male", "Female"])
race = st.selectbox(" 种族", ["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"])
t_stage = st.selectbox("T 分期", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
hist = st.selectbox(" 组织学类型", ["PTC", "FTC", "ATC"])
age = st.slider(" 年龄", 10, 90, 45)

# 构造输入字典
input_dict = {
    "Sex": sex,
    "Race": race,
    "T stage": t_stage,
    "ICD-O-3 Hist": hist,
    "Age": age
}

# 预测函数
def predict_new_patient(input_dict):
    new_data = pd.DataFrame([input_dict])
    new_data_encoded = pd.get_dummies(new_data)
    for col in reference_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[reference_columns]

    if 'Age' in new_data_encoded.columns:
        Q1 = original_df['Age'].quantile(0.25)
        Q3 = original_df['Age'].quantile(0.75)
        IQR = Q3 - Q1
        age_val = np.clip(new_data_encoded['Age'].values[0], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        new_data_encoded['Age'] = scaler.transform([[age_val]])[0][0]

    final_input = new_data_encoded[selected_features]
    probability = model.predict_proba(final_input)[:, 1][0]
    return probability

# 预测按钮
if st.button(" 预测CLNM风险概率"):
    prob = predict_new_patient(input_dict)
    st.markdown(
        f"<div style='text-align: center; font-size: 32px; font-weight: bold; color: green;'>预测 CLNM 概率为：{prob:.4f}</div>",
        unsafe_allow_html=True
    )
