import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import uuid
import os

# 获取当前 app.py 文件所在的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建路径
model_path = os.path.join(BASE_DIR, "models", "best_lr_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "minmax_scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "selected_features.pkl")
columns_path = os.path.join(BASE_DIR, "models", "reference_columns.pkl")
data_path = os.path.join(BASE_DIR, "data", "final_data_11.xlsx")
shap_output_dir = os.path.join(BASE_DIR, "shap_outputs")
os.makedirs(shap_output_dir, exist_ok=True)

# 加载模型与预处理器
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
selected_features = joblib.load(features_path)
reference_columns = joblib.load(columns_path)
original_df = pd.read_excel(data_path)

# Streamlit 页面配置
st.set_page_config(page_title="CLNM 风险预测", layout="centered")
st.title("甲状腺癌中央区淋巴结转移风险预测")
st.write("请填写下列患者特征以预测CLNM的风险。")

# 用户输入界面
sex = st.selectbox("性别", ["Male", "Female"])
race = st.selectbox("种族", ["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"])
t_stage = st.selectbox("T 分期", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
hist = st.selectbox("组织学类型", ["PTC", "FTC", "ATC"])
laterality = st.selectbox("肿瘤侧位", ["Left", "Right", "Bilateral"])
marital_status = st.selectbox("婚姻状态", ["Married", "Single", "Divorced", "Widowed", "Separated"])
age = st.slider("年龄", 10, 90, 45)
tumor_size = st.number_input("肿瘤大小（mm）", min_value=1.0, max_value=200.0, value=15.0)

# 构造输入字典
input_dict = {
    "Sex": sex,
    "Race": race,
    "T stage": t_stage,
    "ICD-O-3 Hist": hist,
    "Laterality": laterality,
    "Marital Status": marital_status,
    "Age": age,
    "Tumor Size": tumor_size
}

# 模型预测函数
def predict_new_patient(input_dict):
    new_data = pd.DataFrame([input_dict])
    new_data_encoded = pd.get_dummies(new_data)
    for col in reference_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[reference_columns]

    # 保留未标准化数据用于 SHAP 展示
    original_values = new_data_encoded.copy()

    # 归一化 Age 用于模型预测
    if 'Age' in new_data_encoded.columns:
        Q1 = original_df['Age'].quantile(0.25)
        Q3 = original_df['Age'].quantile(0.75)
        IQR = Q3 - Q1
        age_val = np.clip(new_data_encoded['Age'].values[0], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        new_data_encoded['Age'] = scaler.transform([[age_val]])[0][0]

    # 选择模型特征
    final_input = new_data_encoded[selected_features]
    probability = model.predict_proba(final_input)[:, 1][0]

    # SHAP 决策图（使用未标准化的值）
    final_input_original = original_values[selected_features]

    explainer = shap.Explainer(model, final_input)
    shap_values = explainer(final_input)

    decision_img_path = os.path.join(shap_output_dir, f"shap_decision_{uuid.uuid4().hex}.png")
    shap.decision_plot(
        base_value=explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        shap_values=shap_values.values[0],
        features=final_input_original.iloc[0],
        feature_names=final_input_original.columns.tolist(),
        show=False
    )
    plt.tight_layout()
    plt.savefig(decision_img_path, dpi=300)
    plt.close()

    return probability, decision_img_path

# 预测按钮
if st.button("预测CLNM风险概率"):
    prob, shap_decision_img = predict_new_patient(input_dict)
    st.success(f"预测CLNM风险概率为：{prob:.4f}")
    st.subheader("SHAP 决策图（逐步解释）")
    st.image(shap_decision_img)
