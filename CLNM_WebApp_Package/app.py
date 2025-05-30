import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
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

    if 'Age' in new_data_encoded.columns:
        Q1 = original_df['Age'].quantile(0.25)
        Q3 = original_df['Age'].quantile(0.75)
        IQR = Q3 - Q1
        age_val = np.clip(new_data_encoded['Age'].values[0], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        new_data_encoded['Age'] = scaler.transform([[age_val]])[0][0]

    final_input = new_data_encoded[selected_features]
    probability = model.predict_proba(final_input)[:, 1][0]

    # SHAP 分析
    explainer = shap.Explainer(model, final_input, feature_names=final_input.columns)
    shap_values = explainer(final_input)

    # 保存 force 图为 HTML
    force_html_filename = f"shap_force_{uuid.uuid4().hex}.html"
    force_html_path = os.path.join(shap_output_dir, force_html_filename)
    force_plot = shap.plots.force(shap_values[0], matplotlib=False)
    shap.save_html(force_html_path, force_plot)

    # 保存 waterfall 图为 PNG
    waterfall_img_filename = f"shap_waterfall_{uuid.uuid4().hex}.png"
    waterfall_img_path = os.path.join(shap_output_dir, waterfall_img_filename)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(waterfall_img_path, dpi=300)
    plt.close()

    return probability, force_html_path, waterfall_img_path

# 预测按钮
if st.button("预测CLNM风险概率"):
    prob, shap_force_html_path, shap_waterfall_img_path = predict_new_patient(input_dict)
    st.success(f"预测CLNM风险概率为：{prob:.4f}")

    # 显示 SHAP 力图
    st.subheader("SHAP 力图（局部解释）")
    with open(shap_force_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=400, scrolling=True)

    # 显示 SHAP Waterfall 图
    st.subheader("SHAP Waterfall 图（局部解释）")
    st.image(shap_waterfall_img_path)
