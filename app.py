import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Telco Churn App", layout="wide")

@st.cache_resource
def load_bundle():
    with open("model_bundle.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
    if isinstance(bundle, dict):
        model = bundle.get('model', list(bundle.values())[0])
        saved_features = bundle.get('features', None)
    else:
        model = bundle
        saved_features = None
        
    st.success("تم تحميل وتجهيز الموديل بنجاح!")
except Exception as e:
    st.error(f"خطأ في تحميل الموديل: {e}")
    st.stop()

st.title("Telco Customer Churn Prediction")
st.write("أدخل بيانات العميل لمعرفة احتمالية مغادرته للخدمة:")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("مدة الاشتراك بالشهور", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("المصاريف الشهرية", min_value=0.0, max_value=200.0, value=50.0)
        
    with col2:
        total_charges = st.number_input("إجمالي المصاريف", min_value=0.0, max_value=10000.0, value=600.0)
        contract = st.selectbox("نوع العقد", ["Month-to-month", "One year", "Two year"])

    submit_button = st.form_submit_button(label="توقع مغادرة العميل")

if submit_button:
    try:
        if saved_features is not None:
            # إنشاء جدول أصفار بنفس الأعمدة الأصلية التي يتوقعها الموديل
            input_data = pd.DataFrame(0, index=[0], columns=saved_features)
            
            # تعبئة القيم الأساسية إذا كانت متوفرة في الأعمدة
            if 'tenure' in input_data.columns:
                input_data['tenure'] = tenure
            if 'MonthlyCharges' in input_data.columns:
                input_data['MonthlyCharges'] = monthly_charges
            if 'TotalCharges' in input_data.columns:
                input_data['TotalCharges'] = total_charges
                
            # التعامل مع نوع العقد إذا كان تم عمل ترميز له
            contract_col = f"Contract_{contract}"
            if contract_col in input_data.columns:
                input_data[contract_col] = 1
            elif 'Contract' in input_data.columns:
                input_data['Contract'] = contract
        else:
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract]
            })
            input_data['Contract'] = input_data['Contract'].astype('category')
        
        # التنبؤ
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        st.subheader("نتائج التنبؤ:")
        if prediction[0] == 1:
            st.error(f"⚠️ هذا العميل معرض لمغادرة الخدمة! (الاحتمالية: {prediction_proba[0][1]:.2%})")
        else:
            st.success(f"✅ هذا العميل مستمر. (احتمالية البقاء: {prediction_proba[0][0]:.2%})")
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
