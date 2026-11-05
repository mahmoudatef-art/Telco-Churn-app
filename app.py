import streamlit as st
import pandas as pd
import joblib

# إعدادات صفحة التطبيق
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="centered")

# تحميل الموديل بـ caching عشان الأداء
@st.cache_resource
def load_model():
    return joblib.load("model_bundle.pkl")

try:
    model = load_model()
    st.success("تم تحميل وتجهيز الموديل بنجاح")
except Exception as e:
    st.error(f"خطأ في تحميل الموديل: {e}")

st.title("Telco Customer Churn Prediction")
st.write("أدخل بيانات العميل لمعرفة احتمالية مغادرته للخدمة")

# مدخلات المستخدم الأساسية
tenure = st.number_input("مدة الإقامة بالشهور", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("المصاريف الشهرية", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("إجمالي المصاريف", min_value=0.0, max_value=10000.0, value=500.0)
contract = st.selectbox("نوع العقد", ["Month-to-month", "One year", "Two year"])

# زر التنبؤ
if st.button("توقع مغادرة العميل"):
    try:
        # تجهيز البيانات في DataFrame
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract]
        })
        
        # تحويل نوع عمود Contract لـ category لتجنب أي مشاكل
        input_data['Contract'] = input_data['Contract'].astype('category')

        # ضبط الأعمدة أوتوماتيك لتتوافق تماماً مع ما يطلبه الموديل دون أخطاء
        if hasattr(model, "feature_names_in_"):
            input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        # التنبؤ
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("نتائج التنبؤ:")
        if prediction[0] == 1:
            st.error(f"⚠️ هذا العميل معرض لمغادرة الخدمة ! (احتمالية الغاء الاهتمام: {prediction_proba[0][1]:.2%})")
        else:
            st.success(f"✅ هذا العميل مستمر (احتمالية البقاء: {prediction_proba[0][0]:.2%})")

    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
