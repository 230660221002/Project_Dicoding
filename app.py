import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Dropout - Jaya Jaya Institut", page_icon="🎓", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/model.joblib")
    scaler = joblib.load("model/scaler.joblib")
    le = joblib.load("model/label_encoder.joblib")
    feature_names = joblib.load("model/feature_names.joblib")
    return model, scaler, le, feature_names

model, scaler, le, feature_names = load_artifacts()

st.title("🎓 Prototype Prediksi Status Mahasiswa")
st.markdown(
    "Aplikasi ini membantu **Jaya Jaya Institut** memprediksi apakah seorang mahasiswa "
    "berpotensi **Dropout**, tetap **Enrolled**, atau **Graduate**, berdasarkan data akademik "
    "dan sosio-demografis. Isi form di bawah ini lalu klik **Prediksi**."
)

st.divider()

with st.form("prediction_form"):
    st.subheader("1. Data Demografis & Sosial-Ekonomi")
    col1, col2 = st.columns(2)
    with col1:
        Marital_status = st.selectbox("Status Pernikahan (1=Single,2=Married,3=Widower,4=Divorced,5=Facto Union,6=Legally Separated)", [1,2,3,4,5,6])
        Gender = st.selectbox("Jenis Kelamin", [1,0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
        Age_at_enrollment = st.number_input("Usia saat Mendaftar", min_value=15, max_value=70, value=20)
        International = st.selectbox("Mahasiswa Internasional?", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        Nacionality = st.number_input("Kode Kewarganegaraan", min_value=1, max_value=200, value=1)
        Displaced = st.selectbox("Displaced (pindah domisili)?", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        Educational_special_needs = st.selectbox("Kebutuhan Khusus?", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")
    with col2:
        Mothers_qualification = st.number_input("Kode Kualifikasi Ibu", min_value=1, max_value=44, value=19)
        Fathers_qualification = st.number_input("Kode Kualifikasi Ayah", min_value=1, max_value=44, value=19)
        Mothers_occupation = st.number_input("Kode Pekerjaan Ibu", min_value=0, max_value=200, value=5)
        Fathers_occupation = st.number_input("Kode Pekerjaan Ayah", min_value=0, max_value=200, value=5)

    st.subheader("2. Data Finansial")
    col3, col4 = st.columns(2)
    with col3:
        Debtor = st.selectbox("Memiliki Tunggakan (Debtor)?", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        Tuition_fees_up_to_date = st.selectbox("UKT Lunas Tepat Waktu?", [1,0], format_func=lambda x: "Ya" if x==1 else "Tidak")
    with col4:
        Scholarship_holder = st.selectbox("Penerima Beasiswa?", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")

    st.subheader("3. Data Pendaftaran")
    col5, col6 = st.columns(2)
    with col5:
        Application_mode = st.number_input("Kode Mode Pendaftaran", min_value=1, max_value=60, value=17)
        Application_order = st.number_input("Urutan Pilihan Pendaftaran", min_value=0, max_value=9, value=1)
        Course = st.number_input("Kode Program Studi", min_value=1, max_value=9999, value=171)
        Daytime_evening_attendance = st.selectbox("Kelas", [1,0], format_func=lambda x: "Siang" if x==1 else "Malam")
    with col6:
        Previous_qualification = st.number_input("Kode Kualifikasi Sebelumnya", min_value=1, max_value=44, value=1)
        Previous_qualification_grade = st.number_input("Nilai Kualifikasi Sebelumnya (0-200)", min_value=0.0, max_value=200.0, value=130.0)
        Admission_grade = st.number_input("Nilai Ujian Masuk (0-200)", min_value=0.0, max_value=200.0, value=130.0)

    st.subheader("4. Performa Akademik Semester 1")
    col7, col8 = st.columns(2)
    with col7:
        Curricular_units_1st_sem_credited = st.number_input("SKS Diakui Sem. 1", min_value=0, max_value=30, value=0)
        Curricular_units_1st_sem_enrolled = st.number_input("SKS Diambil Sem. 1", min_value=0, max_value=30, value=6)
        Curricular_units_1st_sem_evaluations = st.number_input("Jumlah Evaluasi Sem. 1", min_value=0, max_value=50, value=8)
    with col8:
        Curricular_units_1st_sem_approved = st.number_input("SKS Lulus Sem. 1", min_value=0, max_value=30, value=5)
        Curricular_units_1st_sem_grade = st.number_input("Rata-rata Nilai Sem. 1 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
        Curricular_units_1st_sem_without_evaluations = st.number_input("SKS Tanpa Evaluasi Sem. 1", min_value=0, max_value=20, value=0)

    st.subheader("5. Performa Akademik Semester 2")
    col9, col10 = st.columns(2)
    with col9:
        Curricular_units_2nd_sem_credited = st.number_input("SKS Diakui Sem. 2", min_value=0, max_value=30, value=0)
        Curricular_units_2nd_sem_enrolled = st.number_input("SKS Diambil Sem. 2", min_value=0, max_value=30, value=6)
        Curricular_units_2nd_sem_evaluations = st.number_input("Jumlah Evaluasi Sem. 2", min_value=0, max_value=50, value=8)
    with col10:
        Curricular_units_2nd_sem_approved = st.number_input("SKS Lulus Sem. 2", min_value=0, max_value=30, value=5)
        Curricular_units_2nd_sem_grade = st.number_input("Rata-rata Nilai Sem. 2 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
        Curricular_units_2nd_sem_without_evaluations = st.number_input("SKS Tanpa Evaluasi Sem. 2", min_value=0, max_value=20, value=0)

    st.subheader("6. Indikator Makroekonomi (saat mendaftar)")
    col11, col12, col13 = st.columns(3)
    with col11:
        Unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=30.0, value=11.0)
    with col12:
        Inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=-10.0, max_value=10.0, value=1.0)
    with col13:
        GDP = st.number_input("GDP Growth", min_value=-10.0, max_value=10.0, value=0.0)

    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

if submitted:
    input_dict = {
        'Marital_status': Marital_status, 'Application_mode': Application_mode,
        'Application_order': Application_order, 'Course': Course,
        'Daytime_evening_attendance': Daytime_evening_attendance,
        'Previous_qualification': Previous_qualification,
        'Previous_qualification_grade': Previous_qualification_grade,
        'Nacionality': Nacionality, 'Mothers_qualification': Mothers_qualification,
        'Fathers_qualification': Fathers_qualification, 'Mothers_occupation': Mothers_occupation,
        'Fathers_occupation': Fathers_occupation, 'Admission_grade': Admission_grade,
        'Displaced': Displaced, 'Educational_special_needs': Educational_special_needs,
        'Debtor': Debtor, 'Tuition_fees_up_to_date': Tuition_fees_up_to_date,
        'Gender': Gender, 'Scholarship_holder': Scholarship_holder,
        'Age_at_enrollment': Age_at_enrollment, 'International': International,
        'Curricular_units_1st_sem_credited': Curricular_units_1st_sem_credited,
        'Curricular_units_1st_sem_enrolled': Curricular_units_1st_sem_enrolled,
        'Curricular_units_1st_sem_evaluations': Curricular_units_1st_sem_evaluations,
        'Curricular_units_1st_sem_approved': Curricular_units_1st_sem_approved,
        'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
        'Curricular_units_1st_sem_without_evaluations': Curricular_units_1st_sem_without_evaluations,
        'Curricular_units_2nd_sem_credited': Curricular_units_2nd_sem_credited,
        'Curricular_units_2nd_sem_enrolled': Curricular_units_2nd_sem_enrolled,
        'Curricular_units_2nd_sem_evaluations': Curricular_units_2nd_sem_evaluations,
        'Curricular_units_2nd_sem_approved': Curricular_units_2nd_sem_approved,
        'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
        'Curricular_units_2nd_sem_without_evaluations': Curricular_units_2nd_sem_without_evaluations,
        'Unemployment_rate': Unemployment_rate, 'Inflation_rate': Inflation_rate, 'GDP': GDP,
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    pred_label = le.inverse_transform([pred])[0]

    st.divider()
    st.subheader("Hasil Prediksi")

    if pred_label == "Dropout":
        st.error(f"⚠️ Mahasiswa diprediksi berstatus: **{pred_label}**")
    elif pred_label == "Enrolled":
        st.warning(f"📘 Mahasiswa diprediksi berstatus: **{pred_label}**")
    else:
        st.success(f"🎉 Mahasiswa diprediksi berstatus: **{pred_label}**")

    proba_df = pd.DataFrame({
        "Status": le.classes_,
        "Probabilitas": pred_proba
    }).sort_values("Probabilitas", ascending=False)
    st.bar_chart(proba_df.set_index("Status"))
    st.dataframe(proba_df, use_container_width=True, hide_index=True)

    if pred_label == "Dropout":
        st.info(
            "**Rekomendasi:** Mahasiswa ini disarankan mendapat bimbingan akademik/konseling "
            "tambahan segera, serta ditinjau status pembayaran UKT dan kelayakan beasiswa."
        )

st.divider()
st.caption("Prototype Machine Learning — Proyek Akhir Data Science, Jaya Jaya Institut.")
