import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io

from categories import (
    MARITAL_STATUS, APPLICATION_MODE, COURSE, ATTENDANCE, QUALIFICATION,
    NACIONALITY, OCCUPATION, YES_NO, GENDER
)

st.set_page_config(page_title="Prediksi Dropout - Jaya Jaya Institut", page_icon="🎓", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# ----------------------------------------------------------------------------------
# Load model artifacts (path aman untuk Streamlit Cloud)
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required = ["model.joblib", "scaler.joblib", "feature_names.joblib"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        st.error(
            f"File model tidak ditemukan di `{MODEL_DIR}`: {missing}. "
            "Pastikan folder `model/` sudah ikut di-push ke repository GitHub."
        )
        st.stop()
    model = joblib.load(MODEL_DIR / "model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()


def risk_category(proba):
    """Kategori risiko berdasarkan probabilitas dropout, selaras dengan action items di README."""
    if proba >= 0.75:
        return "Sangat Tinggi", "🔴"
    elif proba >= 0.5:
        return "Tinggi", "🟠"
    elif proba >= 0.25:
        return "Sedang", "🟡"
    else:
        return "Rendah", "🟢"


def predict_batch(df_input: pd.DataFrame):
    """df_input harus memiliki seluruh kolom di feature_names (nilai mentah/numeric)."""
    df_use = df_input.copy()
    missing_cols = [c for c in feature_names if c not in df_use.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan pada data: {missing_cols}")
    df_use = df_use[feature_names]
    scaled = scaler.transform(df_use)
    proba = model.predict_proba(scaled)[:, 1]  # probabilitas kelas Dropout (1)
    pred_label = np.where(proba >= 0.5, "Dropout", "Graduate")
    return pred_label, proba


# ----------------------------------------------------------------------------------
# HEADER dengan logo institusi
# ----------------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown(
        """
        <div style="width:64px;height:64px;border-radius:14px;
        background:linear-gradient(135deg,#3b82f6,#a855f7);
        display:flex;align-items:center;justify-content:center;font-size:32px;">🎓</div>
        """,
        unsafe_allow_html=True,
    )
with col_title:
    st.title("Jaya Jaya Institut — Sistem Prediksi Dropout Mahasiswa")
    st.caption("Prototype Machine Learning untuk deteksi dini risiko dropout mahasiswa")

tab_manual, tab_batch, tab_guide = st.tabs(["🧍 Prediksi Individu", "📂 Prediksi Massal (Upload File)", "📖 Panduan Penggunaan"])

# ====================================================================================
# TAB 1 — PREDIKSI INDIVIDU (form dengan label kategorikal, bukan angka mentah)
# ====================================================================================
with tab_manual:
    st.markdown(
        "Isi data mahasiswa di bawah ini untuk memprediksi apakah mahasiswa tersebut "
        "berpotensi **Dropout** atau **Graduate** (menyelesaikan studi)."
    )

    with st.form("prediction_form"):
        st.subheader("1. Data Demografis & Sosial-Ekonomi")
        c1, c2 = st.columns(2)
        with c1:
            Marital_status = st.selectbox("Status Pernikahan", list(MARITAL_STATUS.keys()), format_func=lambda x: MARITAL_STATUS[x])
            Gender = st.selectbox("Jenis Kelamin", list(GENDER.keys()), format_func=lambda x: GENDER[x])
            Age_at_enrollment = st.number_input("Usia saat Mendaftar", min_value=15, max_value=70, value=20)
            International = st.selectbox("Mahasiswa Internasional?", list(YES_NO.keys()), format_func=lambda x: YES_NO[x])
            Nacionality = st.selectbox("Kewarganegaraan", list(NACIONALITY.keys()), format_func=lambda x: NACIONALITY[x])
            Displaced = st.selectbox("Displaced (Pindah Domisili)?", list(YES_NO.keys()), format_func=lambda x: YES_NO[x])
            Educational_special_needs = st.selectbox("Kebutuhan Pendidikan Khusus?", list(YES_NO.keys()), format_func=lambda x: YES_NO[x])
        with c2:
            Mothers_qualification = st.selectbox("Kualifikasi Ibu", list(QUALIFICATION.keys()), format_func=lambda x: QUALIFICATION[x], index=list(QUALIFICATION.keys()).index(19))
            Fathers_qualification = st.selectbox("Kualifikasi Ayah", list(QUALIFICATION.keys()), format_func=lambda x: QUALIFICATION[x], index=list(QUALIFICATION.keys()).index(19))
            Mothers_occupation = st.selectbox("Pekerjaan Ibu", list(OCCUPATION.keys()), format_func=lambda x: OCCUPATION[x], index=list(OCCUPATION.keys()).index(5))
            Fathers_occupation = st.selectbox("Pekerjaan Ayah", list(OCCUPATION.keys()), format_func=lambda x: OCCUPATION[x], index=list(OCCUPATION.keys()).index(5))

        st.subheader("2. Data Finansial")
        c3, c4 = st.columns(2)
        with c3:
            Debtor = st.selectbox("Memiliki Tunggakan (Debtor)?", list(YES_NO.keys()), format_func=lambda x: YES_NO[x])
            Tuition_fees_up_to_date = st.selectbox("UKT Lunas Tepat Waktu?", list(YES_NO.keys()), format_func=lambda x: YES_NO[x], index=0)
        with c4:
            Scholarship_holder = st.selectbox("Penerima Beasiswa?", list(YES_NO.keys()), format_func=lambda x: YES_NO[x])

        st.subheader("3. Data Pendaftaran")
        c5, c6 = st.columns(2)
        with c5:
            Application_mode = st.selectbox("Mode Pendaftaran", list(APPLICATION_MODE.keys()), format_func=lambda x: APPLICATION_MODE[x])
            Application_order = st.number_input("Urutan Pilihan Pendaftaran (0=pilihan pertama)", min_value=0, max_value=9, value=1)
            Course = st.selectbox("Program Studi", list(COURSE.keys()), format_func=lambda x: COURSE[x])
            Daytime_evening_attendance = st.selectbox("Waktu Kelas", list(ATTENDANCE.keys()), format_func=lambda x: ATTENDANCE[x])
        with c6:
            Previous_qualification = st.selectbox("Kualifikasi Sebelumnya", list(QUALIFICATION.keys()), format_func=lambda x: QUALIFICATION[x])
            Previous_qualification_grade = st.number_input("Nilai Kualifikasi Sebelumnya (0-200)", min_value=0.0, max_value=200.0, value=130.0)
            Admission_grade = st.number_input("Nilai Ujian Masuk (0-200)", min_value=0.0, max_value=200.0, value=130.0)

        st.subheader("4. Performa Akademik Semester 1")
        c7, c8 = st.columns(2)
        with c7:
            Curricular_units_1st_sem_credited = st.number_input("SKS Diakui Sem. 1", min_value=0, max_value=30, value=0)
            Curricular_units_1st_sem_enrolled = st.number_input("SKS Diambil Sem. 1", min_value=0, max_value=30, value=6)
            Curricular_units_1st_sem_evaluations = st.number_input("Jumlah Evaluasi Sem. 1", min_value=0, max_value=50, value=8)
        with c8:
            Curricular_units_1st_sem_approved = st.number_input("SKS Lulus Sem. 1", min_value=0, max_value=30, value=5)
            Curricular_units_1st_sem_grade = st.number_input("Rata-rata Nilai Sem. 1 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
            Curricular_units_1st_sem_without_evaluations = st.number_input("SKS Tanpa Evaluasi Sem. 1", min_value=0, max_value=20, value=0)

        st.subheader("5. Performa Akademik Semester 2")
        c9, c10 = st.columns(2)
        with c9:
            Curricular_units_2nd_sem_credited = st.number_input("SKS Diakui Sem. 2", min_value=0, max_value=30, value=0)
            Curricular_units_2nd_sem_enrolled = st.number_input("SKS Diambil Sem. 2", min_value=0, max_value=30, value=6)
            Curricular_units_2nd_sem_evaluations = st.number_input("Jumlah Evaluasi Sem. 2", min_value=0, max_value=50, value=8)
        with c10:
            Curricular_units_2nd_sem_approved = st.number_input("SKS Lulus Sem. 2", min_value=0, max_value=30, value=5)
            Curricular_units_2nd_sem_grade = st.number_input("Rata-rata Nilai Sem. 2 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
            Curricular_units_2nd_sem_without_evaluations = st.number_input("SKS Tanpa Evaluasi Sem. 2", min_value=0, max_value=20, value=0)

        st.subheader("6. Indikator Makroekonomi (saat mendaftar)")
        c11, c12, c13 = st.columns(3)
        with c11:
            Unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=30.0, value=11.0)
        with c12:
            Inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=-10.0, max_value=10.0, value=1.0)
        with c13:
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
        input_df = pd.DataFrame([input_dict])
        pred_label, proba = predict_batch(input_df)
        pred_label, proba = pred_label[0], proba[0]
        cat, icon = risk_category(proba)

        st.divider()
        st.subheader("Hasil Prediksi")

        colr1, colr2 = st.columns([1, 1])
        with colr1:
            if pred_label == "Dropout":
                st.error(f"⚠️ Mahasiswa diprediksi berstatus: **{pred_label}**")
            else:
                st.success(f"🎉 Mahasiswa diprediksi berstatus: **{pred_label}**")
            st.metric("Probabilitas Dropout", f"{proba*100:.1f}%")
            st.markdown(f"**Kategori Risiko:** {icon} {cat}")
        with colr2:
            st.progress(min(max(proba, 0.0), 1.0))
            st.caption("Skala probabilitas dropout (0% = pasti Graduate, 100% = pasti Dropout)")

        if cat in ("Tinggi", "Sangat Tinggi"):
            st.info(
                "**Rekomendasi tindak lanjut:** Jadwalkan sesi konseling akademik dalam 2 minggu, "
                "tinjau status pembayaran UKT/kelayakan beasiswa, dan pantau nilai semester berjalan secara berkala "
                "(lihat Panduan Penggunaan → Ambang Batas Risiko)."
            )

# ====================================================================================
# TAB 2 — MASS UPLOAD (CSV / Excel)
# ====================================================================================
with tab_batch:
    st.markdown(
        "Gunakan fitur ini untuk memprediksi **banyak mahasiswa sekaligus** (misalnya seluruh mahasiswa "
        "berstatus *Enrolled* di akhir semester). Unggah file **CSV atau Excel (.xlsx)** yang memuat seluruh "
        "kolom fitur yang dibutuhkan model."
    )

    template_df = pd.DataFrame([{f: 0 for f in feature_names}])
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh Template CSV Kosong",
        data=csv_template,
        file_name="template_prediksi_mahasiswa.csv",
        mime="text/csv",
    )

    enrolled_path = MODEL_DIR / "enrolled_students_for_prediction.csv"
    if enrolled_path.exists():
        with open(enrolled_path, "rb") as f:
            st.download_button(
                "⬇️ Unduh Contoh Data Mahasiswa Aktif (Enrolled) untuk Dicoba",
                data=f,
                file_name="contoh_data_enrolled.csv",
                mime="text/csv",
            )
        st.caption(
            "File contoh ini berisi 794 mahasiswa berstatus *Enrolled* (masih aktif kuliah, belum ada label akhir) "
            "dari dataset asli — cocok untuk mensimulasikan penggunaan nyata prototype ini."
        )

    uploaded_file = st.file_uploader("Upload file CSV / Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, sep=None, engine="python")
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.write(f"Data terbaca: **{raw_df.shape[0]} baris, {raw_df.shape[1]} kolom**")

            missing_cols = [c for c in feature_names if c not in raw_df.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan pada file yang diunggah: {missing_cols}")
            else:
                pred_labels, probas = predict_batch(raw_df)
                result_df = raw_df.copy()
                result_df["Prediksi_Status"] = pred_labels
                result_df["Probabilitas_Dropout(%)"] = (probas * 100).round(1)
                result_df["Kategori_Risiko"] = [risk_category(p)[0] for p in probas]

                st.success(f"Prediksi selesai untuk {len(result_df)} mahasiswa.")

                summary_cols = st.columns(4)
                for col, cat_name, color in zip(
                    summary_cols, ["Sangat Tinggi", "Tinggi", "Sedang", "Rendah"], ["🔴", "🟠", "🟡", "🟢"]
                ):
                    count = (result_df["Kategori_Risiko"] == cat_name).sum()
                    col.metric(f"{color} {cat_name}", int(count))

                display_cols = ["Prediksi_Status", "Probabilitas_Dropout(%)", "Kategori_Risiko"] + \
                    [c for c in raw_df.columns if c not in feature_names]
                st.dataframe(
                    result_df.sort_values("Probabilitas_Dropout(%)", ascending=False),
                    use_container_width=True,
                    height=420,
                )

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_dropout.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

# ====================================================================================
# TAB 3 — PANDUAN PENGGUNAAN
# ====================================================================================
with tab_guide:
    st.markdown("""
## 📖 Panduan Penggunaan Prototype

### 1. Prediksi Individu (Tab "🧍 Prediksi Individu")
1. Isi seluruh field pada form sesuai data mahasiswa yang ingin diprediksi (demografi, finansial, data pendaftaran, performa akademik semester 1 & 2, indikator makroekonomi).
2. Semua kolom kategorikal (Status Pernikahan, Mode Pendaftaran, Program Studi, dsb.) sudah ditampilkan dalam bentuk **label yang mudah dibaca**, bukan kode angka mentah — cukup pilih dari dropdown.
3. Klik tombol **"🔍 Prediksi"**.
4. Sistem akan menampilkan status prediksi (**Dropout** / **Graduate**), probabilitas dropout dalam persen, dan kategori risiko.

### 2. Prediksi Massal / Mass Upload (Tab "📂 Prediksi Massal")
Gunakan tab ini apabila Anda ingin memprediksi **banyak mahasiswa sekaligus** (misalnya seluruh mahasiswa aktif di akhir semester):
1. Unduh **Template CSV Kosong** untuk melihat kolom-kolom yang dibutuhkan, atau gunakan **Contoh Data Mahasiswa Enrolled** yang sudah disediakan.
2. Isi file tersebut dengan data mahasiswa (satu baris = satu mahasiswa) menggunakan **kode numerik asli** sesuai dataset (bukan label) untuk kolom kategorikal — lihat tabel referensi kode di bagian bawah panduan ini.
3. Upload file (format `.csv` atau `.xlsx`) melalui tombol **"Upload file CSV / Excel"**.
4. Sistem akan menampilkan hasil prediksi untuk seluruh baris data, lengkap dengan probabilitas dan kategori risiko, serta ringkasan jumlah mahasiswa per kategori risiko.
5. Unduh hasil prediksi lengkap melalui tombol **"Unduh Hasil Prediksi (CSV)"**.

### 3. Kategori & Ambang Batas Risiko
| Probabilitas Dropout | Kategori Risiko | Tindakan yang Disarankan |
|---|---|---|
| ≥ 75% | 🔴 Sangat Tinggi | Konseling akademik wajib dalam 2 minggu, tinjau status finansial (UKT/beasiswa) |
| 50% – 74% | 🟠 Tinggi | Pendampingan akademik rutin, monitoring nilai per bulan |
| 25% – 49% | 🟡 Sedang | Pemantauan berkala, dorong keikutsertaan bimbingan belajar |
| < 25% | 🟢 Rendah | Pemantauan standar |

*(Ambang batas ini diselaraskan dengan temuan pada notebook: mahasiswa dengan ≤2 SKS lulus per semester atau nilai rata-rata ≤10 memiliki probabilitas dropout riil di atas 90%.)*

### 4. Hal Penting yang Harus Diperhatikan
- Model dilatih **hanya** menggunakan data mahasiswa berstatus final (**Dropout** atau **Graduate**). Mahasiswa berstatus **Enrolled** tidak digunakan untuk melatih model, melainkan menjadi target prediksi (use case nyata dari prototype ini).
- Nilai (`grade`) menggunakan skala **0–20** (standar akademik Portugal), sedangkan nilai ujian masuk (`Admission_grade`) dan nilai kualifikasi sebelumnya menggunakan skala **0–200**.
- Untuk prediksi massal, pastikan **nama kolom** pada file yang diunggah **persis sama** dengan kolom pada dataset asli (lihat template).
- Hasil prediksi bersifat **prediktif/probabilistik**, bukan kepastian mutlak — tetap perlu penilaian dan konfirmasi oleh dosen wali/pembimbing akademik sebelum mengambil tindakan.

### 5. Tabel Referensi Kode Kategorikal (untuk Upload File)
""")
    with st.expander("Lihat tabel kode Status Pernikahan, Mode Pendaftaran, Program Studi, dll."):
        st.write("**Status Pernikahan (Marital_status)**")
        st.table(pd.DataFrame(MARITAL_STATUS.items(), columns=["Kode", "Label"]))
        st.write("**Mode Pendaftaran (Application_mode)**")
        st.table(pd.DataFrame(APPLICATION_MODE.items(), columns=["Kode", "Label"]))
        st.write("**Program Studi (Course)**")
        st.table(pd.DataFrame(COURSE.items(), columns=["Kode", "Label"]))
        st.write("**Kualifikasi Pendidikan (Previous/Mothers/Fathers qualification)**")
        st.table(pd.DataFrame(QUALIFICATION.items(), columns=["Kode", "Label"]))
        st.write("**Pekerjaan Orang Tua (Mothers/Fathers occupation)**")
        st.table(pd.DataFrame(OCCUPATION.items(), columns=["Kode", "Label"]))
        st.write("**Kewarganegaraan (Nacionality)**")
        st.table(pd.DataFrame(NACIONALITY.items(), columns=["Kode", "Label"]))
        st.write("**Ya/Tidak** — dipakai untuk Displaced, Educational_special_needs, Debtor, Tuition_fees_up_to_date, Scholarship_holder, International")
        st.table(pd.DataFrame(YES_NO.items(), columns=["Kode", "Label"]))
        st.write("**Jenis Kelamin (Gender)**")
        st.table(pd.DataFrame(GENDER.items(), columns=["Kode", "Label"]))
        st.write("**Waktu Kelas (Daytime_evening_attendance)**")
        st.table(pd.DataFrame(ATTENDANCE.items(), columns=["Kode", "Label"]))

st.divider()
st.caption("Prototype Machine Learning — Proyek Akhir Data Science, Jaya Jaya Institut. Model: Random Forest Classifier (Dropout vs Graduate).")
