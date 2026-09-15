# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan sebuah institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Hingga saat ini, Jaya Jaya Institut telah mencetak banyak lulusan dengan reputasi yang sangat baik. Namun, terdapat cukup banyak juga siswa yang tidak menyelesaikan pendidikannya alias mengalami *dropout*.

Tingginya angka dropout ini menjadi salah satu masalah besar bagi sebuah institusi pendidikan, karena berdampak pada reputasi, akreditasi, dan keberlanjutan finansial institusi. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin mahasiswa yang berpotensi melakukan dropout, sehingga dapat diberikan bimbingan khusus sejak dini.

### Permasalahan Bisnis
1. Tingkat dropout mahasiswa cukup tinggi, yaitu **32,1%** dari total 4.424 mahasiswa pada dataset (1.421 mahasiswa berstatus Dropout).
2. Institusi belum memiliki sistem yang dapat mendeteksi secara dini mahasiswa yang berisiko dropout, sehingga intervensi (bimbingan, konseling, keringanan biaya, dll) sering terlambat diberikan.
3. Belum tersedia media monitoring (dashboard) yang memudahkan manajemen memantau performa akademik mahasiswa dan faktor-faktor yang berkontribusi terhadap dropout.

### Cakupan Proyek
1. **Data Understanding & EDA** — mengeksplorasi data untuk menemukan pola dan faktor yang berkaitan dengan dropout.
2. **Data Preparation** — encoding, scaling, dan split data latih/uji.
3. **Modeling** — membangun model klasifikasi (Random Forest) untuk memprediksi status akhir mahasiswa (Dropout / Enrolled / Graduate).
4. **Evaluation** — mengevaluasi performa model dengan accuracy, F1-score, dan confusion matrix, serta menganalisis fitur paling berpengaruh.
5. **Business Dashboard** — membuat dashboard visual untuk membantu tim manajemen memonitor performa siswa.
6. **Prototype Machine Learning** — membangun aplikasi Streamlit agar prediksi dapat digunakan tanpa menjalankan kode.
7. **Conclusion & Action Items** — memberikan kesimpulan serta rekomendasi tindakan bagi Jaya Jaya Institut.

### Persiapan

**Sumber data:** [Students' Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance) (`data.csv`) — 4.424 baris data mahasiswa dengan atribut demografis, sosial-ekonomi, dan akademik, serta kolom target `Status` (Dropout/Enrolled/Graduate).

**Setup environment (Anaconda / local):**
```bash
conda create --name dropout-prediction python=3.11
conda activate dropout-prediction
pip install -r requirements.txt
```

**Setup environment (pipenv/venv):**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
streamlit==1.63.0
numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
joblib==1.5.3
```

**Menjalankan notebook:**
```bash
jupyter notebook notebook.ipynb
```
Seluruh tahapan proses data science — mulai dari *business understanding* hingga *conclusion* — didokumentasikan lengkap di dalam `notebook.ipynb`, termasuk proses penyimpanan model ke folder `model/` (`model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `feature_names.joblib`) yang digunakan oleh prototype Streamlit.

---

## Business Dashboard

Dashboard bisnis dibuat untuk membantu Jaya Jaya Institut memonitor performa dan risiko dropout mahasiswa, memuat informasi seperti:
- Distribusi status mahasiswa (Dropout / Enrolled / Graduate)
- Tingkat dropout berdasarkan status beasiswa
- Tingkat dropout berdasarkan status pelunasan UKT
- Distribusi nilai akademik per status mahasiswa
- Korelasi antar faktor akademik dan sosio-ekonomi

Data siap pakai untuk dashboard tersedia di file **`dashboard_data.csv`** (hasil olahan dari `data.csv`, sudah ditambah label yang mudah dibaca seperti `Gender_label`, `Scholarship_label`, `Tuition_paid_label`).

**Cara membuat dashboard (pilih salah satu):**

**Opsi A — Looker Studio (disarankan, tanpa Docker):**
1. Buka [Looker Studio](https://lookerstudio.google.com/), buat laporan baru.
2. Pilih sumber data **File Upload**, unggah `dashboard_data.csv`.
3. Buat visualisasi: pie/bar chart `Status`, bar chart dropout rate per `Scholarship_label`/`Tuition_paid_label`, histogram nilai semester per status, dsb (lihat referensi visual pada `model/eda_*.png` dan `model/dashboard_overview.png` hasil notebook).
4. Klik **Share** → **Publish to web / Get link**, salin link, lalu tempel di bagian ini:
   **Link Dashboard: `[isi link Looker Studio Anda di sini]`**

**Opsi B — Metabase (sesuai instruksi submission):**
1. Jalankan Metabase melalui Docker, lalu buat akun dengan email `root@mail.com` dan password `root123`.
2. Hubungkan Metabase ke database (import `dashboard_data.csv` atau gunakan database instance sesuai instruksi submission).
3. Setelah dashboard selesai dibuat, ekspor database instance Metabase dengan perintah:
   ```bash
   docker cp metabase.db.mv.db metabase-container:/metabase.db.mv.db
   ```
   Simpan file `metabase.db.mv.db` di folder proyek ini.
   **Link Dashboard (jika menggunakan Tableau Public / akses lain): `[isi di sini]`**

> **Catatan:** Screenshot referensi hasil eksplorasi data yang bisa dijadikan acuan visual dashboard tersedia di folder `model/` (`dashboard_overview.png`, `eda_status_distribution.png`, `eda_scholarship_tuition.png`, `eda_grade_distribution.png`, `eda_correlation.png`).

---

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dibangun menggunakan **Streamlit** dan model **Random Forest Classifier** yang telah dilatih pada `notebook.ipynb`.

**Menjalankan secara lokal:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`. Isi form data mahasiswa (demografi, finansial, akademik semester 1 & 2), lalu klik tombol **Prediksi** untuk melihat prediksi status mahasiswa (Dropout / Enrolled / Graduate) beserta probabilitasnya.

**Mengakses prototype secara online (Streamlit Community Cloud):**
1. Push seluruh isi folder proyek ini (termasuk folder `model/`, `app.py`, `requirements.txt`) ke repository GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io/), login dengan akun GitHub.
3. Klik **New app**, pilih repository dan branch, lalu set *Main file path* ke `app.py`.
4. Klik **Deploy**. Setelah proses selesai, aplikasi dapat diakses secara publik.
   **Link Prototype: `[isi link Streamlit Community Cloud Anda di sini]`**

---

## Conclusion

Berdasarkan proses analisis data dan pemodelan machine learning yang telah dilakukan terhadap data mahasiswa Jaya Jaya Institut, dapat disimpulkan bahwa:

1. Tingkat dropout mahasiswa Jaya Jaya Institut cukup tinggi, yaitu **32,1%** dari total 4.424 mahasiswa pada dataset.
2. Faktor yang paling berkaitan dengan dropout adalah **status pelunasan UKT**, **jumlah SKS yang berhasil diselesaikan (approved)** dan **nilai** pada semester 1 & 2, **kepemilikan beasiswa**, serta **usia saat mendaftar**.
3. Mahasiswa yang belum melunasi UKT (tingkat dropout ±86%) dan tidak memiliki beasiswa (tingkat dropout ±38%) memiliki risiko dropout jauh lebih tinggi dibanding kelompok lainnya (±25% dan ±12%).
4. Model **Random Forest Classifier** yang dibangun mampu memprediksi status mahasiswa dengan **akurasi ±76%** dan **F1-score macro ±0,70**, cukup andal digunakan sebagai alat bantu deteksi dini (*early warning system*) risiko dropout.
5. Model dan dashboard yang dihasilkan dapat dimanfaatkan pihak institusi untuk melakukan intervensi lebih awal kepada mahasiswa yang berisiko tinggi mengalami dropout.

### Rekomendasi Action Items
- **Bangun sistem peringatan dini (early warning system)** menggunakan model prediksi ini, dijalankan setiap akhir semester untuk mengidentifikasi mahasiswa berisiko dropout tinggi, agar tim akademik dapat segera melakukan intervensi (bimbingan konseling, mentoring akademik).
- **Evaluasi kembali program beasiswa dan skema pembayaran UKT** (misalnya opsi cicilan atau keringanan bagi mahasiswa kurang mampu), mengingat status pelunasan UKT dan beasiswa sangat berkorelasi dengan dropout.
- **Lakukan monitoring performa akademik secara berkala** (jumlah SKS disetujui dan nilai per semester) melalui dashboard, khususnya pada mahasiswa dengan penurunan performa drastis di semester berjalan.
- **Berikan perhatian khusus pada mahasiswa dengan usia pendaftaran lebih tua** dari rata-rata, karena kelompok ini menunjukkan risiko dropout lebih tinggi berdasarkan hasil *feature importance*.
- **Lakukan survei/wawancara terhadap mahasiswa yang diberi label risiko tinggi** oleh model, untuk memahami akar masalah non-akademik (finansial, keluarga, kesehatan mental) sebagai pelengkap data kuantitatif.
- **Integrasikan model prediksi ke dalam sistem informasi akademik** agar dosen wali/pembimbing akademik dapat mengakses skor risiko dropout mahasiswa bimbingannya secara real-time.
