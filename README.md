# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut merupakan sebuah institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Hingga saat ini, Jaya Jaya Institut telah mencetak banyak lulusan dengan reputasi yang sangat baik. Namun, terdapat cukup banyak juga siswa yang tidak menyelesaikan pendidikannya alias mengalami *dropout*.

Tingginya angka dropout ini menjadi salah satu masalah besar bagi sebuah institusi pendidikan, karena berdampak pada reputasi, akreditasi, dan keberlanjutan finansial institusi. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin mahasiswa yang berpotensi melakukan dropout, sehingga dapat diberikan bimbingan khusus sejak dini.

### Permasalahan Bisnis
1. Tingkat dropout mahasiswa cukup tinggi, yaitu **39,1%** dari mahasiswa yang telah memiliki status akhir (1.421 dari 3.630 mahasiswa berstatus Dropout atau Graduate).
2. Institusi belum memiliki sistem yang dapat mendeteksi secara dini mahasiswa yang berisiko dropout, sehingga intervensi (bimbingan, konseling, keringanan biaya, dll) sering terlambat diberikan.
3. Belum tersedia media monitoring (dashboard) yang memudahkan manajemen memantau performa akademik mahasiswa dan faktor-faktor yang berkontribusi terhadap dropout.

### Cakupan Proyek
1. **Data Understanding & EDA** — mengeksplorasi data untuk menemukan pola dan faktor yang berkaitan dengan dropout.
2. **Data Preparation** — memfilter data (hanya status **Dropout** dan **Graduate** yang digunakan untuk pemodelan; status **Enrolled** dipisahkan sebagai data prediksi masa depan), encoding target biner, scaling, dan split data latih/uji.
3. **Modeling** — membangun model klasifikasi biner (Random Forest) untuk memprediksi apakah mahasiswa akan **Dropout** (1) atau **Graduate** (0).
4. **Evaluation** — mengevaluasi performa model dengan accuracy, F1-score, ROC-AUC, confusion matrix, serta menganalisis fitur & ambang batas (threshold) paling berpengaruh terhadap dropout.
5. **Business Dashboard** — membuat dashboard visual interaktif untuk membantu tim manajemen memonitor performa dan faktor risiko dropout siswa.
6. **Prototype Machine Learning** — membangun aplikasi Streamlit yang mendukung prediksi individu maupun massal (upload file), lengkap dengan panduan penggunaan.
7. **Conclusion & Action Items** — memberikan rekomendasi tindakan yang konkret dan berbasis data (dengan ambang batas spesifik) bagi Jaya Jaya Institut.

### Persiapan

**Sumber data:** [Students' Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance) (`data.csv`) — 4.424 baris data mahasiswa dengan atribut demografis, sosial-ekonomi, dan akademik, serta kolom target `Status` (Dropout/Enrolled/Graduate).

> **Catatan metodologi penting:** Dari 4.424 mahasiswa, hanya **3.630 mahasiswa berstatus final (Dropout/Graduate)** yang digunakan untuk melatih dan mengevaluasi model. **794 mahasiswa berstatus Enrolled** (masih aktif kuliah, belum memiliki label akhir) **tidak diikutsertakan dalam training** — karena jika dilibatkan akan membuat target menjadi ambigu. Data Enrolled ini justru disimpan terpisah (`model/enrolled_students_for_prediction.csv`) untuk didemokan sebagai data yang diprediksi oleh model & prototype (use case nyata dari sistem ini).

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
Seluruh tahapan proses data science — mulai dari *business understanding* hingga *conclusion* — didokumentasikan lengkap di dalam `notebook.ipynb`, termasuk proses penyimpanan model ke folder `model/` (`model.joblib`, `scaler.joblib`, `feature_names.joblib`) yang digunakan oleh prototype Streamlit.

---

## Business Dashboard

Dashboard bisnis dibuat **khusus untuk keperluan monitoring** (bukan untuk melakukan prediksi) — terpisah dari prototype machine learning. Dashboard ini dibangun di **Looker Studio** dan membantu Jaya Jaya Institut memantau performa dan faktor risiko dropout mahasiswa secara visual, mencakup:
- KPI ringkasan (tingkat dropout keseluruhan, jumlah mahasiswa per status)
- Distribusi status mahasiswa & tingkat dropout per program studi
- Tingkat dropout berdasarkan faktor sosial-ekonomi (UKT, beasiswa, tunggakan)
- Tingkat dropout berdasarkan performa akademik (SKS lulus, nilai rata-rata semester)
- Tingkat dropout berdasarkan demografi (usia, jenis kelamin)

**🔗 Link Dashboard (Looker Studio, dapat diakses & divalidasi langsung oleh reviewer):**
**https://datastudio.google.com/reporting/06d899db-0289-480f-be82-451eab95e6b3**


**Sumber data dashboard:** `looker_dashboard_data.csv` — berisi data mahasiswa yang sudah dilengkapi label kategorikal (Scholarship_Label, Tuition_Label, Debtor_Label, Gender_Label) dan pengelompokan (bin) untuk SKS lulus semester 2, nilai semester 2, serta kelompok usia, sehingga siap langsung dipakai sebagai dimension/metric di Looker Studio tanpa perlu formula tambahan.

**Alternatif/cadangan:**
- Tersedia sebagai dashboard versi Streamlit (dapat diakses melalui `https://projectdicoding.streamlit.app/Dashboard`) apabila dibutuhkan sebagai pembanding.

---

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning (**terpisah dari dashboard di atas**) dibangun menggunakan **Streamlit** dan model **Random Forest Classifier (biner: Dropout vs Graduate)**, dengan 3 fitur utama:

1. **🧍 Prediksi Individu** — form input lengkap dengan seluruh kolom kategorikal ditampilkan sebagai **label yang mudah dibaca** (misalnya "Lajang", "Menikah" — bukan sekadar angka 1, 2), dilengkapi kategori risiko (Rendah/Sedang/Tinggi/Sangat Tinggi).
2. **📂 Prediksi Massal (Upload File)** — untuk memprediksi banyak mahasiswa sekaligus melalui upload CSV/Excel, lengkap dengan template kosong dan contoh data mahasiswa `Enrolled` yang bisa langsung dicoba, serta ringkasan jumlah mahasiswa per kategori risiko dan unduhan hasil prediksi.
3. **📖 Panduan Penggunaan** — petunjuk langkah demi langkah cara memakai prototype (baik input manual maupun upload file), tabel ambang batas kategori risiko, serta tabel referensi kode kategorikal lengkap untuk keperluan upload file.

### File pendukung yang dibutuhkan (sudah disertakan dalam submission ini)
Seluruh file berikut wajib ada agar prototype dapat berjalan, baik secara lokal maupun saat deployment, dan **semuanya sudah tersedia di dalam folder submission ini** — reviewer tidak perlu mencari atau membuat file tambahan:

| File | Fungsi |
|---|---|
| `app.py` | File utama aplikasi Streamlit (halaman prediksi) |
| `categories.py` | Mapping kode kategorikal → label yang mudah dibaca (dipakai oleh `app.py`) |
| `model/model.joblib` | Model Random Forest Classifier yang sudah dilatih |
| `model/scaler.joblib` | StandardScaler yang sudah di-fit pada data training |
| `model/feature_names.joblib` | Daftar & urutan nama fitur yang menjadi input model |
| `model/enrolled_students_for_prediction.csv` | Contoh dataset (794 mahasiswa `Enrolled`) untuk uji coba fitur Prediksi Massal |
| `requirements.txt` | Daftar dependency Python beserta versinya |


### Mengakses Prototype secara Online (Streamlit Community Cloud)

Prototype juga sudah di-deploy dan dapat diakses langsung tanpa instalasi apa pun melalui link berikut:

**🔗 `https://projectdicoding.streamlit.app/`**

---

## Conclusion

Berdasarkan proses analisis data dan pemodelan machine learning yang telah dilakukan terhadap data mahasiswa Jaya Jaya Institut — dengan model biner **Dropout vs Graduate** (mengeluarkan kelas Enrolled dari proses training) — dapat disimpulkan bahwa:

1. Dari 3.630 mahasiswa yang telah memiliki status akhir, **39,1% (1.421 mahasiswa) berstatus Dropout**.
2. Faktor akademik di semester-semester awal — khususnya **jumlah SKS yang berhasil diselesaikan (approved)** dan **nilai rata-rata** pada semester 1 & 2 — merupakan prediktor terkuat terhadap risiko dropout, jauh melampaui faktor demografis.
3. Ditemukan pola ambang batas (threshold) yang jelas dan dapat langsung ditindaklanjuti:
   - Mahasiswa dengan **1-2 SKS lulus di semester 2** memiliki probabilitas dropout **97,6%**; yang **tidak lulus SKS sama sekali (0)** sebesar **90,6%**; sedangkan yang lulus **7+ SKS** turun menjadi **9,1%**.
   - Mahasiswa dengan **nilai rata-rata semester 2 ≤ 10** memiliki probabilitas dropout **94,1%**, turun ke **44,4%** pada rentang nilai 11-12, **17,0%** pada rentang 13-14, dan **9,7%** pada nilai ≥15.
   - Mahasiswa yang **belum melunasi UKT** memiliki probabilitas dropout **94,0%** vs **30,7%** yang sudah lunas; mahasiswa **berstatus debitur (menunggak)** memiliki probabilitas dropout **75,5%** vs **34,5%** yang tidak menunggak.
   - Mahasiswa **tanpa beasiswa**: probabilitas dropout **48,4%**, dibanding **13,8%** pada penerima beasiswa.
   - Mahasiswa yang mendaftar pada **usia 24-30 tahun** memiliki probabilitas dropout tertinggi (**66,5%**), dibanding kelompok usia 15-20 tahun (**26,1%**).
   - Program studi **Informatics Engineering** menunjukkan tingkat dropout tertinggi (**86,8%** dari 106 mahasiswa), jauh di atas rata-rata keseluruhan.
4. Model **Random Forest Classifier (biner)** yang dibangun mampu memprediksi status Dropout/Graduate dengan **akurasi 92,4%**, **F1-score 0,90**, dan **ROC-AUC 0,97** — sangat andal digunakan sebagai *early warning system*.
5. Mahasiswa berstatus **Enrolled** (794 orang) sudah disiapkan dalam file terpisah (`model/enrolled_students_for_prediction.csv`) agar dapat langsung diprediksi menggunakan model dan prototype yang telah dibangun (fitur *Prediksi Massal*), tanpa mencampurnya ke dalam proses training.

### Rekomendasi Action Items

Setiap rekomendasi berikut dikaitkan langsung dengan insight dari analisis data (poin 3 di atas), disertai alasan berbasis data, serta gambaran implementasi konkret:

1. **Bangun sistem peringatan dini berbasis ambang SKS lulus (≤2 SKS).** *Insight:* mahasiswa dengan 1-2 SKS lulus di akhir semester memiliki probabilitas dropout 97,6%. *Implementasi:* jalankan tab **Prediksi Massal** pada prototype terhadap seluruh mahasiswa aktif setiap akhir semester; mahasiswa dengan probabilitas ≥75% (kategori **Sangat Tinggi**, lihat tabel ambang batas di Panduan Penggunaan prototype) wajib dijadwalkan sesi konseling akademik dalam 2 minggu pertama semester berikutnya.
2. **Prioritaskan intervensi finansial pada kombinasi UKT belum lunas + status debitur.** *Insight:* probabilitas dropout pada mahasiswa yang UKT belum lunas mencapai 94,0%, dan pada mahasiswa berstatus debitur 75,5%. *Implementasi:* Bagian Keuangan menyusun skema cicilan UKT atau dana bantuan darurat khusus untuk mahasiswa dalam kategori ini, ditawarkan **proaktif** sebelum masa pelunasan berakhir (bukan menunggu mahasiswa mengajukan), diprioritaskan bagi mahasiswa yang juga masuk kategori risiko akademik Tinggi/Sangat Tinggi.
3. **Perluas kuota beasiswa berbasis risiko, bukan hanya prestasi.** *Insight:* mahasiswa tanpa beasiswa memiliki probabilitas dropout 48,4% vs 13,8% pada penerima beasiswa — selisih 34,6 poin persentase. *Implementasi:* alokasikan 10-15% kuota beasiswa baru khusus untuk mahasiswa yang oleh model diprediksi risiko **Sedang-Tinggi** (probabilitas 25-74%) namun memiliki nilai akademik semester 1 di atas rata-rata (≥12) — kelompok ini berpotensi paling besar "diselamatkan" dengan sedikit dukungan finansial.
4. **Pantau nilai rata-rata semester dengan ambang 10, 12, dan 14 pada dashboard akademik.** *Insight:* nilai ≤10 → risiko dropout 94,1%; nilai 11-12 → 44,4%; nilai 13-14 → 17,0%; nilai ≥15 → 9,7%. *Implementasi:* dashboard akademik (lihat bagian Business Dashboard) menampilkan indikator warna merah (≤10), kuning (11-12), hijau muda (13-14), hijau (≥15) per mahasiswa per semester, agar dosen wali dapat memprioritaskan mahasiswa "merah" terlebih dahulu setiap awal semester baru.
5. **Berikan pendampingan khusus untuk mahasiswa usia 24-30 tahun saat mendaftar.** *Insight:* kelompok usia ini memiliki probabilitas dropout 66,5%, tertinggi di antara seluruh kelompok usia (vs 26,1% pada usia 15-20 tahun), kemungkinan karena tekanan pekerjaan/keluarga. *Implementasi:* tawarkan kelas malam/hybrid dan program pendampingan manajemen waktu khusus bagi kelompok usia ini sejak orientasi mahasiswa baru; targetkan minimal 1 sesi pendampingan di bulan pertama perkuliahan.
6. **Evaluasi kurikulum & dukungan akademik program studi dengan dropout tertinggi**, terutama **Informatics Engineering** (86,8% dari 106 mahasiswa) dan **Equinculture** (65,0% dari 120 mahasiswa). *Implementasi:* lakukan audit beban SKS dan tingkat kesulitan mata kuliah semester 1-2 pada kedua program ini, serta pertimbangkan penambahan sesi tutorial/asistensi khusus.
7. **Integrasikan model prediksi ke dalam sistem informasi akademik**, menjalankan model terhadap seluruh data mahasiswa `Enrolled` setiap akhir semester menggunakan fitur **Prediksi Massal** pada prototype, sehingga dosen pembimbing akademik menerima daftar mahasiswa bimbingannya lengkap dengan skor probabilitas dropout dan kategori risiko (Rendah/Sedang/Tinggi/Sangat Tinggi) untuk memprioritaskan tindak lanjut.
