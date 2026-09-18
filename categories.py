"""
Mapping kode kategorikal -> label yang mudah dibaca.
Sumber: UCI ML Repository - Predict Students' Dropout and Academic Success
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
"""

MARITAL_STATUS = {
    1: "Lajang (Single)",
    2: "Menikah (Married)",
    3: "Duda/Janda (Widower)",
    4: "Bercerai (Divorced)",
    5: "Union Facto",
    6: "Berpisah Secara Hukum (Legally Separated)",
}

APPLICATION_MODE = {
    1: "Fase 1 - Kontingen Umum",
    2: "Ordinance No. 612/93",
    5: "Fase 1 - Kontingen Khusus (Pulau Azores)",
    7: "Pemegang Kualifikasi Pendidikan Tinggi Lain",
    10: "Ordinance No. 854-B/99",
    15: "Mahasiswa Internasional (S1)",
    16: "Fase 1 - Kontingen Khusus (Pulau Madeira)",
    17: "Fase 2 - Kontingen Umum",
    18: "Fase 3 - Kontingen Umum",
    26: "Ordinance No. 533-A/99, item b2 (Rencana Berbeda)",
    27: "Ordinance No. 533-A/99, item b3 (Institusi Lain)",
    39: "Usia di atas 23 Tahun",
    42: "Pindah (Transfer)",
    43: "Pindah Jurusan",
    44: "Pemegang Diploma Spesialisasi Teknologi",
    51: "Pindah Institusi/Jurusan",
    53: "Pemegang Diploma Siklus Singkat",
    57: "Pindah Institusi/Jurusan (Internasional)",
}

COURSE = {
    33: "Teknologi Produksi Biofuel",
    171: "Animasi dan Desain Multimedia",
    8014: "Pelayanan Sosial (Kelas Malam)",
    9003: "Agronomi",
    9070: "Desain Komunikasi",
    9085: "Keperawatan Hewan",
    9119: "Teknik Informatika",
    9130: "Equinculture (Peternakan Kuda)",
    9147: "Manajemen",
    9238: "Pelayanan Sosial",
    9254: "Pariwisata",
    9500: "Keperawatan (Nursing)",
    9556: "Kebersihan Mulut (Oral Hygiene)",
    9670: "Manajemen Periklanan dan Pemasaran",
    9773: "Jurnalistik dan Komunikasi",
    9853: "Pendidikan Dasar",
    9991: "Manajemen (Kelas Malam)",
}

ATTENDANCE = {1: "Siang (Daytime)", 0: "Malam (Evening)"}

QUALIFICATION = {
    1: "Pendidikan Menengah (SMA)",
    2: "Pendidikan Tinggi - Sarjana Muda (Bachelor)",
    3: "Pendidikan Tinggi - Sarjana (Degree)",
    4: "Pendidikan Tinggi - Magister",
    5: "Pendidikan Tinggi - Doktor",
    6: "Sedang Menempuh Pendidikan Tinggi",
    9: "Kelas 12 - Tidak Selesai",
    10: "Kelas 11 - Tidak Selesai",
    11: "Kelas 7 (Lama)",
    12: "Lainnya - Kelas 11",
    13: "Kelas 2 Sekolah Menengah Lanjutan",
    14: "Kelas 10",
    15: "Kelas 10 - Tidak Selesai",
    18: "Kursus Perdagangan Umum",
    19: "Pendidikan Dasar Siklus 3 (Kelas 9/10/11)",
    20: "Kursus Sekolah Menengah Lanjutan",
    22: "Kursus Teknik-Profesional",
    25: "Sekolah Menengah Lanjutan - Tidak Selesai",
    26: "Kelas 7",
    27: "Siklus 2 Sekolah Menengah Umum",
    29: "Kelas 9 - Tidak Selesai",
    30: "Kelas 8",
    31: "Kursus Umum Administrasi dan Perdagangan",
    33: "Akuntansi dan Administrasi Tambahan",
    34: "Tidak Diketahui",
    35: "Tidak Bisa Membaca/Menulis",
    36: "Bisa Membaca tanpa Kelas 4",
    37: "Pendidikan Dasar Siklus 1 (Kelas 4/5)",
    38: "Pendidikan Dasar Siklus 2 (Kelas 6/7/8)",
    39: "Kursus Spesialisasi Teknologi",
    40: "Pendidikan Tinggi - Gelar (Siklus 1)",
    41: "Kursus Studi Tinggi Khusus",
    42: "Kursus Teknik Tinggi Profesional",
    43: "Pendidikan Tinggi - Magister (Siklus 2)",
    44: "Pendidikan Tinggi - Doktor (Siklus 3)",
}

NACIONALITY = {
    1: "Portugis", 2: "Jerman", 6: "Spanyol", 11: "Italia", 13: "Belanda",
    14: "Inggris", 17: "Lituania", 21: "Angola", 22: "Tanjung Verde",
    24: "Guinea", 25: "Mozambik", 26: "Sao Tome", 32: "Turki", 41: "Brazil",
    62: "Rumania", 100: "Moldova", 101: "Meksiko", 103: "Ukraina",
    105: "Rusia", 108: "Kuba", 109: "Kolombia",
}

OCCUPATION = {
    0: "Pelajar",
    1: "Anggota Legislatif/Eksekutif, Direktur, Manajer Eksekutif",
    2: "Spesialis Kegiatan Intelektual dan Ilmiah",
    3: "Teknisi dan Profesi Tingkat Menengah",
    4: "Staf Administrasi",
    5: "Pekerja Layanan Personal, Keamanan, dan Penjual",
    6: "Petani dan Pekerja Terampil Pertanian/Perikanan",
    7: "Pekerja Terampil Industri, Konstruksi, Pengrajin",
    8: "Operator Instalasi dan Mesin",
    9: "Pekerja Tidak Terampil",
    10: "Profesi Angkatan Bersenjata",
    90: "Situasi Lainnya",
    99: "Kosong/Tidak Diketahui",
    122: "Profesional Kesehatan",
    123: "Guru/Pengajar",
    125: "Spesialis TIK",
    131: "Teknisi Sains dan Teknik Tingkat Menengah",
}

YES_NO = {1: "Ya", 0: "Tidak"}
GENDER = {1: "Laki-laki", 0: "Perempuan"}

FEATURE_LABELS_ID = {
    'Marital_status': 'Status Pernikahan',
    'Application_mode': 'Mode Pendaftaran',
    'Application_order': 'Urutan Pilihan Pendaftaran',
    'Course': 'Program Studi',
    'Daytime_evening_attendance': 'Waktu Kelas',
    'Previous_qualification': 'Kualifikasi Sebelumnya',
    'Previous_qualification_grade': 'Nilai Kualifikasi Sebelumnya',
    'Nacionality': 'Kewarganegaraan',
    'Mothers_qualification': 'Kualifikasi Ibu',
    'Fathers_qualification': 'Kualifikasi Ayah',
    'Mothers_occupation': 'Pekerjaan Ibu',
    'Fathers_occupation': 'Pekerjaan Ayah',
    'Admission_grade': 'Nilai Ujian Masuk',
    'Displaced': 'Displaced (Pindah Domisili)',
    'Educational_special_needs': 'Kebutuhan Pendidikan Khusus',
    'Debtor': 'Memiliki Tunggakan',
    'Tuition_fees_up_to_date': 'UKT Lunas Tepat Waktu',
    'Gender': 'Jenis Kelamin',
    'Scholarship_holder': 'Penerima Beasiswa',
    'Age_at_enrollment': 'Usia saat Mendaftar',
    'International': 'Mahasiswa Internasional',
}
