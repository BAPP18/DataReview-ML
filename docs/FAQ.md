# FAQ & Troubleshooting

## Setup

**Q: `pip install -r requirements.txt` gagal di bagian `recordlinkage`?**
A: Butuh compiler C di Windows — install Microsoft C++ Build Tools, atau
pakai environment Anaconda (`conda install -c conda-forge recordlinkage`).
Notebook 01–10 dan Board tetap jalan untuk bagian lain tanpa package ini
kecuali sel FEBRL4.

**Q: Dataset Kaggle (Solar Power Generation) gagal diunduh?**
A: Buat API token di https://www.kaggle.com/settings → API → Create New Token,
simpan sebagai `~/.kaggle/kaggle.json`, lalu:
`kaggle datasets download anikannal/solar-power-generation-data`.
Semua dataset lain (NAB, DeepMatcher, HoloClean, UCI) bebas login —
lihat `notebooks/00_colab_setup.ipynb`.

**Q: Aplikasi Streamlit tidak terbuka di `localhost:8501`?**
A: Pastikan tidak ada proses lama di port itu (`netstat -ano | findstr 8501`),
lalu jalankan ulang `run_app.bat`. Jangan jalankan dua instance bersamaan.

## Data & Modeling

**Q: Mengapa F1 anomaly detection NAB rendah (mis. Isolation Forest 0,508)?**
A: Karena labelnya anomali dunia nyata yang sulit — derau, musiman rusak,
dan drift bertahap. Angka kecil yang jujur lebih baik dari angka besar
buatan. Rolling MAD bahkan nol di beberapa seri; itu temuan, bukan bug.

**Q: Bagaimana menginterpretasi `match_probability`?**
A: Skor Logistic Regression terkalibrasi di split validasi — **bukan vonis**.
Aturan routing tunggal ada di `config/models.yaml` (auto ≥ 0,90,
review ≥ 0,30). Reviewer selalu memutus berdasarkan tabel evidence,
bukan angka ini.

**Q: Mengapa recall model dirty Walmart-Amazon jeblok?**
A: Varian *dirty* sengaja dikosongkan/dipindah atributnya. Tepat di titik
inilah antrean review manual dibutuhkan — bukan kegagalan, melainkan
desain workflow-nya.

**Q: Dari mana reason code berasal?**
A: Daftar tetap di `config/models.yaml → review.reason_codes`. Kode terstruktur
inilah yang memungkinkan analisis akar masalah (dashboard → Decisions by
reason code). Free text saja tidak bisa diagregasi.

## Glossary singkat

| Istilah | Arti di proyek ini |
|---|---|
| Entity resolution | Menentukan apakah dua record beda sumber = satu entitas |
| Blocking | Mengelompokkan kandidat agar tidak bandingkan semua-vs-semua |
| Pairwise features | Skor kemiripan per pasangan (Jaro-Winkler, TF-IDF cosine, …) |
| Anomaly score | Seberapa menyimpang satu observasi dari perilaku historis |
| Reason code | Kode terstruktur penyebab keputusan (bukan teks bebas) |
| QA sampling | 10% kasus resolved diverifikasi ulang reviewer kedua |
| Source of truth | Sumber paling otoritatif saat konflik (signed doc > CRM > ERP > partner) |
| SLA breach | Kasus terbuka melewati 72 jam |
| Handoff file | CSV koreksi approved — padanan file-based dari "sync ke CRM" |
