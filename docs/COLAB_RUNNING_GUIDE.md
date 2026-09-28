# Panduan Running di Google Colab — 3 Level

Proyek ini dikembangkan lokal (Windows, path `D:/data viewer`), tapi semua
notebook bisa jalan di Google Colab. Pilih salah satu cara di bawah.

---

## Cara MUDAH — Upload manual (cocok untuk coba-coba cepat)

**Waktu: ±10 menit. Skill: bisa klik-klik Colab.**

1. Buka https://colab.research.google.com → **File → Upload notebook** →
   pilih `notebooks/00_colab_setup.ipynb` dari komputer.
2. Jalankan semua cell berurutan (Runtime → Run all):
   - Cell install akan memasang `recordlinkage`, `scikit-learn`, dll.
   - Cell dataset NAB/DeepMatcher/HoloClean/UCI/FEBRL4 jalan otomatis.
   - Cell Solar Power Generation meminta upload `kaggle.json`
     (buat di https://www.kaggle.com/settings → API → Create New Token).
3. Upload notebook 01–10 satu per satu (File → Upload notebook),
   lalu di tiap notebook lakukan **satu langkah wajib**:
   Edit → Find and replace → ganti `D:/data viewer` menjadi
   `/content/solar-review` → Replace all.
4. Runtime → Run all di tiap notebook, urut 01 → 10.

**Keterbatasan:** file di `/content` hilang saat runtime restart —
ulangi langkah 2 jika sesi Colab terputus.

---

## Cara MEDIUM — Google Drive (cocok untuk kerja berhari-hari)

**Waktu setup: ±15 menit sekali. Skill: tahu mount Drive.**

1. Di Drive buat folder `solar-review/`, upload ke dalamnya:
   - seluruh folder `notebooks/`
   - folder `data/raw/` (atau cukup file-file besar: solar ZIP, donation ZIP)
2. Di Colab, cell pertama setiap sesi:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   BASE = "/content/drive/MyDrive/solar-review"
   ```
3. Jalankan `00_colab_setup.ipynb` dari Drive — ubah variabel `BASE`
   di cell pertama menjadi path Drive di atas, lalu Run all.
   (Cell download otomatis SKIP file yang sudah ada.)
4. Untuk notebook 01–10: ganti `D:/data viewer` menjadi
   `/content/drive/MyDrive/solar-review` (sekali per notebook,
   lalu File → Save a copy in Drive agar permanen).
5. Run all urut 01 → 10. Output (`data/processed`, `reports/`)
   tersimpan permanen di Drive.

**Kelebihan:** tidak perlu re-download tiap sesi; hasil processed/reports awet.

---

## Cara SUSAH — GitHub + pipeline penuh (cocok untuk portfolio/showcase)

**Waktu setup: ±30 menit. Skill: git, terminal, paham arsitektur proyek.**

1. Push repo ini ke GitHub (tanpa `data/raw/` — sudah di-`.gitignore`
   kecuali `.gitkeep`, sesuai constraint lisensi 1.4).
2. Di Colab:
   ```python
   !git clone https://github.com/<username>/<repo>.git /content/solar-review
   %cd /content/solar-review
   !pip -q install -r requirements.txt
   ```
3. Jalankan `notebooks/00_colab_setup.ipynb` dengan `BASE` bawaan
   (`/content/solar-review`), lalu ubah path di notebook 01–10
   (`D:/data viewer` → `/content/solar-review`) dan Run all 01 → 10.
4. Jalankan test suite:
   ```python
   !python -m unittest discover -s tests -v
   ```
   (43 tests — termasuk 9 tests dataset berantakan baru.)
5. Demo Streamlit di Colab via tunnel (satu cell, biarkan jalan):
   ```python
   !pip -q install streamlit cloudflared
   !streamlit run app/dashboard.py --server.port 8501 --server.headless true & npx -y cloudflared tunnel --url http://localhost:8501
   ```
   Buka URL `*.trycloudflare.com` yang muncul untuk akses dashboard.
6. Untuk hasil permanen, mount Drive (lihat Cara Medium) dan salin
   `data/processed` + `reports/` ke Drive di akhir sesi.

**Kelebihan:** reproducible end-to-end dari repo bersih — persis seperti
yang dinilai reviewer portfolio (clone → setup → run → demo).

---

## Catatan umum (semua cara)

- Urutan wajib: `00` → `01` → … → `10` (tiap notebook memakai output sebelumnya).
- Epoch 2026 tidak dipakai (competition ended, API 403) — diganti
  NAB + DeepMatcher dirty + Hospital/Flights + UCI Donation (lihat
  `reports/dataset_suitability.md`).
- File `.py` di `src/` dan `app/dashboard.py` tidak perlu diubah untuk Colab;
  cukup notebook-nya saja. Streamlit hanya jalan di Cara Susah (tunnel)
  atau lokal (`streamlit run app/dashboard.py`).
