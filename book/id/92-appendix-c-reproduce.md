# Lampiran C — Mereproduksi setiap hasil

Setiap angka, grafik, dan tangkapan layar dalam buku ini dihasilkan oleh kode di
repositori ini. Lampiran ini mencantumkan perintah persisnya, secara berurutan.
Semua path relatif terhadap root repositori.

## 1. Perangkat

| Perangkat | Kegunaan |
|---|---|
| PostgreSQL di `localhost:5432` (user `postgres`, password `postgres`) | basis data marketplace (bab 13, tangkapan layar, latensi) |
| Rust toolchain (`cargo`) dan `protoc` | server API di `backend/` |
| Node.js dan `npm` | aplikasi web di `frontend/`, serta mermaid/KaTeX untuk membangun PDF |
| `uv` | semua lingkungan Python: `recommender/` dan `book/` |

## 2. Skrip pendamping offline (bab 1–12)

Skrip-skrip ini hanya memerlukan Python. Semuanya memakai marketplace simulasi
yang sama (seed 7), yang di-*cache* di `book/build/sim.pkl` pada pemakaian pertama.

```sh
cd book
uv sync                                  # installs recsys from ../recommender in editable mode
uv run python tools/run_all.py --skip-live
```

`run_all.py` menjalankan `code/ch01_explore.py` … `code/ch12_coldstart_bias.py`
secara berurutan. Setiap skrip mencetak teks yang dikutip di babnya, yang disimpan
ke `results/<script>.txt`, dan menulis angkanya ke `results/<script>.json`.
Eksekusi berhenti pada skrip pertama yang gagal. Untuk menjalankan ulang skrip satu bab saja:

```sh
uv run python tools/run_all.py --only ch06_mf
```

Skrip-skrip ini deterministik: mesin yang sama menghasilkan berkas yang identik.
Antar mesin, waktu (milidetik per kueri di bab 7) berbeda, dan LightGBM serta
PyTorch bisa berbeda pada angka desimal terakhir.

## 3. Sistem langsung (bab 13)

Penyiapan satu kali:

```sh
PGPASSWORD=postgres createdb -h localhost -U postgres marketplace
cp backend/.env.example backend/.env
(cd backend && cargo run -- migrate)
(cd recommender && uv sync && uv run recsys seed-db --yes)
(cd frontend && npm i)
```

Lalu jalankan ketiga layanan, masing-masing di terminalnya sendiri:

```sh
cd recommender && uv run recsys serve
cd backend && cargo run
cd frontend && npm run dev
```

Dengan recommender berjalan, skrip latensi memanggilnya langsung lewat gRPC:

```sh
cd book && uv run python tools/run_all.py --only ch13_latency
```

Karena `seed-db` menggeser riwayat simulasi agar berakhir tepat pada saat Anda
menjalankannya, metrik dan latensi model langsung akan sedikit berbeda dari yang
dicetak dalam buku ini.

## 4. Grafik

```sh
cd book && uv run python tools/make_figures.py
```

Perintah ini membaca `results/*.json` dan menulis setiap PNG di `assets/charts/`.
Grafik tiap bab digambar oleh `tools/figures/chNN.py`.

## 5. Tangkapan layar

Dengan ketiga layanan berjalan dan aplikasi di `http://localhost:5173`:

```sh
cd book && uv run python tools/screenshots.py
```

Skrip ini login lewat UI sungguhan sebagai `user43@demo.test`, membuat pesanan,
lalu login sebagai penjual toko tersebut dan sebagai `admin@demo.test`, dan
menulis PNG ke `assets/screenshots/`. Banner *fallback* mengharuskan recommender
dalam keadaan **dihentikan**:

```sh
# stop `recsys serve`, then:
cd book && uv run python tools/screenshots.py --only fallback
```

## 6. PDF

```sh
cd book
uv run playwright install chromium
npm i                                    # mermaid + KaTeX, rendered inside the page
uv run python tools/check.py --lang en
uv run python tools/check.py --lang id
uv run python tools/build.py --lang all
```

`check.py` memverifikasi bahwa setiap *include*, berkas keluaran, dan gambar ada,
serta bahwa setiap bab memiliki kotak fakta, diagram, gambar, kuis, dan panjang
minimum. `build.py` merender Markdown menjadi HTML, membukanya di Chromium
*headless*, menunggu diagram dan rumus selesai dirender, lalu mencetak
`dist/recsys-book-en.pdf` dan `dist/recsys-book-id.pdf` beserta jumlah halamannya.
