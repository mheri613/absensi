# YOLO Absensi

python versi 3.9.13

## Cara Instalasi dan Menjalankan Aplikasi

### Persyaratan Instalasi
```
pip install -r requirements.txt
```

### Menjalankan Aplikasi
```
streamlit run main.py
```

## Akun Pengguna

### Akun Admin
- **Email**: admin@example.com
- **Password**: admin

## Panduan Penggunaan

1. Login sebagai admin
2. Tambahkan gambar untuk siswa
3. Lakukan proses absensi


cara pakai api.py
python api.py

cara pakai mobile flutter
ganti address api dengan network wifi sendiri, connect hp ke wifi yang sama dengan laptop
jalankan di hp




cara testing confusion matrix
1. masukkan gambar wajah yang sudah terdaftar di folder test-absensi/test_registered
2. masukkan gambar wajah yang tidak terdaftar di folder test-absensi/test_unregistered
3. jalankan api.py dengan cara ketik ini di terminal: uvicorn api:app --reload
4. jalankan test.py yang ada di dalam folder test-absensi dengan cara ketik ini di terminal: python .\test-absensi\test.py
5. hasil testing akan muncul di folder test-absensi# absensi
