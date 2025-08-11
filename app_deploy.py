import streamlit as st
import os
import faiss
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
import time

# --- Konfigurasi Halaman & API Key ---
st.set_page_config(page_title="Sistem Pakar Imigrasi", layout="wide")
st.title("🇮🇩 Sistem Pakar Keimigrasian Indonesia")

# Menggunakan st.secrets untuk deploy, atau sidebar untuk input lokal
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (FileNotFoundError, KeyError):
    st.sidebar.warning("Google API Key tidak ditemukan. Harap masukkan di bawah.")
    GOOGLE_API_KEY = st.sidebar.text_input("Masukkan Google API Key Anda:", type="password")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.info("Harap masukkan Google API Key di sidebar untuk memulai aplikasi.")
        st.stop()

FOLDER_DOKUMEN = 'dokumen_hukum'

# --- Database Q&A ---
DATABASE_QA = [
    {"pertanyaan": "Berapa denda overstay?", "jawaban": "Denda overstay berdasarkan Pasal 78 UU No. 6 Tahun 2011 adalah Rp1.000.000 per hari untuk keterlambatan di bawah 60 hari."},
    {"pertanyaan": "Siapa saja yang bisa dapat izin tinggal tetap?", "jawaban": "Menurut Pasal 54 UU No. 6/2011, ITAP dapat diberikan kepada Orang Asing pemegang ITAS, keluarga karena perkawinan campuran, dan investor."},
    {"pertanyaan": "Untuk apa Visa Kunjungan digunakan?", "jawaban": "Visa Kunjungan diberikan kepada Orang Asing untuk tujuan seperti pariwisata, keluarga, sosial, seni dan budaya, serta tugas pemerintahan non-komersial."},
]

# --- Contoh Statis untuk Prompt ---
FEW_SHOT_EXAMPLES = "--- CONTOH CARA MENJAWAB ---\nPertanyaan: Apa itu penjamin?\nJawaban: Penjamin adalah orang atau korporasi yang bertanggung jawab atas keberadaan dan kegiatan Orang Asing selama berada di Wilayah Indonesia.\n--- AKHIR CONTOH ---"

# --- FUNGSI-FUNGSI DENGAN CACHING UNTUK DEPLOYMENT ---

@st.cache_resource
def muat_dan_bangun_index():
    """Fungsi ini hanya akan dijalankan sekali untuk memuat semua data dan index."""
    st.info("Memulai proses muat data dan pembangunan index (hanya sekali)...")
    
    # 1. Proses Dokumen PDF
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    semua_potongan = []
    for filename in os.listdir(FOLDER_DOKUMEN):
        if filename.endswith('.pdf'):
            try:
                file_path = os.path.join(FOLDER_DOKUMEN, filename)
                reader = PdfReader(file_path)
                teks_lengkap = "".join(page.extract_text() or "" for page in reader.pages)
                if teks_lengkap:
                    potongan_teks = text_splitter.split_text(teks_lengkap)
                    for pot in potongan_teks:
                        semua_potongan.append({"sumber": filename, "konten": pot})
            except Exception:
                pass
    
    if not semua_potongan:
        st.error("Tidak ada dokumen yang bisa diproses.")
        return None, None, None, None

    konten_dokumen = [doc['konten'] for doc in semua_potongan]
    embeddings_dokumen = genai.embed_content(model="models/text-embedding-004", content=konten_dokumen, task_type="RETRIEVAL_DOCUMENT")["embedding"]
    index_dokumen = faiss.IndexFlatL2(np.array(embeddings_dokumen).shape[1])
    index_dokumen.add(np.array(embeddings_dokumen, dtype='float32'))
    st.write(f"✅ Index Dokumen berhasil dibuat dengan {len(semua_potongan)} potongan.")

    # 2. Proses Database Q&A
    questions_qa = [item['pertanyaan'] for item in DATABASE_QA]
    embeddings_qa = genai.embed_content(model="models/text-embedding-004", content=questions_qa, task_type="RETRIEVAL_DOCUMENT")["embedding"]
    index_qa = faiss.IndexFlatL2(np.array(embeddings_qa).shape[1])
    index_qa.add(np.array(embeddings_qa, dtype='float32'))
    st.write("✅ Index Q&A berhasil dibuat.")
    
    st.success("Semua data berhasil dimuat dan index siap digunakan!")
    return index_dokumen, semua_potongan, index_qa, DATABASE_QA

def cari_info(pertanyaan, index, bank_data, tipe, top_k=2):
    embedding_pertanyaan = np.array([genai.embed_content(model="models/text-embedding-004", content=pertanyaan, task_type="RETRIEVAL_QUERY")["embedding"]], dtype='float32')
    _, indices = index.search(embedding_pertanyaan, top_k)
    hasil = [bank_data[i] for i in indices[0]]
    if tipe == "dokumen":
        return "\n---\n".join([f"Kutipan dari {doc['sumber']}:\n{doc['konten']}" for doc in hasil])
    elif tipe == "qa":
        return "\n---\n".join([f"Pertanyaan Serupa: {doc['pertanyaan']}\nJawaban yang Disarankan: {doc['jawaban']}" for doc in hasil])
    return ""

# --- ALUR UTAMA APLIKASI WEB ---

# Muat semua data menggunakan fungsi cache
index_dokumen, db_dokumen, index_qa, db_qa = muat_dan_bangun_index()

if index_dokumen and index_qa:
    st.markdown("Sistem siap menjawab. Silakan ajukan pertanyaan Anda di bawah ini.")
    
    pertanyaan_user = st.text_input("Ketik pertanyaan Anda tentang keimigrasian di sini:", "")

    if pertanyaan_user:
        with st.spinner("Menganalisis dan mencari jawaban..."):
            # 1. Lakukan pencarian ganda
            konteks_dokumen = cari_info(pertanyaan_user, index_dokumen, db_dokumen, "dokumen", top_k=2)
            konteks_qa = cari_info(pertanyaan_user, index_qa, db_qa, "qa", top_k=1)
            
            # 2. Susun Prompt Hibrida
            prompt = f"""
            Anda adalah Sistem Pakar Keimigrasian Indonesia. Jawab pertanyaan pengguna dengan akurat, jelas, dan relevan berdasarkan informasi yang tersedia.
            Prioritaskan informasi dari "KONTEKS DARI JAWABAN SERUPA". Gunakan "KONTEKS DARI DOKUMEN HUKUM" sebagai pendukung.
            Tiru gaya jawaban dari "CONTOH CARA MENJAWAB".

            {FEW_SHOT_EXAMPLES}

            --- KONTEKS YANG DITEMUKAN ---
            [KONTEKS DARI JAWABAN SERUPA YANG SUDAH ADA]
            {konteks_qa}

            [KONTEKS DARI DOKUMEN HUKUM]
            {konteks_dokumen}
            --- AKHIR KONTEKS ---

            Berdasarkan semua informasi di atas, jawablah pertanyaan pengguna berikut.

            PERTANYAAN PENGGUNA:
            {pertanyaan_user}

            JAWABAN PAKAR:
            """
            
            # 3. Hasilkan Jawaban
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            st.divider()
            st.subheader("Jawaban")
            st.markdown(response.text)