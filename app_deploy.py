import streamlit as st
import os
import faiss
import json
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
from pptx import Presentation
import time

# --- KODE DIAGNOSIS SEMENTARA ---
st.subheader("Pengecekan File di Server:")
try:
    # List semua file dan folder di direktori utama
    files_in_directory = os.listdir('.')
    st.code("\n".join(files_in_directory))
except Exception as e:
    st.error(f"Gagal membaca direktori: {e}")
st.divider()
# --- AKHIR KODE DIAGNOSIS ---

# --- Konfigurasi Halaman & API Key ---
st.set_page_config(page_title="Sistem Pakar Imigrasi", layout="wide")
st.title("🇮🇩 Sistem Pakar Izin Tinggal Keimigrasian Indonesia")

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
try:
    with open('database_qa.json', 'r', encoding='utf-8') as f:
        DATABASE_QA = json.load(f)
except FileNotFoundError:
    st.error("File 'database_qa.json' tidak ditemukan. Harap buat file tersebut.")
    # Provide a default empty list so the app doesn't crash
    DATABASE_QA = []

# --- Contoh Statis untuk Prompt ---
FEW_SHOT_EXAMPLES = "--- CONTOH CARA MENJAWAB ---\nPertanyaan: Apa itu penjamin?\nJawaban: Penjamin adalah orang atau korporasi yang bertanggung jawab atas keberadaan dan kegiatan Orang Asing selama berada di Wilayah Indonesia.\n--- AKHIR CONTOH ---"

# --- FUNGSI-FUNGSI DENGAN CACHING UNTUK DEPLOYMENT ---

@st.cache_resource
def muat_dan_bangun_index():
    """Fungsi ini sekarang akan memuat 3 sumber data: Dokumen (PDF & PPTX), Q&A, dan Tabel."""
    st.info("Memulai proses muat data dan pembangunan index (hanya sekali)...")
    
    # 1. Proses Dokumen (PDF & PPTX)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    semua_potongan = []
    
    for filename in os.listdir(FOLDER_DOKUMEN):
        teks_lengkap = ""
        file_path = os.path.join(FOLDER_DOKUMEN, filename)
        
        try:
            # Handle PDF files
            if filename.endswith('.pdf'):
                reader = PdfReader(file_path)
                teks_lengkap = "".join(page.extract_text() or "" for page in reader.pages)
            
            # --- BAGIAN BARU: Handle PowerPoint files ---
            elif filename.endswith('.pptx'):
                prs = Presentation(file_path)
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            teks_lengkap += shape.text + "\n"
            # -------------------------------------------

            if teks_lengkap:
                potongan_teks = text_splitter.split_text(teks_lengkap)
                for pot in potongan_teks:
                    semua_potongan.append({"sumber": filename, "konten": pot})

        except Exception as e:
            st.warning(f"Gagal memproses file {filename}: {e}")
    
    if not semua_potongan:
        st.error("Tidak ada dokumen yang bisa diproses.")
        # Return enough None values to match the expected output
        return None, None, None, None

    konten_dokumen = [doc['konten'] for doc in semua_potongan]
    embeddings_dokumen = genai.embed_content(model="models/text-embedding-004", content=konten_dokumen, task_type="RETRIEVAL_DOCUMENT")["embedding"]
    index_dokumen = faiss.IndexFlatL2(np.array(embeddings_dokumen).shape[1])
    index_dokumen.add(np.array(embeddings_dokumen, dtype='float32'))
    st.write(f"✅ Index Dokumen berhasil dibuat dengan {len(semua_potongan)} potongan.")

    # 2. Proses Database Q&A
    questions_qa = []
    for item in DATABASE_QA:
        if "pertanyaan" in item and item["pertanyaan"]:
            questions_qa.append(item["pertanyaan"])
        else:
        # This will show a warning in your app for any bad data, instead of crashing.
        st.warning(f"Melewatkan entri Q&A yang tidak valid atau kosong: {item}")
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




