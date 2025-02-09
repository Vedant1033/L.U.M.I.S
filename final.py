import os
import pathlib
import tempfile
import threading
import time
import pyautogui
import numpy as np
import cv2
import pyaudio
import wave
import moviepy.editor as mp
import keyboard
import pdfplumber
import streamlit as st
import google.generativeai as genai
import pygetwindow as gw
from pywinauto.application import Application
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  # Ensure API Key is set

def process_uploaded_files(uploaded_files):
    file_metadata = []
    for uploaded_file in uploaded_files:
        file_path = pathlib.Path(uploaded_file.name)
        file_path.write_bytes(uploaded_file.getvalue())  # Save file locally

        metadata = genai.upload_file(
            path=file_path,
            mime_type="application/pdf" if uploaded_file.type == "application/pdf" else "text/plain",
        )
        file_metadata.append(metadata)

    return file_metadata
def generate_rag_response(file_metadata, query):
    if not file_metadata or not query:
        return "No valid input provided."
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([*file_metadata, query])
    return response.text
# Streamlit UI Setup
st.set_page_config(page_title="LUMIS - LLM-based Unified Multimodal Intelligent System", layout="wide")
st.title("LUMIS - LLM-Powered Assistance")

# Sidebar: Task Selection
st.sidebar.title("Choose Task")
task = st.sidebar.selectbox(
    "Select task:",
    ["YouTube Video Transcription & Summarization", "Image & Text Processing", "Chatbot", "Summarize Multiple PDFs",  "T.A.P.A.S"]  
)

### 📌 1️⃣ YouTube Video Processing
def process_youtube_video(youtube_url):
    try:
        video_id = youtube_url.split("v=")[-1].split("&")[0]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript_data])

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Summarize this transcript:\n{transcript_text}")
        return response.text
    except (TranscriptsDisabled, NoTranscriptFound):
        return "❌ No transcript available for this video."
    except Exception as e:
        return f"❌ Error: {str(e)}"

### 📌 2️⃣ Image & Text Processing
def process_image_text(text_input=None, image_file=None):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if text_input and image_file:
        image = Image.open(image_file)
        response = model.generate_content([f"Analyze this image and text:\n{text_input}", image])
    elif image_file:
        image = Image.open(image_file)
        response = model.generate_content(["Analyze this image:", image])
    elif text_input:
        response = model.generate_content(f"Analyze this text:\n{text_input}")
    else:
        return "Please provide either text or an image."
    return response.text

### 📌 3️⃣ General Chatbot
def chatbot_tasks(user_input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Answer this query:\n{user_input}")
    return response.text

### 📌 4️⃣ RAG (Retrieval-Augmented Generation) for Multiple Files
def summarize_pdfs(uploaded_files):
    model = genai.GenerativeModel("gemini-1.5-flash")
    summaries = {}
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            pdf_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
        response = model.generate_content(f"Summarize this document:\n{pdf_text}")
        summaries[uploaded_file.name] = response.text
    return summaries

### 📌 5️⃣ Screen & Audio Recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio_filename = "output.wav"
video_filename = "output.mp4"
final_filename = "final_output.mp4"

def cleanup_files():
    """Deletes old files before recording starts."""
    for file in [audio_filename, video_filename, final_filename]:
        if os.path.exists(file):
            os.remove(file)

import keyboard  # Ensure you have the `keyboard` module installed

def record_audio(filename, stop_event):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            print(f"Audio recording error: {e}")
            break  # Exit on error

        # Check if 'Q' is pressed
        if keyboard.is_pressed('q'):
            stop_event.set()
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))



def record_screen(filename, stop_event):
    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 8, (screen_size.width, screen_size.height))
    while not stop_event.is_set():
        frame = np.array(pyautogui.screenshot())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def combine_audio_video():
    video_clip = mp.VideoFileClip(video_filename)
    audio_clip = mp.AudioFileClip(audio_filename)
    video_clip.set_audio(audio_clip).write_videofile(final_filename, codec='libx264')

def analyze_recording():
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(["Analyze this recording:", final_filename])
    return response.text

# 🔹 **Task Execution**
if task == "YouTube Video Transcription & Summarization":
    youtube_url = st.text_input("Enter YouTube Video URL")
    if youtube_url:
        st.subheader("Summary")
        st.write(process_youtube_video(youtube_url))

elif task == "Image & Text Processing":
    text_input = st.text_area("Enter text (optional)")
    image_file = st.file_uploader("Upload an Image (optional)", type=["jpg", "png", "jpeg"])
    if text_input or image_file:
        st.subheader("AI Response")
        st.write(process_image_text(text_input, image_file))

elif task == "Chatbot":
    user_input = st.text_area("Enter your query")
    if user_input:
        st.subheader("AI Response")
        st.write(chatbot_tasks(user_input))

elif task == "Summarize Multiple PDFs":
    uploaded_files = st.file_uploader("Upload files (PDF, TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    query = st.text_input("Enter your query related to the documents")
    
    if uploaded_files and query:
        file_metadata = process_uploaded_files(uploaded_files)
        response = generate_rag_response(file_metadata, query)
        st.subheader("AI Response")
        st.write(response)

elif task == "T.A.P.A.S":
    stop_event = threading.Event()
    
    if st.button("Start Recording"):
        cleanup_files()
        audio_thread = threading.Thread(target=record_audio, args=(audio_filename, stop_event))
        screen_thread = threading.Thread(target=record_screen, args=(video_filename, stop_event))
        audio_thread.start()
        screen_thread.start()
        st.write("Recording started... Press 'Q' to stop.")
        
        # Continuously check if stop_event is set (to update UI)
        while not stop_event.is_set():
            time.sleep(1)  # Prevent excessive CPU usage

        audio_thread.join()
        screen_thread.join()
        combine_audio_video()
        st.success("Recording Completed! Analyzing...")
        st.write(analyze_recording())

