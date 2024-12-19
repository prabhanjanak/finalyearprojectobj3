import streamlit as st
import cv2
import numpy as np
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from deep_translator import GoogleTranslator
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gtts import gTTS
from io import BytesIO
import yt_dlp as youtube_dl
import speech_recognition as sr
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to download YouTube video
def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': './downloads/%(id)s.%(ext)s',
        'quiet': True
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_id = info_dict.get('id', '')
        file_path = f"./downloads/{video_id}.mp3"  # Audio path
        return file_path
# Function to extract frames from YouTube video
def extract_frames_from_youtube_video(url):
    video_id = urlparse(url).query.split('v=')[1]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Use yt-dlp or any download mechanism to fetch the video
    video_capture = cv2.VideoCapture(video_url)
    frames = []
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    
    return frames
# YOLOv8 object detection
def apply_yolov8_on_frames(frames):
    # Load YOLOv8 model (using the latest model from ultralytics YOLOv8)
    model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # YOLOv8 Nano model for faster processing
    
    detected_objects = []
    
    for frame in frames:
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference with YOLOv8
        results = model(frame_rgb)
        
        # Extract labels of detected objects
        labels = results.names
        for *box, conf, cls in results.xywh[0]:
            label = labels[int(cls)]
            detected_objects.append(label)
    
    return detected_objects
# Function to generate a transcript from YouTube (audio)
def generate_transcript_from_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    
    try:
        transcript = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        transcript = "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        transcript = "Sorry, there was an issue with the audio service."
    
    return transcript
# Function to generate a transcript from YouTube (audio)
def generate_transcript_from_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    
    try:
        transcript = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        transcript = "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        transcript = "Sorry, there was an issue with the audio service."
    
    return transcript
# Function to generate a summary from the transcript and detected objects
def generate_summary_from_transcript_and_objects(transcript, detected_objects):
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    summ = tokenizer.decode(outputs[0])
    
    # Add detected objects to the summary
    object_summary = ", ".join(detected_objects)
    final_summary = f"{summ} Additional detected objects in the video: {object_summary}"
    
    return final_summary
# Streamlit UI for dynamic user interaction
st.set_page_config(page_title="Court Session Video Summarizer", layout="wide")

# Light/Dark mode toggle
theme = st.sidebar.radio('Select Theme', ['Light', 'Dark'])
if theme == 'Dark':
    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
        }
        .stTextInput input {
            background-color: #333;
            color: white;
        }
        .stSelectbox select {
            background-color: #333;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("Court Session Video Summarizer with YOLOv8")
st.markdown("### Enter the YouTube video link below to get a summary along with object detection")

# Input for YouTube URL
url = st.text_input("Enter YouTube Video URL:")

# Language Selection for Translation
languages_dict = {
    'en': 'English', 'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic', 'hy': 'Armenian', 
    'az': 'Azerbaijani', 'eu': 'Basque', 'be': 'Belarusian', 'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian',
    'ca': 'Catalan', 'ceb': 'Cebuano', 'ny': 'Chichewa', 'zh-cn': 'Chinese (simplified)', 'zh-tw': 'Chinese (traditional)',
    'co': 'Corsican', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'eo': 'Esperanto', 'et': 'Estonian',
    'tl': 'Filipino', 'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian', 'gl': 'Galician', 'ka': 'Georgian', 'de': 'German',
    'el': 'Greek', 'gu': 'Gujarati', 'ht': 'Haitian creole', 'ha': 'Hausa', 'haw': 'Hawaiian', 'he': 'Hebrew', 'hi': 'Hindi',
    'hmn': 'Hmong', 'hu': 'Hungarian', 'is': 'Icelandic', 'ig': 'Igbo', 'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian',
    'ja': 'Japanese', 'jw': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh', 'km': 'Khmer', 'ko': 'Korean', 'ku': 'Kurdish (kurmanji)',
    'ky': 'Kyrgyz', 'lo': 'Lao', 'la': 'Latin', 'lv': 'Latvian', 'lt': 'Lithuanian', 'lb': 'Luxembourgish', 'mk': 'Macedonian',
    'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese', 'mi': 'Maori', 'mr': 'Marathi', 'mn': 'Mongolian',
    'my': 'Myanmar (burmese)', 'ne': 'Nepali', 'no': 'Norwegian', 'or': 'Odia', 'ps': 'Pashto', 'fa': 'Persian', 'pl': 'Polish',
    'pt': 'Portuguese', 'pa': 'Punjabi', 'ro': 'Romanian', 'ru': 'Russian', 'sm': 'Samoan', 'gd': 'Scots gaelic', 'sr': 'Serbian',
    'st': 'Sesotho', 'sn': 'Shona', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali',
    'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili', 'sv': 'Swedish', 'tg': 'Tajik', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'ug': 'Uyghur', 'uz': 'Uzbek', 'vi': 'Vietnamese',
    'cy': 'Welsh', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zu': 'Zulu'
}
add_selectbox = st.sidebar.selectbox(
    "Select Language", list(languages_dict.values())
)
# If Summarize button is clicked
if st.button("Summarize Video[_{{{CITATION{{{_1{](https://github.com/animespotflix/yez/tree/77db45b4a063747aaa36dc16c246dbe2878de40f/README.md)[_{{{CITATION{{{_2{](https://github.com/shubham2652/Dragoman-Server/tree/87100ee13c94092b21fec3f5acfa565ac6d7ecbb/dragmon_server.py)[_{{{CITATION{{{_3{](https://github.com/innovation-discovery/se-scraper/tree/36c6aa7340f5f6b55bf64f59dfc0e79e68aeaccc/src%2Fmodules%2Fgoogle.js)[_{{{CITATION{{{_4{](https://github.com/mraykov/multilingo-km16/tree/7e7424ee643f68a3dd6b483c672a55f6abcb1bca/server%2Fsrc%2Fcommon%2Fsupported%20languages%2Fsupported-languages-s-l.ts)[_{{{CITATION{{{_5{](https://github.com/handleryouth/Translator_Tkinter/tree/f2741a3f127c8d2bdca8993bd4405449b909104b/app.py)[_{{{CITATION{{{_6{](https://github.com/Kxganzzz/galaxy-v7-new/tree/8fe6bd79d2ed0fae8461e9e27e29017d9980a72a/node_modules%2F@vitalets%2Fgoogle-translate-api%2Flanguages.js)[_{{{CITATION{{{_7{](https://github.com/mujohn26/translate-api/tree/b53419803565fd33d54d5268bead6e305bedb438/languages.js)
