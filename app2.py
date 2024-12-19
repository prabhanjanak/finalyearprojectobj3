import streamlit as st
import cv2
import numpy as np
import torch
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
import yt_dlp as youtube_dl
import speech_recognition as sr
import os

# Function to download the YOLOv11 model from GitHub
def download_yolov11_model():
    model_url = "https://github.com/prabhanjanak/finalyearprojectobj3/raw/main/yolo11n.pt"
    model_path = "yolo11n.pt"
    response = requests.get(model_url)
    
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("YOLOv11 model downloaded successfully!")
        return model_path
    else:
        print("Failed to download YOLOv11 model.")
        return None

# Function to download YouTube video (Full Video)
def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': './downloads/%(id)s.%(ext)s',
        'quiet': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_id = info_dict.get('id', '')
        video_file_path = f"./downloads/{video_id}.mp4"  # Full video path
        return video_file_path

# Function to extract frames from the downloaded video file
def extract_frames_from_video(file_path):
    video_capture = cv2.VideoCapture(file_path)
    frames = []
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    
    return frames

# Load YOLOv11 model
def load_yolov11_model():
    model_path = download_yolov11_model()
    if model_path:
        model = torch.load(model_path)
        return model
    else:
        print("Model not downloaded, exiting...")
        return None

# YOLOv11 object detection
def apply_yolov11_on_frames(frames):
    model = load_yolov11_model()  # Load YOLOv11 model
    detected_objects = []
    
    if model:
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
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
    
    summ = tokenizer.decode(outputs[0], skip_special_tokens=True)
    object_summary = ", ".join(detected_objects)
    final_summary = f"{summ} Additional detected objects in the video: {object_summary}"
    
    return final_summary

# Streamlit UI for dynamic user interaction
st.set_page_config(page_title="Court Session Video Summarizer with YOLOv11", layout="wide")

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

st.title("Court Session Video Summarizer with YOLOv11")
st.markdown("### Enter the YouTube video link below to get a summary along with object detection")

# Input for YouTube URL
url = st.text_input("Enter YouTube Video URL:")

# If Summarize button is clicked
if st.button("Summarize Video"):
    if url:
        with st.spinner("Processing video..."):
            # Step 1: Download video from the YouTube link
            video_file_path = download_youtube_video(url)
            
            # Step 2: Extract frames from the downloaded video
            frames = extract_frames_from_video(video_file_path)
            
            # Step 3: Detect objects using YOLOv11
            detected_objects = apply_yolov11_on_frames(frames)
            
            # Step 4: Generate transcript from the audio
            transcript = generate_transcript_from_audio(video_file_path)
            
            # Step 5: Generate summary using the transcript and detected objects
            summary = generate_summary_from_transcript_and_objects(transcript, detected_objects)
            
            # Display the summary
            st.markdown("### Summary:")
            st.write(summary)
    else:
        st.error("Please enter a valid YouTube video URL.")

# Clean up temporary files
if os.path.exists('./downloads'):
    for file in os.listdir('./downloads'):
        os.remove(os.path.join('./downloads', file))
    os.rmdir('./downloads')  # Remove the downloads directory after cleanup

st.success("Processing complete!")
