import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import tempfile
import librosa
import torch
import numpy as np
import time
import base64
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the lyrics generation functions
from lyrics_generator import (
    initialize_model_and_tokenizer,
    generate_lyrics_from_sliders,
    generate_lyrics_from_audio
)

# Initialize session state for API debugging
if 'api_status_code' not in st.session_state:
    st.session_state['api_status_code'] = None
    
if 'api_response' not in st.session_state:
    st.session_state['api_response'] = None
    
if 'api_status' not in st.session_state:
    st.session_state['api_status'] = None

# Set page configuration
st.set_page_config(
    page_title="Hindi Lyrics Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3366FF;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #666666;
    }
    .output-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        color: #333333;
        white-space: pre-line;
    }
    .stProgress > div > div > div > div {
        background-color: #FF5733;
    }
    .api-success {
        padding: 0.75rem;
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .api-error {
        padding: 0.75rem;
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to enhance lyrics using language model API
def enhance_lyrics(lyrics, sentiment, rhyme_pattern):
    """
    Send the generated lyrics to language model API to make them more structured and semantically coherent
    """
    # Get API key from environment or Streamlit secrets
    try:
        api_key = os.environ.get("ENHANCEMENT_API_KEY") or st.secrets["ENHANCEMENT_API_KEY"]
    except Exception as e:
        st.warning(f"API key not found: {str(e)}. Using original lyrics.")
        return lyrics, False, "API key not found"
    
    # API endpoint
    api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Construct the prompt
    prompt = f"""
    Please improve the following Hindi lyrics to make them more structured, semantically coherent, and natural in Hindi language writen in Latin script.
    
    Original lyrics:
    ```
    {lyrics}
    ```
    
    Sentiment: {sentiment}
    Rhyming pattern: {rhyme_pattern}
    
    Please keep the same overall theme and sentiment, but improve the flow, structure, and semantic coherence.
    Make sure the rhyming scheme follows the pattern {rhyme_pattern}.
    Return only the improved lyrics without any explanations or comments.
    """
    
    # Prepare the request
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        # Send request to API
        response = requests.post(api_endpoint, headers=headers, json=data)
        response_json = response.json()
        
        # Log the response status and content for debugging
        st.session_state['api_status_code'] = response.status_code
        st.session_state['api_response'] = response_json
        
        # Extract the enhanced lyrics from the response
        if response.status_code == 200 and "candidates" in response_json:
            enhanced_lyrics = response_json["candidates"][0]["content"]["parts"][0]["text"]
            # Remove any markdown code block syntax if present
            enhanced_lyrics = enhanced_lyrics.replace("```", "").strip()
            print(f"Enhanced Lyrics: {enhanced_lyrics}")
            return enhanced_lyrics, True, f"Success (HTTP {response.status_code})"
        else:
            error_msg = f"API Error (HTTP {response.status_code}): "
            if "error" in response_json:
                error_msg += f"{response_json['error'].get('message', 'Unknown error')}"
            else:
                error_msg += "Unexpected response format"
            
            print(f"API Error: {error_msg}")
            return lyrics, False, error_msg
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        print(f"API Exception: {error_msg}")
        return lyrics, False, error_msg

def main():
    # Custom header
    st.markdown('<div class="main-header">Hindi Lyrics Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
        Generate Hindi lyrics using a fine-tuned GPT-2 model based on audio features, sentiment, and rhyming scheme.
        Enhanced for more structured and semantically coherent output.
    </div>
    """, unsafe_allow_html=True)
    
    # Define common sentiment options
    sentiment_options = ["Positive", "Negative", "Neutral", "Happy", "Sad", "Romantic", "Energetic", "Calm", "Nostalgic", "Angry"]
    
    # Define common rhyming patterns
    rhyme_patterns = {
        "AABB (Couplets)": "AABB",
        "ABAB (Alternating)": "ABAB",
        "ABBA (Enclosed)": "ABBA",
        "AAAA (Monorhyme)": "AAAA",
        "ABCB (Ballad)": "ABCB",
        "ABCD (Free Verse)": "ABCD",
        "AABCCB (Sestina-like)": "AABCCB"
    }
    
    # Set up tabs for the two different methods
    tab1, tab2 = st.tabs(["Generate from Audio", "Generate from Manual Features"])
    
    with tab1:
        st.markdown('<div class="sub-header">Generate Lyrics from Audio File</div>', unsafe_allow_html=True)
        
        # File uploader for audio
        uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, etc.)", type=["mp3", "wav", "ogg", "m4a"])
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment = st.selectbox("Select Sentiment", sentiment_options, key="sentiment_audio")
        
        with col2:
            rhyme_pattern_name = st.selectbox("Select Rhyming Scheme", list(rhyme_patterns.keys()), key="rhyme_audio")
            rhyme_pattern = rhyme_patterns[rhyme_pattern_name]
        
        # Advanced options
        with st.expander("Advanced Options"):
            chunk_duration = st.slider("Chunk Duration (seconds)", 5, 30, 15, key="chunk_duration")
            max_new_tokens = st.slider("Max New Tokens", 100, 500, 200, key="max_tokens_audio")
            context_tokens = st.slider("Context Tokens", 50, 200, 100, key="context_tokens")
            lines_per_chunk = st.slider("Lines Per Chunk", 2, 8, 4, key="lines_per_chunk")
            enhance_lyrics_option = st.checkbox("Enhance lyrics quality", value=True, key="enhance_lyrics_audio")
        
        # Generate button
        if st.button("Generate Lyrics from Audio", key="generate_audio"):
            if uploaded_file is not None:
                try:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Show processing status
                    status_text.text("Initializing model...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    status_text.text("Processing audio...")
                    progress_bar.progress(30)
                    time.sleep(0.5)
                    
                    status_text.text("Extracting audio features...")
                    progress_bar.progress(50)
                    time.sleep(0.5)
                    
                    status_text.text("Generating lyrics...")
                    progress_bar.progress(70)
                    
                    # Generate lyrics
                    lyrics = generate_lyrics_from_audio(
                        temp_file_path,
                        sentiment=sentiment,
                        rhyme_pattern=rhyme_pattern,
                        chunk_duration=chunk_duration,
                        max_new_tokens=max_new_tokens,
                        context_tokens=context_tokens,
                        lines_per_chunk=lines_per_chunk
                    )
                    
                    # Enhance lyrics if enabled
                    if enhance_lyrics_option:
                        status_text.text("Enhancing lyrics quality...")
                        progress_bar.progress(85)
                        enhanced_lyrics, api_success, api_status = enhance_lyrics(lyrics, sentiment, rhyme_pattern)
                        st.session_state['api_status'] = api_status
                        final_lyrics = enhanced_lyrics
                        
                        # Display API status
                        if api_success:
                            st.markdown(f'<div class="api-success">API Enhancement successful: {api_status}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="api-error">API Enhancement failed: {api_status}</div>', unsafe_allow_html=True)
                            st.markdown('<div class="info-text">Using original unenhanced lyrics.</div>', unsafe_allow_html=True)
                    else:
                        final_lyrics = lyrics
                    
                    progress_bar.progress(100)
                    status_text.text("Lyrics generated successfully!")
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                    # Display results
                    st.markdown('<div class="sub-header">Generated Lyrics</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="output-box">{final_lyrics}</div>', unsafe_allow_html=True)
                    
                    # Add download button for lyrics
                    download_button = create_download_button(final_lyrics, "hindi_lyrics.txt", "Download Lyrics")
                    st.markdown(download_button, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please upload an audio file.")
    
    with tab2:
        st.markdown('<div class="sub-header">Generate Lyrics from Manual Features</div>', unsafe_allow_html=True)
        
        # Basic parameters
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment = st.selectbox("Select Sentiment", sentiment_options, key="sentiment_manual")
            seed_phrase = st.text_input("Seed Phrase (Optional)", key="seed_phrase")
        
        with col2:
            rhyme_pattern_name = st.selectbox("Select Rhyming Scheme", list(rhyme_patterns.keys()), key="rhyme_manual")
            rhyme_pattern = rhyme_patterns[rhyme_pattern_name]
            lines_per_stanza = st.slider("Lines Per Stanza", 2, 8, 4, key="lines_per_stanza")
        
        # Audio feature sliders
        st.markdown('<div class="sub-header">Audio Features</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tempo = st.slider("Tempo", 60.0, 200.0, 120.0, key="tempo")
            energy = st.slider("Energy", 0.0, 1.0, 0.5, key="energy")
            loudness = st.slider("Loudness", -60.0, 0.0, -30.0, key="loudness")
            danceability = st.slider("Danceability", 0.0, 1.0, 0.5, key="danceability")
        
        with col2:
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5, key="speechiness")
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, key="acousticness")
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5, key="instrumentalness")
            liveness = st.slider("Liveness", 0.0, 1.0, 0.5, key="liveness")
        
        with col3:
            valence = st.slider("Valence", 0.0, 1.0, 0.5, key="valence")
            chroma = st.slider("Chroma", 0.0, 1.0, 0.5, key="chroma")
            spectral_contrast = st.slider("Spectral Contrast", 0.0, 1.0, 0.5, key="spectral_contrast")
            zero_crossings = st.slider("Zero Crossings", 0.0, 1.0, 0.5, key="zero_crossings")
        
        # Other parameters
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            explicit = st.selectbox("Explicit Content", ["False", "True"], key="explicit")
        
        with col2:
            popularity = st.slider("Popularity", 0, 100, 50, key="popularity")
            max_new_tokens = st.slider("Max New Tokens", 100, 500, 200, key="max_tokens_manual")
        
        with col3:
            enhance_lyrics_option = st.checkbox("Enhance lyrics quality", value=True, key="enhance_lyrics_manual")
        
        # Generate button
        if st.button("Generate Lyrics from Features", key="generate_manual"):
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Show processing status
                status_text.text("Initializing model...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                status_text.text("Processing features...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("Generating lyrics...")
                progress_bar.progress(70)
                
                # Generate lyrics
                lyrics = generate_lyrics_from_sliders(
                    seed_phrase=seed_phrase,
                    sentiment=sentiment,
                    rhyme_pattern=rhyme_pattern,
                    tempo=tempo,
                    energy=energy,
                    loudness=loudness,
                    danceability=danceability,
                    speechiness=speechiness,
                    acousticness=acousticness,
                    instrumentalness=instrumentalness,
                    liveness=liveness,
                    valence=valence,
                    explicit=explicit,
                    popularity=popularity,
                    chroma=chroma,
                    spectral_contrast=spectral_contrast,
                    zero_crossings=zero_crossings,
                    max_new_tokens=max_new_tokens,
                    lines_per_stanza=lines_per_stanza
                )
                
                # Enhance lyrics if enabled
                if enhance_lyrics_option:
                    status_text.text("Enhancing lyrics quality...")
                    progress_bar.progress(85)
                    enhanced_lyrics, api_success, api_status = enhance_lyrics(lyrics, sentiment, rhyme_pattern)
                    st.session_state['api_status'] = api_status
                    final_lyrics = enhanced_lyrics
                    
                    # Display API status
                    if api_success:
                        st.markdown(f'<div class="api-success">API Enhancement successful: {api_status}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="api-error">API Enhancement failed: {api_status}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="info-text">Using original unenhanced lyrics.</div>', unsafe_allow_html=True)
                else:
                    final_lyrics = lyrics
                
                progress_bar.progress(100)
                status_text.text("Lyrics generated successfully!")
                
                # Display results
                st.markdown('<div class="sub-header">Generated Lyrics</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="output-box">{final_lyrics}</div>', unsafe_allow_html=True)
                
                # Add download button for lyrics
                download_button = create_download_button(final_lyrics, "hindi_lyrics.txt", "Download Lyrics")
                st.markdown(download_button, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    # Add information about the model and features
    with st.expander("About the Model"):
        st.markdown("""
        This web app uses a fine-tuned GPT-2 model specifically trained on Hindi lyrics. The model generates
        lyrics based on audio features, sentiment, and rhyming scheme. The generated lyrics are further enhanced
        using the Gemini API to improve structural coherence and semantic meaning in Hindi.
        
        ### Audio Features:
        - **Tempo**: Speed of the music in BPM
        - **Energy**: Overall energy level of the track
        - **Loudness**: Overall loudness in dB
        - **Danceability**: How suitable the track is for dancing
        - **Speechiness**: Presence of spoken words
        - **Acousticness**: Confidence measure of whether the track is acoustic
        - **Instrumentalness**: Predicts whether a track contains no vocals
        - **Liveness**: Presence of an audience in the recording
        - **Valence**: Musical positiveness conveyed by a track
        - **Chroma**: Representation of the 12 different pitch classes
        - **Spectral Contrast**: The level of difference between peaks and valleys in the sound spectrum
        - **Zero Crossings**: Rate of sign-changes along a signal
        
        ### Rhyming Schemes:
        - **AABB**: Lines 1 and 2 rhyme, lines 3 and 4 rhyme (couplets)
        - **ABAB**: Lines 1 and 3 rhyme, lines 2 and 4 rhyme (alternating)
        - **ABBA**: Lines 1 and 4 rhyme, lines 2 and 3 rhyme (enclosed)
        - **AAAA**: All lines rhyme (monorhyme)
        - **ABCB**: Only lines 2 and 4 rhyme (ballad)
        - **ABCD**: No rhyming pattern (free verse)
        - **AABCCB**: Complex rhyming pattern
        
        ### Sentiment:
        The sentiment affects the emotional tone of the generated lyrics.
        
        ### Enhanced Output:
        The generated lyrics are optionally enhanced to improve:
        - Structure and flow of the lyrics
        - Semantic coherence and natural Hindi language usage
        - Adherence to the selected rhyming pattern
        - Overall quality and readability of the lyrics
        """)
    
    # Add API Debugging panel
    with st.expander("API Debugging"):
        st.markdown("### API Response Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Status")
            if st.session_state['api_status'] is not None:
                st.write(f"API Status: {st.session_state['api_status']}")
            else:
                st.write("No API call made yet")
                
            if st.session_state['api_status_code'] is not None:
                st.write(f"HTTP Status Code: {st.session_state['api_status_code']}")
        
        with col2:
            st.markdown("#### Test API Connection")
            if st.button("Test API Connection"):
                try:
                    # Get API key
                    try:
                        api_key = os.environ.get("ENHANCEMENT_API_KEY") or st.secrets["ENHANCEMENT_API_KEY"]
                        api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
                        
                        # Simple test request
                        headers = {
                            "Content-Type": "application/json",
                            "x-goog-api-key": api_key
                        }
                        
                        test_data = {
                            "contents": [{
                                "parts": [{
                                    "text": "Hello, testing API connection."
                                }]
                            }],
                            "generationConfig": {
                                "temperature": 0.7,
                                "maxOutputTokens": 10
                            }
                        }
                        
                        response = requests.post(api_endpoint, headers=headers, json=test_data)
                        st.session_state['api_status_code'] = response.status_code
                        st.session_state['api_response'] = response.json()
                        
                        if response.status_code == 200:
                            st.success(f"API Connection Successful! (HTTP {response.status_code})")
                        else:
                            st.error(f"API Error: HTTP {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"API Key Error: {str(e)}")
                except Exception as e:
                    st.error(f"Connection Test Error: {str(e)}")
        
        # Display full API response
        st.markdown("#### Raw API Response")
        if st.session_state['api_response'] is not None:
            with st.expander("Show Full Response"):
                st.json(st.session_state['api_response'])
        else:
            st.write("No API response data available")
            
        # API key configuration
        st.markdown("#### API Key Configuration")
        api_key_method = st.radio("API Key Source", 
                                ["Environment Variable", "Streamlit Secrets", "Manual Entry"],
                                horizontal=True)
        
        if api_key_method == "Environment Variable":
            st.info("Make sure the environment variable 'ENHANCEMENT_API_KEY' is set")
            env_api_key = os.environ.get("ENHANCEMENT_API_KEY")
            if env_api_key:
                st.success("Environment variable 'ENHANCEMENT_API_KEY' is set")
                # Show partial key for verification
                masked_key = env_api_key[:4] + "*" * (len(env_api_key) - 8) + env_api_key[-4:] if len(env_api_key) > 8 else "****"
                st.write(f"API Key (masked): {masked_key}")
            else:
                st.warning("Environment variable 'ENHANCEMENT_API_KEY' is not set")
                
        elif api_key_method == "Streamlit Secrets":
            st.info("Make sure the API key is set in Streamlit secrets with key 'ENHANCEMENT_API_KEY'")
            try:
                if "ENHANCEMENT_API_KEY" in st.secrets:
                    st.success("API key found in Streamlit secrets")
                    # Show partial key for verification
                    secret_key = st.secrets["ENHANCEMENT_API_KEY"]
                    masked_key = secret_key[:4] + "*" * (len(secret_key) - 8) + secret_key[-4:] if len(secret_key) > 8 else "****"
                    st.write(f"API Key (masked): {masked_key}")
                else:
                    st.warning("API key not found in Streamlit secrets")
            except Exception:
                st.error("Could not access Streamlit secrets")
                
        elif api_key_method == "Manual Entry":
            temp_api_key = st.text_input("Enter API Key", type="password")
            if st.button("Set Temporary API Key"):
                if temp_api_key:
                    os.environ["ENHANCEMENT_API_KEY"] = temp_api_key
                    st.success("Temporary API key set for this session")
                else:
                    st.error("Please enter an API key")
                    
        st.markdown("### API Troubleshooting Tips")
        st.markdown("""
        - Check if your API key is valid and has not expired
        - Verify you have sufficient quota/credits for the API
        - Ensure network connectivity to the API endpoint
        - Check if the model 'gemini-pro' is available and not deprecated
        - Make sure your prompt doesn't violate content policies
        - Try reducing the size of the input if it's very large
        """)

def create_download_button(text, filename, button_text):
    """Create a download button for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration:none;">'\
           f'<div style="display:inline-block; padding:0.5em 1em; color:white; background-color:#FF5733; '\
           f'border-radius:0.3em; text-align:center; font-weight:bold; cursor:pointer;">{button_text}</div></a>'
    return href

if __name__ == "__main__":
    main()