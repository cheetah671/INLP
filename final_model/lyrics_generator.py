import os
import torch
import librosa
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Find the absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/home/samarth/SEM6/INLP/Project/final_model/Best_model/aniket27gupta03_lyricGPT-HINDI"

# Initialize model and tokenizer (load once when the module is imported)
def initialize_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens if they don't already exist
        special_tokens = [
            "[Sentiment]", "[Rhyme]", "[Tempo]", "[Energy]", "[Loudness]",
            "[Danceability]", "[Speechiness]", "[Acousticness]", "[Instrumentalness]",
            "[Liveness]", "[Valence]", "[Explicit]", "[Popularity]", "[Chroma]",
            "[SpectralContrast]", "[ZeroCrossings]", "[RhymeScheme]", "[Stanza]",
            "[EndStanza]", "[Line]", "[EndLine]"
        ]
        tokens_to_add = [token for token in special_tokens if token not in tokenizer.get_vocab()]
        if tokens_to_add:
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
            model.resize_token_embeddings(len(tokenizer))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise e

# Common Hindi rhyme endings
common_endings = {
    'A': 'na', 'B': 'ye', 'C': 'ta', 'D': 'aa', 'E': 'ra', 'F': 'di', 'G': 'ki',
    'H': 'se', 'I': 'le', 'J': 'ho', 'K': 'ka', 'L': 'ja', 'M': 'ni', 'N': 'me',
    'O': 'ga', 'P': 'sa'
}

def extract_audio_features(y, sr):
    """Extract audio features using Librosa from an audio segment."""
    try:
        features = {}
        
        # Duration
        duration = librosa.get_duration(y=y, sr=sr)
        features["duration_ms"] = duration * 1000
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = tempo
        
        # Energy
        features["energy"] = np.mean(librosa.feature.rms(y=y))
        
        # Loudness
        S = np.abs(librosa.stft(y))
        features["loudness"] = np.mean(librosa.amplitude_to_db(S, ref=np.max))
        
        # Danceability
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) > 0:
            beat_strength = np.mean([y[i] ** 2 for i in beats if i < len(y)])
            beat_intervals = np.diff(beats)
            regularity = 1.0 / (np.std(beat_intervals) + 1e-10)
            features["danceability"] = np.mean([beat_strength, regularity])
        else:
            features["danceability"] = 0.0
        
        # Speechiness
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["speechiness"] = np.mean(np.std(mfccs, axis=1))
        
        # Acousticness
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["acousticness"] = 1.0 - (np.mean(spectral_centroid) / (sr/2))
        
        # Instrumentalness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features["instrumentalness"] = np.mean(spectral_flatness)
        
        # Liveness
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features["liveness"] = np.mean(onset_env)
        
        # Valence
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["valence"] = np.mean(spectral_contrast)
        
        # Explicit content (placeholder)
        features["explicit"] = 0.0
        
        # Popularity (placeholder)
        features["popularity"] = 50.0
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma"] = np.mean(chroma)
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["spectral_contrast"] = np.mean(spectral_contrast)
        
        # Zero Crossings
        zero_crossings = librosa.zero_crossings(y, pad=False)
        features["zero_crossings"] = np.mean(zero_crossings)
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return {}

def format_lyrics(generated_text, words_per_line=5, lines_per_paragraph=4):
    """Format the generated lyrics by removing tags and organizing into paragraphs."""
    # Remove the prompt section
    prompt_end = generated_text.find("[Stanza]1[EndStanza]")
    if prompt_end != -1:
        lyrics_text = generated_text[prompt_end + len("[Stanza]1[EndStanza]"):]
    else:
        lyrics_text = generated_text

    # Remove tags like [A], [B], [C], etc.
    lyrics_text = re.sub(r'\[[A-Z]\]', '', lyrics_text)
    lyrics_text = re.sub(r'\[EndLine\]', '', lyrics_text)
    lyrics_text = re.sub(r'\[\w+\]', '', lyrics_text)  # Remove any remaining tags

    # Split the text into words
    words = lyrics_text.split()

    # Group words into lines
    lines = []
    for i in range(0, len(words), words_per_line):
        line = ' '.join(words[i:i + words_per_line])
        lines.append(line)

    # Group lines into paragraphs
    paragraphs = []
    for i in range(0, len(lines), lines_per_paragraph):
        paragraph = '\n'.join(lines[i:i + lines_per_paragraph])
        paragraphs.append(paragraph)

    # Join paragraphs with double newlines
    formatted_lyrics = '\n\n'.join(paragraphs)

    return formatted_lyrics

def generate_lyrics_from_sliders(
    seed_phrase="",
    sentiment="Neutral",
    rhyme_pattern="AABB",
    tempo=120,
    energy=0.5,
    loudness=-30,
    danceability=0.5,
    speechiness=0.5,
    acousticness=0.5,
    instrumentalness=0.5,
    liveness=0.5,
    valence=0.5,
    explicit="False",
    popularity=50,
    chroma=0.5,
    spectral_contrast=0.5,
    zero_crossings=0.5,
    max_new_tokens=200,
    lines_per_stanza=4
):
    """Generate Hindi lyrics using manual feature settings."""
    try:
        # Get model instances
        model, tokenizer, device = initialize_model_and_tokenizer()
        
        # Generate rhyme_scheme_list and rhyme_dict
        rhyme_scheme_list = list(rhyme_pattern)
        unique_letters = sorted(set(rhyme_pattern))
        rhyme_dict = {letter: common_endings.get(letter, 'na') for letter in unique_letters}
        rhyme_dict_str = str(rhyme_dict)
        rhyme_scheme_str = str(rhyme_scheme_list)
        
        # Construct prompt
        prompt = (
            f"[Sentiment]: {sentiment} {tokenizer.eos_token} "
            f"[Rhyme]: {rhyme_dict_str} {tokenizer.eos_token} "
            f"[Tempo]: {tempo} {tokenizer.eos_token} "
            f"[Energy]: {energy} {tokenizer.eos_token} "
            f"[Loudness]: {loudness} {tokenizer.eos_token} "
            f"[Danceability]: {danceability} {tokenizer.eos_token} "
            f"[Speechiness]: {speechiness} {tokenizer.eos_token} "
            f"[Acousticness]: {acousticness} {tokenizer.eos_token} "
            f"[Instrumentalness]: {instrumentalness} {tokenizer.eos_token} "
            f"[Liveness]: {liveness} {tokenizer.eos_token} "
            f"[Valence]: {valence} {tokenizer.eos_token} "
            f"[Explicit]: {explicit} {tokenizer.eos_token} "
            f"[Popularity]: {popularity} {tokenizer.eos_token} "
            f"[Chroma]: {chroma} {tokenizer.eos_token} "
            f"[SpectralContrast]: {spectral_contrast} {tokenizer.eos_token} "
            f"[ZeroCrossings]: {zero_crossings} {tokenizer.eos_token} "
            f"[RhymeScheme]: {rhyme_scheme_str}\n"
            "[Stanza]1[EndStanza]\n"
        )
        
        if seed_phrase:
            prompt += seed_phrase
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=True,
                temperature=0.6,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.15,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
        
        # Decode and format
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        formatted_lyrics = format_lyrics(generated_text, words_per_line=5, lines_per_paragraph=lines_per_stanza)
        
        return formatted_lyrics
    
    except Exception as e:
        print(f"Error generating lyrics from sliders: {e}")
        raise e

def generate_lyrics_from_audio(audio_file_path, sentiment="Neutral", rhyme_pattern="AABB", chunk_duration=30, max_new_tokens=200, context_tokens=100, lines_per_chunk=4):
    """Generate lyrics from an audio file by extracting features from the audio."""
    try:
        # Get model instances
        model, tokenizer, device = initialize_model_and_tokenizer()
        
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=None)
        chunk_size = int(chunk_duration * sr)
        y_chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]
        
        # Generate rhyme_dict based on rhyme_pattern
        unique_letters = sorted(set(rhyme_pattern))
        rhyme_dict = {letter: common_endings.get(letter, 'na') for letter in unique_letters}
        rhyme_dict_str = str(rhyme_dict)
        
        # Initialize running token IDs
        running_token_ids = torch.tensor([], dtype=torch.long).to(device)
        
        for k, y_chunk in enumerate(y_chunks):
            # Extract features for the current chunk
            features = extract_audio_features(y_chunk, sr)
            if not features:
                continue
            
            # Build features string
            features_str = (
                f"[Sentiment]: {sentiment} {tokenizer.eos_token} "
                f"[Rhyme]: {rhyme_dict_str} {tokenizer.eos_token} "
                f"[Tempo]: {features['tempo']} {tokenizer.eos_token} "
                f"[Energy]: {features['energy']} {tokenizer.eos_token} "
                f"[Loudness]: {features['loudness']} {tokenizer.eos_token} "
                f"[Danceability]: {features['danceability']} {tokenizer.eos_token} "
                f"[Speechiness]: {features['speechiness']} {tokenizer.eos_token} "
                f"[Acousticness]: {features['acousticness']} {tokenizer.eos_token} "
                f"[Instrumentalness]: {features['instrumentalness']} {tokenizer.eos_token} "
                f"[Liveness]: {features['liveness']} {tokenizer.eos_token} "
                f"[Valence]: {features['valence']} {tokenizer.eos_token} "
                f"[Explicit]: {features['explicit']} {tokenizer.eos_token} "
                f"[Popularity]: {features['popularity']} {tokenizer.eos_token} "
                f"[Chroma]: {features['chroma']} {tokenizer.eos_token} "
                f"[SpectralContrast]: {features['spectral_contrast']} {tokenizer.eos_token} "
                f"[ZeroCrossings]: {features['zero_crossings']} {tokenizer.eos_token} "
            )
            
            # Tokenize features string correctly
            features_token_ids = tokenizer(features_str, return_tensors='pt')['input_ids'].to(device)[0]
            
            if k == 0:
                # First chunk: Start with features and initial stanza
                stanza_str = f"[Stanza]1[EndStanza]\n"
                stanza_token_ids = tokenizer(stanza_str, return_tensors='pt')['input_ids'].to(device)[0]
                prompt_token_ids = torch.cat([features_token_ids, stanza_token_ids], dim=0)
            else:
                # Subsequent chunks: Use current features and last context_tokens
                last_context = running_token_ids[-context_tokens:] if len(running_token_ids) >= context_tokens else running_token_ids
                prompt_token_ids = torch.cat([features_token_ids, last_context], dim=0)
            
            # Generate lyrics
            with torch.no_grad():
                generated_ids = model.generate(
                    prompt_token_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )[0]
            
            # Extract new tokens (excluding the prompt)
            new_token_ids = generated_ids[len(prompt_token_ids):]
            running_token_ids = torch.cat([running_token_ids, new_token_ids], dim=0)
        
        # Decode the generated tokens into text
        generated_text = tokenizer.decode(running_token_ids, skip_special_tokens=False)
        
        # Format the lyrics
        formatted_lyrics = format_lyrics(generated_text, words_per_line=5, lines_per_paragraph=lines_per_chunk)
        
        return formatted_lyrics
    
    except Exception as e:
        print(f"Error generating lyrics from audio: {e}")
        raise e

# Initialize model on module import for faster subsequent calls
try:
    initialize_model_and_tokenizer()
except Exception as e:
    print(f"Warning: Failed to preload model: {e}")