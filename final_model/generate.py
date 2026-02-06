import torch
import librosa
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model and Tokenizer Initialization
tokenizer = AutoTokenizer.from_pretrained("/home/samarth/SEM6/INLP/Project/final_model/Best_model/fine_tuned_impyadav_GPT2_hindi_lyrics")
model = AutoModelForCausalLM.from_pretrained("/home/samarth/SEM6/INLP/Project/final_model/Best_model/fine_tuned_impyadav_GPT2_hindi_lyrics")


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

def generate_lyrics_from_audio(audio_file_path, sentiment="Neutral", rhyme_pattern="AABB", chunk_duration=15, max_new_tokens=50, context_tokens=100, lines_per_chunk=4):
    """
    Generate lyrics from an audio file by chunking the audio, extracting features, and generating lyrics sequentially.
    
    Args:
        audio_file_path (str): Path to the input audio file.
        sentiment (str): Desired sentiment of the lyrics (default: "Neutral").
        rhyme_pattern (str): Rhyme scheme pattern (e.g., "AABB").
        chunk_duration (int): Duration of each audio chunk in seconds (default: 15).
        max_new_tokens (int): Maximum number of new tokens to generate per chunk (default: 50).
        context_tokens (int): Number of previous tokens to include in the prompt (default: 100).
        lines_per_chunk (int): Number of lines per stanza (default: 4).
    
    Returns:
        str: Formatted lyrics with stanzas.
    """
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
                temperature=0.85,
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
    
    # Extract lyrics lines with rhyme tags
    pattern = r'\[Line\](.*?)\[(.*?)\]\[EndLine\]'
    matches = re.findall(pattern, generated_text, re.DOTALL)
    
    # Format lyrics into stanzas
    formatted_lyrics = ""
    for i, (line, rhyme) in enumerate(matches):
        if i % lines_per_chunk == 0:
            if i > 0:
                formatted_lyrics += "\n"
            formatted_lyrics += f"Stanza_{(i // lines_per_chunk) + 1}:\n"
        formatted_lyrics += f"{line.strip()} ({rhyme})\n"
    
    return formatted_lyrics.strip()

# Example usage
if __name__ == "__main__":
    audio_file = "audios/calm.mp3"
    lyrics = generate_lyrics_from_audio(audio_file, sentiment="Romantic", rhyme_pattern="AABB")
    print("Generated Lyrics:\n", lyrics)