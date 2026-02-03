
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import zipfile
import re

# Function to generate lyrics based on audio features
def generate_lyrics(model, tokenizer, audio_features, max_length=256, 
                   temperature=0.5, top_k=40, top_p=0.9):
    """Generate lyrics based on provided audio features"""
    
    # Create prompt from features
    prompt = "[FEATURES] "
    for key, value in audio_features.items():
        prompt += f"{key}: {value}, "
    prompt += "Generate hindi lyrics linewise in structured way"
    prompt += "[LANG: ROMANIZED_HINDI] [LYRICS]"
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to model's device
    device = model.device
    input_ids = input_ids.to(device)
    
    # Set attention mask
    attention_mask = torch.ones(input_ids.shape, device=device)
    
    # Generate lyrics
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,  # Prevent immediate repetition
        num_return_sequences=3   # Generate multiple alternatives
    )
    
    # Decode outputs and filter
    generated_texts = []
    for seq in output:
        generated_text = tokenizer.decode(seq, skip_special_tokens=False)
        
        # Extract only the lyrics part
        lyrics_match = re.search(r'\[LYRICS\](.*)', generated_text, re.DOTALL)
        if lyrics_match:
            lyrics = lyrics_match.group(1).strip()
            
            # Basic filtering for meaningful lyrics
            if len(lyrics.split()) > 20:  # Minimum word count
                generated_texts.append(lyrics)
    
    # Return the most semantically coherent text
    return generated_texts[0] if generated_texts else "Could not generate meaningful lyrics."
    
# Example of generating lyrics
sample_features = {
"tempo": 112.5,
"energy": 0.1343526, 
"loudness": -50.16268,
"danceability": 0.1596686712,
"speechiness": 19.04962,
"acousticness": 0.2,
"instrumentalness": 0.01,
"liveness": 0.15,
"valence": 22.6,
"Sentiment": "Sad"
}

# 1. Extract the zip (if not already done)
model_zip_path = "model.zip"
extract_dir = "./hindi_lyrics_model/kaggle/working/hindi_lyrics_gpt2"

with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 2. Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(extract_dir)
model = GPT2LMHeadModel.from_pretrained(extract_dir)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

generated_lyrics = generate_lyrics(model, tokenizer, sample_features)
print("Generated Lyrics:")
print(generated_lyrics)