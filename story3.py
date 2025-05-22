import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure pad_token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Streamlit UI ---
st.set_page_config(page_title="AI Story Generator", layout="centered")
st.title("📚 AI Story Generator")
st.markdown("Create unique story continuations using GPT-2. Just enter a beginning and select a genre!")

# Genre selection
genre = st.selectbox(
    "Choose a Genre",
    ["Fantasy", "Horror", "Sci-Fi", "Romance", "Comedy", "Adventure", "Mystery", "Historical", "Custom"]
)

# User prompt input
user_input = st.text_area("✏️ Enter the beginning of your story", height=200, placeholder="Once upon a time in a distant land...")

# Generate button
generate = st.button("Generate Story ✨")

if generate and user_input.strip():
    with st.spinner("Generating..."):
        # Construct genre label
        genre_label = f"[Genre: {genre}]" if genre.lower() != "custom" else ""
        prompt = f"{genre_label}\n{user_input.strip()}\nContinue the story..."

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2,
            num_return_sequences=3,
        )

        st.markdown("### 📝 Generated Stories")
        story_options = []
        for i, output in enumerate(outputs):
            story = tokenizer.decode(output, skip_special_tokens=True)
            continuation = story[len(prompt):].strip()
            story_text = f"{user_input.strip()}\n\n{continuation}"
            story_options.append(story_text)
            with st.expander(f"Story {i+1}"):
                st.write(story_text)

        selected_index = st.selectbox("Choose a story to download", [f"Story {i+1}" for i in range(len(story_options))])
        selected_story = story_options[int(selected_index.split()[-1]) - 1]

        # Download button
        st.download_button(
            label="💾 Download Selected Story",
            data=selected_story,
            file_name="generated_story.txt",
            mime="text/plain"
        )

elif generate:
    st.warning("Please enter the beginning of your story to continue.")
