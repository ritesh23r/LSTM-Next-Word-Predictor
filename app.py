
import streamlit as st
import numpy as np
import keras
import pickle
from pathlib import Path


# ==========================================
# 1. PROJECT FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# ==========================================
# 2. LOAD MODEL AND SUPPORTING FILES
# ==========================================

@st.cache_resource
def load_files():

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    with open(MAX_LEN_PATH, "rb") as file:
        max_len = pickle.load(file)

    return model, tokenizer, int(max_len)


# ==========================================
# 3. PADDING FUNCTION
# ==========================================

def prepare_input(text, tokenizer, max_len):

    sequence = tokenizer.texts_to_sequences([text])

    # Create an array filled with zeros
    padded = np.zeros((1, max_len), dtype=np.int32)

    # Keep the most recent words
    tokens = sequence[0][-max_len:]

    # Pre-padding: place tokens at the end
    if len(tokens) > 0:
        padded[0, -len(tokens):] = tokens

    return padded


# ==========================================
# 4. NEXT WORD PREDICTION
# ==========================================

def predict_next_word(text, model, tokenizer, max_len):

    padded = prepare_input(text, tokenizer, max_len)

    predictions = model.predict(padded, verbose=0)[0]

    # Get the five highest probability predictions
    top_indices = np.argsort(predictions)[-5:][::-1]

    results = []

    for index in top_indices:

        word = tokenizer.index_word.get(int(index))

        if word is not None:
            confidence = float(predictions[index]) * 100

            results.append({
                "word": word,
                "confidence": confidence
            })

    return results


# ==========================================
# 5. STREAMLIT INTERFACE
# ==========================================

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 LSTM Next Word Predictor")

st.write(
    "Enter a sentence and let the LSTM model "
    "predict what word comes next."
)

st.divider()


# ==========================================
# 6. LOAD FILES
# ==========================================

try:

    with st.spinner("Loading your trained LSTM model..."):

        model, tokenizer, max_len = load_files()

    st.success("Model and supporting files loaded successfully!")

except Exception as error:

    st.error(f"Error loading files: {error}")
    st.stop()


# ==========================================
# 7. DISPLAY MODEL INFORMATION
# ==========================================

with st.expander("Model Information"):

    st.write("**Maximum sequence length:**", max_len)

    st.write(
        "**Vocabulary size:**",
        len(tokenizer.word_index)
    )

    st.write("**Model input shape:**", model.input_shape)

    st.write("**Model output shape:**", model.output_shape)


st.divider()


# ==========================================
# 8. USER INPUT
# ==========================================

user_text = st.text_area(
    "Enter your sentence",
    placeholder="Example: I love machine learning",
    height=120
)

number_of_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=10,
    value=1
)


# ==========================================
# 9. PREDICTION BUTTON
# ==========================================

if st.button("Predict Next Word", type="primary"):

    if not user_text.strip():

        st.warning("Please enter a sentence first.")

    else:

        current_text = user_text.strip()

        st.subheader("Prediction")

        for step in range(number_of_words):

            results = predict_next_word(
                current_text,
                model,
                tokenizer,
                max_len
            )

            if not results:

                st.warning(
                    "No known words found for prediction."
                )
                break

            best_prediction = results[0]

            predicted_word = best_prediction["word"]

            confidence = best_prediction["confidence"]

            current_text += " " + predicted_word

            st.write(
                f"**Step {step + 1}:** {predicted_word}"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            st.caption(
                f"Model probability: {confidence:.2f}%"
            )

        st.success("Generated text:")

        st.write(current_text)

        # Display alternative predictions for the first word
        st.subheader("Top 5 Predictions")

        first_results = predict_next_word(
            user_text.strip(),
            model,
            tokenizer,
            max_len
        )

        for result in first_results:

            st.write(
                f"**{result['word']}** — "
                f"{result['confidence']:.2f}%"
            )
