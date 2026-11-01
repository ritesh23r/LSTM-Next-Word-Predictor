
# LSTM Next Word Predictor

An interactive next-word prediction application built using Deep Learning and deployed with Streamlit.

The application takes a user-provided sentence and predicts the most likely next word using a trained Long Short-Term Memory (LSTM) neural network. It can also generate multiple words iteratively.

## Features

- Predicts the next word from an input sentence.
- Displays the top 5 predicted words with their probabilities.
- Supports iterative multi-word text generation.
- Interactive interface built with Streamlit.
- Loads a pre-trained LSTM model and tokenizer.

## Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit

## Model Architecture

The model uses the following layers:

1. **Embedding Layer:** Converts word indices into dense vector representations.
2. **LSTM Layer:** Learns sequential patterns and relationships in text.
3. **Dense Layer:** Produces probabilities for the next word.
4. **Softmax Activation:** Converts model outputs into a probability distribution over the vocabulary.

## Project Structure

```text
LSTM-Next-Word-Predictor/
│
├── app.py
├── model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/LSTM-Next-Word-Predictor.git
```

### 2. Navigate to the project folder

```bash
cd LSTM-Next-Word-Predictor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. The user enters a sentence.
2. The tokenizer converts the text into numerical sequences.
3. The input sequence is padded to the model's expected length.
4. The trained LSTM model predicts probabilities for the next word.
5. The application maps predicted word indices back to actual words.
6. For multi-word generation, the predicted word is appended to the input and the process repeats.

## Learning Outcomes

Through this project, I explored:

- Text tokenization and numerical representation.
- Word embeddings.
- Recurrent Neural Networks and LSTM.
- Sequence padding and model input preparation.
- Softmax probability predictions.
- Integrating a trained Deep Learning model with Streamlit.

## Future Improvements

- Improve prediction quality with more training data.
- Add temperature-based sampling for more diverse generation.
- Deploy the application online.
- Experiment with different sequence models.

## Author

**Ritesh Das**

GitHub: https://github.com/rites23r

---

If you found this project interesting, feel free to explore the repository!
