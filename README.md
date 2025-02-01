## Speech-to-Text and Spectrogram Analysis (Question 1)

This repository contains the implementation and analysis of **Speech-to-Text (STT) models** and **Spectrogram Windowing Techniques**, focusing on state-of-the-art methodologies, datasets, and performance evaluations.

## Overview
This project explores **speech-to-text (STT) systems** and their applications, including real-time transcription, virtual assistants, and accessibility enhancements. Additionally, it investigates different **spectrogram windowing techniques** used in Short-Time Fourier Transform (STFT) for signal processing.

The research involves analyzing **high-resource (English)** and **low-resource (Punjabi)** languages using multiple deep-learning models and evaluating **Word Error Rate (WER)** as a key performance metric. The second part of the study compares **Hann, Hamming, and Rectangular windows** to understand their effects on spectrogram generation and classification accuracy.

---


## Speech-to-Text (STT) Analysis

### State-of-the-Art Models
We evaluated the following STT models:

| Language | Model | Strengths | Limitations |
|----------|------------|--------------------|------------------|
| English | **Wav2Vec2-Base-960h** | High accuracy for clean data | Struggles with noise and accents |
| English | **SpeechT5-ASR** | Strong performance on speech data | High computational cost |
| Punjabi | **Wav2Vec2-Large-XLSR-Punjabi** | Works well in low-resource settings | Small dataset limits robustness |
| Punjabi | **Whisper** | Robust multilingual performance | High error rate, needs additional translation |

---
### Datasets
The datasets used for training and evaluation include:
- [LJSpeech Dataset (English)](https://www.kaggle.com/datasets/awsaf49/ljspeech-sr16k-dataset/data)
- [Punjabi Speech Dataset](https://data.mendeley.com/datasets/sdbc8f5b77/2)

We used the **first 500 sentences** from each dataset to evaluate STT models.


### Evaluation Metrics
- **Word Error Rate (WER)**:
  ```
  WER = (S + D + I) / N
  ```
  where:
  - **S** = Number of substitutions
  - **D** = Number of deletions
  - **I** = Number of insertions
  - **N** = Total words in the reference

---

### Results and Observations
- **For English:**
  - **Wav2Vec2-Base-960h** achieved the lowest WER for simple sentences.
  - **SpeechT5-ASR** performed well for general sentences but had higher WER for long or complex sentences.
- **For Punjabi:**
  - **Wav2Vec2-Large-XLSR-Punjabi** produced relatively stable results across sentence lengths.
  - **Whisper** showed consistent performance but had **higher base error rates**, requiring additional text-to-text translation.

---
### Question 2

## Spectrogram Windowing Analysis

### Windowing Methods
Three windowing techniques were analyzed:

| Aspect | **Hann Window** | **Hamming Window** | **Rectangular Window** |
|--------|---------------|----------------|----------------|
| **Frequency Resolution** | Moderate | Moderate-High | High |
| **Spectral Leakage** | Minimal | Low | High |
| **Edge Effects** | Smooth tapering | Gradual tapering | Abrupt transitions |
| **Best Use Case** | Music, audio | Speech signals | Synthetic signals |

---

### Comparative Analysis
- The **Hann window** effectively reduced spectral leakage, making it suitable for real-world applications.
- The **Hamming window** achieved the best balance between **frequency resolution and leakage suppression**.
- The **Rectangular window** had the **highest spectral leakage**, making it less effective for general speech analysis.

---
### SVM Classifier Performance
We evaluated the performance of a **Linear SVM classifier** using features extracted from spectrograms generated with different windowing techniques.

| Window Type | Train Accuracy | Test Accuracy | F1 Score | Precision | Recall |
|------------|---------------|--------------|---------|----------|--------|
| **Hann** | 0.835 | 0.779 | 0.779 | 0.782 | 0.779 |
| **Hamming** | **0.836** | **0.780** | **0.783** | **0.783** | **0.780** |
| **Rectangular** | 0.822 | 0.770 | 0.770 | 0.773 | 0.770 |

- **Hamming Window** achieves the highest accuracy and generalization performance.
- **Hann Window** provides competitive results with balanced performance.
- **Rectangular Window** has the lowest accuracy due to high spectral leakage.

These results emphasize the **importance of window selection** for feature extraction in spectrogram-based classification models.
---
## Music Genre Spectrogram Analysis

### Genre-wise Spectrogram Features
This section analyzes spectrograms of different music genres based on frequency range, intensity, dynamics, and temporal features.

| Aspect          | Romantic Song | Pop Song | Classical Song | Party Song |
|---------------|--------------|---------|--------------|------------|
| **Frequency Range** | 0-3000 Hz (Low-Mid) | 0-8000 Hz (Broad) | Balanced across all frequencies | 0-10,000 Hz (Full range) |
| **Energy Intensity** | Smooth and gradual | Rhythmic bursts | Subtle harmonic variations | High-intensity bursts |
| **Dynamics** | Slow transitions, emotional | Quick transitions, vibrant | Gradual, tonal complexity | Fast-paced, structured beats |
| **Temporal Features** | Prolonged tones, smooth shifts | Frequent peaks, high-energy | Continuous, rich harmonics | Regular rhythmic patterns |

### Comparative Summary
- **Romantic Songs**: Low-mid frequency emphasis, smooth and emotional transitions, dominated by soft harmonics.
- **Pop Songs**: Broader frequency range, repetitive rhythmic bursts, energetic dynamics.
- **Classical Songs**: Balanced frequencies, rich harmonic structures, subtle variations in intensity.
- **Party Songs**: Full-spectrum energy, strong bass, structured beats for high-energy music.

This analysis highlights how different windowing techniques and frequency distributions shape the auditory characteristics of various music genres.

---
## Installation & Usage

### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install librosa numpy matplotlib torch torchaudio scikit-learn
```

### Running the Speech-to-Text Models
1. Clone this repository:
   ```bash
   git clone https://github.com/m23csa516/speechunderstandingPA1.git
   cd speechunderstandingPA1
   ```
2. Run the Jupyter Notebooks for STT model evaluation:
   ```bash
   jupyter notebook Question1.ipynb
   jupyter notebook Question2.ipynb
   ```

### Running the Spectrogram Analysis
1. Open `Question2.ipynb` in Jupyter Notebook.
2. Execute the notebook to generate and compare spectrograms.

---

## References
- [Hugging Face Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Hugging Face SpeechT5](https://huggingface.co/microsoft/speecht5_asr)
- [Hugging Face Whisper](https://huggingface.co/openai/whisper-base)
- [STFT Documentation](https://librosa.org/doc/main/generated/librosa.stft.html)
- [Scikit-Learn SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

---

## Contributors
- **Himani (M23CSA516)** - Email: m23csa516@iitj.ac.in
- **Ankit Kumar Chauhan (M23CSA509)** - Email: m23csa509@iitj.ac.in

For any queries, please reach out via email.

---
