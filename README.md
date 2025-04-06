# GreenGaze – Eco Waste Advisor (DeepSeek + Vision + Voice)

GreenGaze helps you upload or capture waste items and get sustainable disposal advice aligned with India’s SDG 11.

## 🔥 Features

- Object detection with your own trained model
- Eco advice from DeepSeek AI
- Text-to-Speech: it speaks your advice
- Feedback option for smarter future tips

## ⚙️ Setup

1. Clone this repo
2. Add `.env` file:
   ```
   GEMINI_API_KEY=your-deepseek-key-here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run greengaze_app.py
   ```
5. To train a model on your own like waste_classifier_model.h5

   ```

   Create a new notebook on google colab
   Go to this url for code : https://colab.research.google.com/drive/1ZCG8aBG1Ibg4vcw7Eg3EiyNgt5N4jtu6?usp=sharing
   ```

## 🧠 Powered by

- Gemini Chat API
- TensorFlow MobileNet
- Streamlit
- pyttsx3 for voice

MIT Licensed.
