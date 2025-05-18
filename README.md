# تثبيت المكتبات اللازمة
!pip install openai-whisper transformers torchaudio librosa soundfile matplotlib

import whisper
from transformers import pipeline
import matplotlib.pyplot as plt
import IPython.display as ipd
from google.colab import files

# تحميل نموذج Whisper
asr_model = whisper.load_model("base")

# تحميل نموذج تحليل المشاعر
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# رفع ملف صوتي
uploaded = files.upload()
audio_path = list(uploaded.keys())[0]

# تشغيل الملف
print(f"تشغيل الملف: {audio_path}")
ipd.display(ipd.Audio(audio_path))

# تحويل الصوت إلى نص
print("🔄 تحويل الصوت إلى نص...")
result = asr_model.transcribe(audio_path)
text = result['text']
print("✅ تم استخراج النص:")
print(text)

# تحليل المشاعر
print("🔍 تحليل المشاعر...")
sentiment = sentiment_pipeline(text)
print("✅ النتيجة:")
print(sentiment)

# رسم بياني
labels = [s['label'] for s in sentiment]
scores = [s['score'] for s in sentiment]

plt.bar(labels, scores, color='skyblue')
plt.xlabel("نوع الشعور")
plt.ylabel("الدرجة")
plt.title("توزيع المشاعر")
plt.show()
