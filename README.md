# 🎧 مشروع تحليل مشاعر المراجعات الصوتية باستخدام Hugging Face وWhisper

## وصف المشروع
يهدف هذا المشروع إلى تحليل مشاعر المستخدمين من خلال ملفات صوتية. يتم:
1. رفع ملف صوتي.
2. تحويله إلى نص باستخدام نموذج Whisper.
3. تحليل النص لاستخراج المشاعر باستخدام نموذج BERT.
4. عرض نتائج التحليل مع رسم بياني.

---

## المتطلبات

```bash
pip install openai-whisper transformers torchaudio librosa soundfile matplotlib

