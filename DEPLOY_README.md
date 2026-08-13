# نشر التطبيق عالمياً - مجاني بالكامل

## الفكرة
- **Streamlit Community Cloud** = استضافة مجانية بتديك رابط عالمي (زي `yourapp.streamlit.app`) يفتح لأي حد في أي مكان في الدنيا.
- **Groq** = بديل مجاني لـ OpenAI (موديل Llama 3.3 سريع ومجاني).
- **Tavily** = محرك بحث بباقة مجانية (1000 بحث/شهر تقريباً).
- **Scraper** = مبني يدوي بـ `requests` + `BeautifulSoup`، مفيش أي API مدفوع فيه خالص.

يعني التطبيق كله مجاني 100%، وهيبقى شغال Global مش Local.

---

## الخطوات

### 1) اعمل حساب GitHub (لو مفيش عندك)
هترفع عليه الملفات دي: `app.py`, `requirements.txt`

### 2) اعمل Repository جديد وارفع الملفين
```bash
git init
git add app.py requirements.txt
git commit -m "procurement assistant app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```
(أو ارفعهم من واجهة GitHub مباشرة بزرار "Add file" → "Upload files")

### 3) روح Streamlit Community Cloud
افتح: **https://share.streamlit.io**
- سجّل دخول بحساب GitHub بتاعك
- دوس **"New app"**
- اختار الـ Repository اللي رفعته
- في خانة "Main file path" اكتب: `app.py`
- دوس **Deploy**

بعد دقيقة أو اتنين هيديك رابط زي:
```
https://your-app-name.streamlit.app
```
الرابط ده شغال عالمي، أي حد يقدر يفتحه من موبايله أو لابتوبه من أي دولة.

### 4) المفاتيح (API Keys)
عندك خيارين:

**الخيار أ (الأسهل):** سيب المستخدم (أو انت) يدخل المفاتيح في الشريط الجانبي جوه التطبيق نفسه (App بيطلبهم أوتوماتيك). مفيش أي إعداد إضافي مطلوب.

**الخيار ب (لو عايز تخبي مفاتيحك من المستخدمين):**
- في صفحة الـ App على Streamlit Cloud → **⋮ (Settings)** → **Secrets**
- الصق فيها محتوى ملف `secrets.toml.example` بعد ما تحط مفاتيحك الحقيقية
- التطبيق هيقراها تلقائي بدل ما المستخدم يدخلها بنفسه

### 5) فين تجيب المفاتيح المجانية؟
| المفتاح | الرابط | مجاني؟ |
|---|---|---|
| Groq | https://console.groq.com/keys | ✅ مجاني بالكامل |
| Tavily | https://app.tavily.com | ✅ Free tier (~1000 طلب/شهر) |

---

## ملحوظات مهمة
- التطبيق بيستخدم موديل `llama-3.3-70b-versatile` من Groq — سريع جداً ومجاني، لكن أضعف شوية من GPT-4 في الجودة. لو حبيت تجرب موديل تاني من Groq (زي `llama-3.1-8b-instant` للسرعة، أو `mixtral-8x7b-32768`) غيّر السطر ده في `app.py`:
  ```python
  llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.3)
  ```
- الـ Scraper المجاني (requests + BeautifulSoup) مش هيقدر يفتح كل المواقع (بعض المواقع بتحجب البوتات أو بتحتاج JavaScript rendering)، فبعض المنتجات ممكن ما يجبش بيانات كاملة عنها. ده الفرق الوحيد عن النسخة اللي بتستخدم ScrapeGraph المدفوعة.
- Streamlit Community Cloud مجاني لكن فيه حدود استخدام (الـ App بينام لو معملوش حد استخدام لفترة، ولازم تعمله "wake up" أول مرة).
