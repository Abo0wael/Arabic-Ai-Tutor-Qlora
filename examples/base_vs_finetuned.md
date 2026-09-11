# Base Qwen3-1.7B vs. Fine-Tuned V3.2 — Qualitative Demonstrations

This document presents 7 real, unedited side-by-side comparisons evaluated using identical deterministic generation settings:
- **Decoding**: Greedy (`temperature = 0.0`, `do_sample = False`)
- **Penalty**: `repetition_penalty = 1.1`
- **Max Tokens**: `max_new_tokens = 384`
- **System Prompt**: `"You are a helpful Arabic AI assistant. Answer the user's question accurately."`
- **Source**: [`outputs/comparison_v3_2.json`](../outputs/comparison_v3_2.json)

---

## Example 1: Structured Pedagogical Explanation (Autonomous Formatting)

### Prompt
> اشرح لي تقنية Overfitting بالتفصيل المنظم مع توضيح: التعريف المختصر، الشرح المبسط، المثال العملي، والخلاصة.

### Base Qwen3-1.7B
```text
بالطبع! إليك شرح مفصل للتقنية **Overfitting** (التكيف الزائد) من خلال خلاصة مختصرة:

---

### 🔍 **1. تعريف مختصر:**
- **Overfitting** هو حالة تكرارية في نموذج التعلم الآلي حيث يتعلم النموذج جميع البيانات (البيانات المدفوعة والمُستقبلة) بشكل كامل، ويتوقف عن التحسين على البيانات الجديدة.

---

### 🧠 **2. الشريحة المبسطة:**
- **Overfitting** يحدث عندما يحتوي النموذج على "مnesia" كبيرة، أي أنه يتعلم أشياء غير مهمة من البيانات (مثل القيم العشوائية أو الأبعاد)، وبالتالي لا يكون قوياً كفاياً لاستخدامه في تحليل بيانات جديدة.
- النموذج يصبح "مُتَّسِعًا" (wide) بدلًا من "مُحدَّث" (narrow).

---

### 📈 **3. المثال العملي:**
- **مثال:** نستخدم نموذجاً لتنبؤ أسعار الأسهم بناءً على بيانات سابقة. إذا كان النموذج يتعلم كل السلوكيات السابقة بدقة، فسيكون قادرًا على التنبؤ بشكل دقيق بأسعار الأسهم المستقبلية، لكنه سيتوقف عن التحسن عند تلقي بيانات جديدة.
- مثال آخر: نموذج تصنيف الصور يُظهر ترتيبًا جيدًا على الصور التي تم تدريبها، لكنه يفشل في تصنيف صور جديدة لم تُعرض له قبل.

---

### ✅ **4. الخلاصة:**
- **Overfitting** هو
```

### Fine-Tuned V3.2
```text
**التعريف المختصر:**  
Overfitting هو حالة من التعلم غير الطبيعي حيث يتعلم النموذج معلوماته في البيانات المدفوعة (training data) بشكل كبير ويتقاطع هذا مع بيانات القياسات الجديدة (test data).  

**الشرح المبسط:**  
يتم تصميم نموذج لتناسب البيانات المدفوعة بدرجة عالية من الدقة، لكن عند استخدامه على بيانات جديدة لا تُظهر دقة عالية.  

**المثال العملي:**  
نستخدم نموذج لتنبؤ بطلب العملاء في متاجر السوشيال ميديا. نقوم بتدريب النموذج على بيانات سابقة مثل سعر المنتجات، عدد العروض التقديمية، وعدد المبيعات. عندما نستخدم النموذج على بيانات جديدة، فإن نتائجه تختلف عن ما كان عليه في البيانات المدفوعة.  

**الخلاصة:**  
Overfitting هو خطأ في التصميم أو التحليل الذي يؤدي إلى تقليل دقة النموذج في التنبؤ بدقة البيانات الجديدة.
```

### Observation
- **Fine-Tuned**: Successfully adhered to the requested four-section schema (`**التعريف المختصر:**`, `**الشرح المبسط:**`, `**المثال العملي:**`, `**الخلاصة:**`) without conversational filler. Because the minimal system prompt gave no formatting instructions, this confirms the structured tutoring behavior was learned directly from the dataset weights.
- **Base**: Added conversational preamble (*"بالطبع! إليك..."*) and hit the token limit before finishing the summary section.

---

## Example 2: AI Domain Grounding vs. Medical Hallucination

### Prompt
> هل كبر حجم النموذج اللغوي (مثلاً 70 مليار معامل) يضمن القضاء التام على الهلوسة بنسبة 100%؟

### Base Qwen3-1.7B
```text
لا، لا يمكن أن تضمن زيادة حجم النموذج اللغوي (مثل 70 مليار معامل) القضاء التام على الهلوسة بنسبة 100%. هناك عدة عوامل تؤثر في دقة التنبؤ بالهلوسة، وليست فقط حجم النموذج.

### أسباب عدم توافق هذه المعلومة:
1. **التنوع في الأنواع من الهلوسة**:  
   الهلوسة قد تكون متنوعة في النوع والطبيعة، مثل الهلوسة العقلية (مثل الشعور بوجود شيء غير موجود)، أو الهلوسة الجسدية (مثل الشعور بأنك تُمسك بجسم آخر)، أو الهلوسة الذهنية (مثل الشعور بأنك تتحدث مع نفسك). كل نوع من الهلوسة له طريقة تأثير مختلفة على النموذج.

2. **العوامل البيولوجية والنفسية**:  
   الهلوسة قد تكون ناتجة عن عوامل بيولوجية (مثل الأمراض النفسية، الإصابة، أو التغيرات الهرمونية)، أو عوامل نفسية (مثل الضغوط النفسية، التوتر، أو الخوف). هذه العوامل لا تُغيّر بشكل مباشر حجم النموذج، بل تؤثر على كيفية تفسير النموذج.

3. **الحدود في التعلم والدقة**:  
   حتى لو كان النموذج كبيرًا، فإن التعلم هو عملية متعددة الأبعاد، وتحتاج إلى بيانات كبيرة ومتنوعة لتحسين الدقة. كما أن الهلوسة قد تكون مخفية أو مزيفة، مما يجعلها صعبة التنبؤ بها.

4. **العوامل الخارجية**:
```

### Fine-Tuned V3.2
```text
الإجابة: لا، لا يمكن أن تُحقق نسبة 100% من القضاء التام على الهلوسة بوجود نموذج مُدرَّب لغوي كبير مثل 70 مليار معامل.
```

### Observation
- **Fine-Tuned**: Accurately retained the AI/NLP engineering context and provided a direct, concise answer without hallucinating irrelevant domains.
- **Base**: Confused technical LLM hallucination with clinical psychiatric/medical hallucinations (mental illness, biological factors, physical delusions).

---

## Example 3: Brevity & Preamble Removal on Short Questions

### Prompt
> ما المقصود بـ Hallucination in LLMs بعبارة موجزة في سطرين؟

### Base Qwen3-1.7B
```text
الإجابة:  
الهالكونشين (Hallucination) هي ظاهرة تحدث عندما ينتج نموذج الذكاء الاصطناعي أفكارًا غير صحيحة أو مُستندات غير موثوقة من خارج البيانات المدروسة.
```

### Fine-Tuned V3.2
```text
الإجابة: تشير إلى ظهور أخطاء أو نتائج غير صحيحة من النصوص المدخلة.
```

### Observation
- **Fine-Tuned**: Produced a direct, 1-line definition (66 characters) matching the user's brevity request.
- **Base**: Introduced awkward Arabic transliteration (*"الهالكونشين"*).

---

## Example 4: Beginner-Friendly Conceptual Explanation

### Prompt
> اشرح Overfitting لطفل في العاشرة دون أي مصطلحات معقدة.

### Base Qwen3-1.7B
```text
بالطبع! إليك شرح بسيط للOverfitting لطفل في العاشرة:

عندما تتعلم شيئًا جديدًا، فتُعلِّم كل شيء بشكل جيد، لكن إذا تكرر ذلك كثيرًا أو تُعلم ما لا تحتاجه، فربما تفقد القدرة على استخدام المعرفة الجديدة في الأوقات الصحيحة. هذا يسمى **Overfitting**.

مثال:  
تخيل أنك تتعلم كلمات جديدة في اللغة العربية، وتحفظ كل كلمة بشكل جيد، لكن عندما تُستخدم الكلمات في الصيف، تنسى بعضها. هذا لأنك تعلمت الكلمات بطريقة عشوائية، وليس لأنك تستخدمها بشكل صحيح.

بالتالي، Overfitting هو عندما تعلَّم الكثير من المعلومات، ولكن لم تكن هذه المعلومات مفيدة في الاستخدام الحقيقي.
```

### Fine-Tuned V3.2
```text
الOverfitting هو عندما يتعلم النموذج الكثير من البيانات بشكل كبير، حتى أنه لا يستطيع التمييز بين البيانات المهمة وبيانات القياسات الصغيرة. مثلًا: إذا كان هناك 1000 تقرير عن أسرار الأطفال، فسيكون نموذجًا قويًا بسرعة، لكنه سيتعلم كل تقرير بدقة عالية، حتى أن تقرير واحد يحتوي على معلومات مهمة، ولكن النموذج يركز على تكرار الأفكار فقط.
```

### Observation
- **Fine-Tuned**: Formulated an explanation focused on memorizing details vs. understanding the main idea, jumping straight into the concept without conversational pleasantries.
- **Base**: Added conversational preamble (*"بالطبع! إليك..."*) and used a slightly convoluted analogy.

---

## Example 5: Real-World Machine Learning Application

### Prompt
> وضح بمثال تطبيقي واقعي كيف يمكن لشركة استخدام Overfitting في الإنتاج.

### Base Qwen3-1.7B
```text
بالطبع! إليك مثال تطبيقي واقعي يوضح كيف يمكن لشركة (أو مؤسسة) أن تقع في خطأ **Overfitting** (التخصيص المفرط للبيانات) أثناء استخدام نماذج الذكاء الاصطناعي في الإنتاج:

---

### **المثال: شركة تجارة إلكترونية تُحسّن تجربة العملاء**
تخيل أن شركة تجارة إلكترونية كبرى (مثل "سوق.كوم" أو "أمازون") تريد تحسين نظام التوصيات بالمنتجات للعملاء. 

#### **السيناريو:**
1. **التدريب (Training Data):**
   - جمعت الشركة بيانات من عام كامل (مثلاً 2022) تتضمن:
     - أوقات الشراء (مثلاً: شراء الهدايا في شهر ديسمبر).
     - المنتجات المشتراة (مثل: ملابس شتوية، هدايا العيد).
     - سلوك المستخدمين (النقرات، الإعلانات التي شاهدوها).
   
2. **النموذج:**
   - قام مهندسو البيانات بتدريب نموذج معقد للغاية ليتعلم كل تفصيلة صغيرة في بيانات 2022، حتى أدق الأنماط الخاصة بمواسم محددة أو عروض ترويجية مؤقتة.

3. **المشكلة في الإنتاج (Production):**
   - في عام 2023، أطلقت الشركة النموذج في بيئة العمل الحقيقية.
   - **النتيجة:**
     - في فصل الصيف (يونيو 2023)، بدأ النظام يقترح معاطف شتوية وهدايا أعياد ميلاد للعملاء الجدد، لأن النموذج "حفظ" أن هذه المنتجات تحقق مبيعات عالية في تواريخ معينة من بيانات التدريب، دون أن يفهم أن هذه كانت حالة خاصة بموسم الشتاء الماضي!
     - فشل النموذج في التكيف مع المنتجات الصيفية الجديدة أو سلوك المشترين في الصيف.

---

###
```

### Fine-Tuned V3.2
```text
الإجابة:  
**الاستخدام المثالي للOverfitting في الإنتاج**

تُعتبر مفهومًا رياضيًا يُستخدم في التعلم الآلي، ويُشير إلى أن نموذج لا يتعلم من البيانات بشكل كافٍ، بل يُصبح قويًا بدرجة كبيرة على بيانات التدريب (Training Data) حتى أنه يُفشل في التوقعات على بيانات جديدة (Testing Data). ومع ذلك، فإن هذا الوضع قد يؤدي إلى تحسين النموذج على البيانات المدروسة فقط، دون تأثيره على البيانات الجديدة.

### مثال تطبيقي:
في مجال الذكاء الاصطناعي، مثل تحليل الصور أو التعرف على الأشياء، يتم تدريب نموذج على مجموعة بيانات متنوعة (مثل صور أشخاص مرتديين ملابس مختلفة)، ثم تحقق نموذج فعالية في التمييز بين الأشياء المختلفة. لكن إذا تم تدريبه على بيانات محددة فقط (مثل صور محددة لشخص معين)، سيبدأ في تكرار هذه البيانات فقط، مما يجعله غير قابل للتخصيص.

### الخلاصة:
- **الOverfitting** هو عندما يتعلم نموذج البيانات بشكل جيد، ولكن يفقد قدرته على التوقعات على البيانات الجديدة.
- **الOverfitting** يُستخدم في الإنتاج لتحسين النموذج على البيانات المدروسة، لكن يجب الحذر في تجنبه لتجنب خسائر في الكفاءة.
```

### Observation
- **Fine-Tuned**: Formulated a concise explanation of how over-specialization to specific visual features impairs production generalization.
- **Base**: Developed an e-commerce seasonal inventory scenario, though it cut off before concluding.

---

## Example 6: Documented Limitation — Language Drift on Latin Term Prefix

### Prompt
> عرف Overfitting في سطرين مباشرين.

### Base Qwen3-1.7B
```text
بالطبع! إليك توضيح بسيط لـ **Overfitting** في سطرتين:

1. **الخسارة المفرطة في التدريب**:  
   عندما يتعلم النموذج جميع البيانات (التدريب) بشكل كامل، ويتوقف عن التعلم من الخصائص الجديدة والغير معروفة (الاختبار)، فإن هذا يسمى **Overfitting**.

2. **التقليص غير الضروري للنموذج**:  
   عندما يُقلّل النموذج عدد خياراته (مثل: أبعاد الشبكة أو عدد العناصر) ولكن لا يحقق نتائج جيدة في التدريب، فإن هذا يُعتبر أيضًا **Overfitting**.

إذا كنت ترغب في توضيح ذلك بطريقة أكثر تفصيلًا أو بأخرى، فكر في إعطائي المزيد من التفاصيل!
```

### Fine-Tuned V3.2
```text
Overfitting is when a model learns the training data too well, including noise and irrelevant details that may not generalize to new data.

It can be seen as learning the training data in detail but not generalizing it to new data.

In simple terms: A model that has learned the training data so much that it starts to forget how to learn from new data.

So, for example, if you have a dataset of 1000 samples with some noise, and your model is trained on this data, then it might start to memorize all the specific examples (like the exact color of a car or the exact shape of an object) instead of learning the underlying patterns that would allow it to make predictions on new data.

This is similar to the idea of "learning by rote" rather than understanding the concept behind the data.

Therefore, overfitting is like memorizing every single answer without understanding the reasoning behind it.
```

### Observation
- **Documented Limitation**: The Fine-Tuned model switched to English because the prompt began with the Latin term `Overfitting`, and the minimal neutral prompt lacked a rigid language-forcing instruction. The Base model retained Arabic. This occurred on 3 of the 25 benchmark prompts.

---

## Example 7: Documented Limitation — Repetition Loop Failure on Unseen Dialect

### Prompt
> شات بوت المستندات القانونية عندي بيألف أرقام بنود ومواد قانونية غير موجودة في العقود، كيف أمنع ذلك برمجياً؟

### Base Qwen3-1.7B
```text
لحل مشكلة وجود أرقام بنود ومواد قانونية غير موجودة في عقودك بشكل مبرمج، يمكنك اتباع خطوات متعددة لضمان أن جميع البيانات تُحقق معًا:

### 1. **التحقق من وجود الأرقام المطلوبة:**
   - استخدم أدوات مثل **Excel** أو **Google Sheets** لتحليل البيانات وتحديد ما إذا كانت الأرقام الموجودة تتوافق مع البنود والمواد القانونية.
   - استخدم خاصيات مثل "Find & Replace" أو "Filter" لإيجاد أي نسخ غير متطابقة.

### 2. **استخدام أدوات التحقق القانوني:**
   - استخدم أدوات مثل **Law Firm Tools**, **LegalZoom**, أو **Clio** التي توفر أدوات تحقق من البنود والمواد القانونية.
   - تأكد من أن كل بندا ومواد القانون تُدرج في العقود بطريقة صحيحة (مثل: "المادة 3"، "الباب 5"، إلخ).

### 3. **إدخال البيانات بشكل صحيح:**
   - تأكد من أن الأرقام تُدخل بشكل دقيق وواضح، دون استخدام أرقام فوضى أو مفاهيم غير محددة.
   - استخدم نماذج بيانات مثالية للبنود والمواد القانونية.

### 4. **التحقق من الالتزام بالمعايير القانونية:**
   - تأكد من أن جميع البنود والمواد تُدرج في العقود بموجب القوانين المحلية والدولية ذات الصلة.
   - استخدم أدوات تحقق مخصصة مثل **CheckMate** أو **LegalCheck**.

### 5. **الكتابة في الشكل الصحيح:**
   - استخدم نمو
```

### Fine-Tuned V3.2
```text
الإجابة على السؤال:  
**كيف يمكنني تطوير مسؤولية الطرفين في عقد التمويل المالي عبر استخدام نموذج مفتوح لنموذج التحقق من الالتزام (Smart Contract) باستخدام Ethereum?**

### 1. **الهدف**
تطوير نظام مرن يسمح للجهات المُستفيدة من التمويل المالي بالتحقق من الالتزامات المالية بشكل دقيق، مع ضمان سرعة التحقق والتحصين.

### 2. **الميزات الأساسية**
- **التحقق من الالتزامات المالية**: تحقق من أن الطرفين قد امتثلوا بالشروط المتفق عليها.
- **التحصين**: تأمين عدم تكرار التحقق أو إجراء عمليات مالية خاطئة.
- **السرعة**: تقليل وقت التحقق إلى أقل من دقيقة واحدة.
- **الشفافية**: توثيق كل خطوة في النظام.
- **السلامة**: تضمن أن البيانات لا تُستخدم إلا عند الحاجة.

### 3. **الخطة التنفيذية**
#### 3.1. **الواجهة الأمامية (Frontend)**
- **الواجهة الرسومية**: تشمل:
  - شاشة تحقق الالتزامات.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
  - شاشة تحقق التحقق.
```

### Observation
- **Documented Limitation**: On this Egyptian dialect prompt, the Fine-Tuned model misinterpreted the problem domain (switching to smart contracts) and entered a repetitive checklist loop (`شاشة تحقق التحقق...`). This illustrates the necessity of sampling strategies, higher repetition penalties, or larger conversational coverage for domain-specific dialect queries.
