"""Developer code snippets, trade-offs, and practical engineering scenarios for Dataset V3.2."""

CODE_SNIPPETS = [
    {
        "user": "كيف أطبق K-Fold Cross-Validation في بايثون باستخدام scikit-learn؟",
        "assistant": "باستخدام مكتبة `scikit-learn`، يمكنك تطبيقها في أربعة أسطر بسيطة:\n```python\nfrom sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier\n\nkf = KFold(n_splits=5, shuffle=True, random_state=42)\nmodel = RandomForestClassifier()\nscores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')\nprint(f'Average Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}')\n```\nلاحظ استخدام `shuffle=True` لخلط البيانات عشوائياً قبل التقسيم."
    },
    {
        "user": "كيف أضيف طبقة Dropout في شبكة عصبية بـ PyTorch؟",
        "assistant": "في PyTorch، تستدعي `nn.Dropout(p=0.3)` وتضعها بين الطبقات الخطية وتنشيطاتها:\n```python\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(128, 64)\n        self.dropout = nn.Dropout(p=0.3)  # إسقاط 30% من التنشيطات عشوائياً أثناء التدريب\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(64, 10)\n\n    def forward(self, x):\n        x = self.relu(self.dropout(self.fc1(x)))\n        return self.fc2(x)\n```\nتذكر استدعاء `model.eval()` أثناء الاختبار لتعطيل الـ Dropout تلقائياً."
    },
    {
        "user": "كيف أحسب F1-score و Confusion Matrix في بايثون؟",
        "assistant": "باستخدام `sklearn.metrics`:\n```python\nfrom sklearn.metrics import classification_report, confusion_matrix, f1_score\n\n# حساب F1 الإجمالي الموزون أو الماكرو\nf1 = f1_score(y_true, y_pred, average='macro')\nprint('Macro F1:', f1)\n\n# تقرير تفصيلي يوضح Precision و Recall و F1 لكل فئة\nprint(classification_report(y_true, y_pred))\nprint('Confusion Matrix:\\n', confusion_matrix(y_true, y_pred))\n```"
    },
    {
        "user": "كيف أضبط إعداد LoRA باستخدام مكتبة PEFT في بايثون؟",
        "assistant": "باستخدام مكتبة `peft` من Hugging Face:\n```python\nfrom peft import LoraConfig, get_peft_model\n\nlora_config = LoraConfig(\n    r=8,                     # رتبة المصفوفات المنخفضة\n    lora_alpha=16,           # معامل مقياس التحديث\n    lora_dropout=0.05,       # نسبة الإسقاط العشوائي\n    target_modules=['q_proj', 'v_proj'],  # وحدات الانتباه المستهدفة\n    bias='none',\n    task_type='CAUSAL_LM'\n)\nmodel = get_peft_model(base_model, lora_config)\nmodel.print_trainable_parameters()  # يطبع نسبة المعاملات القابلة للتدريب (~0.1%)\n```"
    },
    {
        "user": "كيف أحسب Cosine Similarity بين متجهين في بايثون؟",
        "assistant": "يمكنك استخدام `sklearn` أو `PyTorch`:\n```python\n# باستخدام scikit-learn\nfrom sklearn.metrics.pairwise import cosine_similarity\nsim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]\n\n# أو باستخدام PyTorch مباشرة على الـ GPU\nimport torch\nimport torch.nn.functional as F\nsim = F.cosine_similarity(vec1, vec2, dim=0).item()\n```\nالقيمة تتراوح بين -1 و 1، حيث 1 تعني تطابقاً تاماً في الاتجاه الدلالي."
    },
    {
        "user": "كيف أطبق Early Stopping في PyTorch لحماية النموذج من الـ Overfitting؟",
        "assistant": "تراقب خسارة التحقق (Validation Loss)؛ وإذا لم تتحسن لعدة دورات متتالية (Patience)، توقف التدريب وتحفظ أفضل أوزان:\n```python\nbest_loss = float('inf')\npatience = 5\npatience_counter = 0\n\nfor epoch in range(epochs):\n    train(model)\n    val_loss = evaluate(model)\n    if val_loss < best_loss:\n        best_loss = val_loss\n        torch.save(model.state_dict(), 'best_model.pt')\n        patience_counter = 0\n    else:\n        patience_counter += 1\n        if patience_counter >= patience:\n            print(f'Early stopping at epoch {epoch}')\n            break\n```"
    },
    {
        "user": "كيف أستخدم محسّن AdamW مع Gradient Clipping في بايثون؟",
        "assistant": "في حلقة التدريب بـ PyTorch:\n```python\nimport torch\nfrom torch.optim import AdamW\n\noptimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)\n\nfor batch in dataloader:\n    optimizer.zero_grad()\n    loss = model(batch)\n    loss.backward()\n    # قص التدرجات لحمايتها من الانفجار قبل تحديث الأوزان\n    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n    optimizer.step()\n```"
    },
    {
        "user": "كيف أقسم البيانات إلى Train و Validation و Test بدون تسريب بيانات؟",
        "assistant": "نقسم البيانات أولاً إلى تدريب ومجموعة مؤقتة، ثم نقسم المؤقتة إلى تحقق واختبار:\n```python\nfrom sklearn.model_selection import train_test_split\n\n# 80% تدريب، 20% مؤقت\nX_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n# تقسيم الـ 20% إلى 10% تحقق و 10% اختبار\nX_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)\n\n# نطبق المعالجة المسبقة بحساب المعاملات على X_train فقط، ثم نطبق التحويل على X_val و X_test\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_val = scaler.transform(X_val)\nX_test = scaler.transform(X_test)\n```"
    },
    {
        "user": "كيف أطبق Stratified K-Fold في بايثون مع بيانات غير متوازنة؟",
        "assistant": "نستخدم `StratifiedKFold` للحفاظ على نسبة الفئات في كل طي:\n```python\nfrom sklearn.model_selection import StratifiedKFold\n\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nfor fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n    X_tr, y_tr = X[train_idx], y[train_idx]\n    X_va, y_va = X[val_idx], y[val_idx]\n    print(f'Fold {fold}: Train size={len(train_idx)}, Val size={len(val_idx)}')\n```"
    },
    {
        "user": "كيف أكتب كود استدلال بنموذج Hugging Face مع تحديد Temperature و Top-p؟",
        "assistant": "باستخدام `model.generate`:\n```python\ninputs = tokenizer(prompt, return_tensors='pt').to(device)\noutputs = model.generate(\n    **inputs,\n    max_new_tokens=150,\n    temperature=0.7,   # ضبط درجة المغامرة في التوليد\n    top_p=0.9,         # استبقاء الكلمات التراكمية الأكثر ترجيحاً\n    do_sample=True,\n    repetition_penalty=1.15\n)\nresponse = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n```"
    },
    {
        "user": "كيف أحسب متجهات التضمين بنموذج Sentence Transformers في بايثون؟",
        "assistant": "باستخدام مكتبة `sentence-transformers`:\n```python\nfrom sentence_transformers import SentenceTransformer\n\nmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\nsentences = ['الذكاء الاصطناعي يغير العالم', 'تعلم الآلة فرع من الذكاء الاصطناعي']\nembeddings = model.encode(sentences, normalize_embeddings=True)\nprint('Shape:', embeddings.shape)  # (2, 384)\n```"
    },
    {
        "user": "كيف أطبق كشف القيم الشاذة باستخدام Interquartile Range (IQR) في بايثون؟",
        "assistant": "باستخدام `numpy` أو `pandas`:\n```python\nimport numpy as np\n\nq25, q75 = np.percentile(data, [25, 75])\niqr = q75 - q25\nlower_bound = q25 - 1.5 * iqr\nupper_bound = q75 + 1.5 * iqr\noutliers = data[(data < lower_bound) | (data > upper_bound)]\nprint(f'Detected {len(outliers)} outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]')\n```"
    },
    {
        "user": "كيف أطبق تقنية Gradient Accumulation في حلقة تدريب PyTorch؟",
        "assistant": "نقسم الخسارة على عدد الخطوات التراكمية ولا نصفر التدرجات إلا بعد اكتمالها:\n```python\naccum_steps = 4\noptimizer.zero_grad()\n\nfor i, (inputs, labels) in enumerate(dataloader):\n    outputs = model(inputs)\n    loss = criterion(outputs, labels) / accum_steps\n    loss.backward()\n    \n    if (i + 1) % accum_steps == 0:\n        optimizer.step()\n        optimizer.zero_grad()\n```"
    },
    {
        "user": "كيف أكتب كود تقطيع نصوص بسيط باستخدام Hugging Face Tokenizer؟",
        "assistant": "تستدعي الـ Tokenizer وتفحص الرموز والـ IDs:\n```python\nfrom transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')\ntext = 'تعلم الذكاء الاصطناعي ممتع ومفيد'\ntokens = tokenizer.tokenize(text)\ntoken_ids = tokenizer.encode(text)\nprint('Tokens:', tokens)\nprint('IDs:', token_ids)\n```"
    },
    {
        "user": "كيف أضبط أوزان الفئات (Class Weights) في دالة خسارة Cross-Entropy بـ PyTorch؟",
        "assistant": "نحسب مقلوب تكرار الفئات ونمرره لـ `CrossEntropyLoss`:\n```python\nimport torch\nimport torch.nn as nn\n\n# فئة 0 شائعة (900 عينة)، فئة 1 نادرة (100 عينة)\nweights = torch.tensor([1.0, 9.0], device='cuda')\ncriterion = nn.CrossEntropyLoss(weight=weights)\nloss = criterion(model_logits, target_labels)\n```"
    }
]

ENGINEERING_DEBUGGING = [
    {
        "user": "الـ GPU شغال بنسبة 100% بس التدريب بطيء جداً، إيه السبب؟",
        "assistant": "هذا العَرَض الشهير يعني غالباً وجود اختناق في نقل البيانات (Data Loading Bottleneck) وليس في حسابات المعالج الرسومي:\n1. المعالج (CPU) يستغرق وقتاً طويلاً في قراءة وفك ضغط وتجهيز عينات التدريب، مما يترك كرت الشاشة ينتظر وصول البيانات.\n2. الحل: ارفع عدد خيوط محمل البيانات في بايثون (`DataLoader(..., num_workers=4, pin_memory=True)`).\n3. تأكد من قراءة البيانات من قرص SSD سريع بدلاً من قرص صلب HDD ميكانيكي بطيء أو عبر شبكة محلية."
    },
    {
        "user": "النموذج بيطلع نتائج مختلفة تماماً في كل مرة أعمل Run، إيه الحل؟",
        "assistant": "عدم استقرار النتائج بين كل تشغيل يعني أن هناك عمليات حسابية تعتمد على أرقام عشوائية غير مثبتة:\n1. ثبت البذور العشوائية في بايثون وبايتورش:\n```python\nimport random, os, numpy as np, torch\nrandom.seed(42)\nnp.random.seed(42)\ntorch.manual_seed(42)\ntorch.cuda.manual_seed_all(42)\n```\n2. اضبط إعدادات cuDNN الحتمية:\n`torch.backends.cudnn.deterministic = True`\n`torch.backends.cudnn.benchmark = False`\n3. إذا كان نموذجاً لغوياً أثناء الاستدلال، فتأكد من ضبط `temperature=0` أو `do_sample=False` للحصول على إجابة حتمية واحدة."
    },
    {
        "user": "الـ F1 score عندي صفر للفئة الموجبة، مع إن الـ Accuracy 95%، أصلح ده إزاي؟",
        "assistant": "هذا تأكيد عملي على مشكلة عدم توازن الفئات (Class Imbalance)؛ النموذج تعلم التنبؤ بالفئة السالبة الغالبة دائماً:\n1. اضبط عتبة اتخاذ القرار (Classification Threshold)؛ خفضها من 0.5 إلى 0.2 مثلاً لتشجيع النموذج على التوقع الإيجابي.\n2. أضف أوزاناً دالة لدالة الخسارة تعاقب النموذج بشدة عند تفويت الفئة النادرة (`pos_weight` في PyTorch أو `class_weight='balanced'` في Scikit-Learn).\n3. طبق تقنية SMOTE لتوليد عينات اصطناعية من الفئة الأقلية في بيانات التدريب حصراً."
    },
    {
        "user": "النموذج اللغوي بيتكلم إنجليزي فجأة وأنا سائله بالعربي، ليه؟",
        "assistant": "هذا يحدث نتيجة انحياز لغوي في أوزان النموذج الأساسي (Pre-training Bias)؛ لأن الغالبية الساحقة من بيانات تدريب النماذج العامة هي باللغة الإنجليزية:\n1. تأكد من موجه النظام (System Prompt) واطلب صراحة: 'أجب على الأسئلة باللغة العربية الفصحى الواضحة حصراً'.\n2. إذا كنت تستخدم تقنية Few-Shot، فتأكد من أن جميع الأمثلة التوضيحية مكتوبة باللغة العربية.\n3. عند الضبط الدقيق (Fine-Tuning)، احرص على أن تكون مخرجات المساعد في بيانات التدريب عربية نقية مع الحفاظ فقط على المصطلحات التقنية الإنجليزية الأساسية."
    },
    {
        "user": "لما بستخدم RAG الإجابة بتطلع مبتورة في نص الجملة، المشكلة فين؟",
        "assistant": "بتر الإجابة في المنتصف ينشأ عادة من أحد سببين:\n1. معامل الحد الأقصى للرموز المولدة (`max_new_tokens`) صغير جداً ولا يتسع للإجابة الكاملة؛ ارفعه إلى 512 أو 1024 رمزاً.\n2. طول المستندات المسترجعة في موجه الـ RAG استهلك معظم نافذة السياق (Context Window)، مما لم يترك مساحة كافية للنموذج لتوليد رده؛ الحل هو تلخيص المستندات المسترجعة أو تقليص عدد المقاطع (Top-k Chunks)."
    },
    {
        "user": "خسارة التدريب بتنزل طبيعي بس فجأة بتسجل NaN، ما التشخيص الأرجح؟",
        "assistant": "ظهور `NaN` يعني حدوث فيض حسابي (Numerical Overflow) أو انفجار في التدرجات (Exploding Gradients):\n1. السبب الأرجح هو تضخم التدرجات؛ الحل الفوري هو تفعيل قص التدرجات: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.\n2. معدل التعلم قد يكون مرتفعاً جداً ودفع الأوزان لأرقام فلكية؛ اخفض الـ Learning Rate فوراً.\n3. فحص دالة الخسارة للتأكد من عدم وجود عملية لوغاريتم لصفر `log(0)` أو قسمة على صفر، واستخدام `torch.clamp` لحماية المقامات."
    },
    {
        "user": "بيظهر لي خطأ CUDA out of memory بمجرد بدء أول إيبوك، كيف أحلها دون تقليل النموذج؟",
        "assistant": "لعلاج CUDA OOM مع الحفاظ على معمارية النموذج:\n1. خفض حجم الدفعة (`batch_size=1` أو `2`) وعوض عن ذلك بـ Gradient Accumulation للحفاظ على حجم دفعة فعال كبير.\n2. فعل `model.gradient_checkpointing_enable()` لتوفير حتى 60% من ذاكرة التنشيطات مقابل زيادة طفيفة في وقت الحساب.\n3. استخدم دقة مختلطة مثل `torch.bfloat16` أو `torch.float16` بدلاً من FP32.\n4. احذف المصفوفات المؤقتة في بايثون واستدعِ `torch.cuda.empty_cache()` دورياً."
    },
    {
        "user": "النموذج بيعمل Overfitting من أول إيبوك بالظبط، إيه السبب السريع؟",
        "assistant": "حدوث Overfitting فوري في الدورة الأولى يشير عادة إلى:\n1. حجم بيانات التدريب شديد الصغر مقارنة بحجم معاملات النموذج.\n2. معدل التعلم مرتفع جداً مما دفع الأوزان لحفظ عينات البداية.\n3. تسريب بيانات مباشر (Data Leakage)؛ مثلاً تكرار نفس العينات في التدريب والتحقق.\n4. غياب أي نوع من التنظيم (Dropout = 0، Weight Decay = 0)."
    },
    {
        "user": "الـ Loss مبتنزلش خالص وواقفة عند رقم ثابت من أول التدريب لآخره.",
        "assistant": "ثبات الخسارة تماماً وعدم تغيرها يعني:\n1. معدل التعلم منخفض جداً لدرجة تجعل التحديثات شبه معدومة ($10^{-7}$ مثلاً)، ارفعه إلى $10^{-4}$.\n2. التدرجات تتلاشى تماماً (Vanishing Gradients) بسبب دوال مثل Sigmoid في شبكة عميقة.\n3. نسيت استدعاء `optimizer.step()` في حلقة التدريب بعد `loss.backward()`.\n4. الأوزان مجمدة كلها ولم يتم تفعيل `requires_grad = True` للمعاملات المراد تدريبها."
    },
    {
        "user": "الموديل بيطلع إجابة واحدة مكررة لكل المستخدمين أياً كان السؤال.",
        "assistant": "هذه الظاهرة تُعرف بانهيار التوليد (Mode Collapse) أو استسهال التوقع (Majority Class Shortcut):\n1. إذا كان مصنفاً: النموذج استنتج أن تصنيف كل المدخلات كفئة واحدة يحقق خسارة منخفضة كافية؛ أضف أوزان فئات لدالة الخسارة.\n2. إذا كان نموذجاً لغوياً: قد يكون التدريب تم بمعدل تعلم مرتفع دمر أوزان الانتباه فصار يولد رمز الفراغ أو أكثر كلمة تكراراً؛ أعد التدريب بـ Learning Rate محافظ وتأكد من سلامة دالة الخسارة."
    },
    {
        "user": "الذاكرة بتزيد تدريجياً مع كل إيبوك لحد ما يحصل Crash بعد ساعة تدريب.",
        "assistant": "الزيادة التدريجية في الذاكرة تدل على تسريب ذاكرة برمجية (Memory Leak) داخل حلقة التدريب:\n1. الخطأ الأكثر شيوعاً: جمع الخسارة لحساب المتوسط كـ `total_loss += loss` بدلاً من `total_loss += loss.item()`؛ فالأولى تحتفظ بكامل شجرة الحسابات (Computation Graph) في الذاكرة لكل دفعة.\n2. عدم حذف المتغيرات المؤقتة أو حفظ نواتج التنبؤ كاملة في قوائم بايثون دون استدعاء `.detach().cpu()`.\n3. تراكم السجلات (Logging) للمصفوفات مباشرة دون تحويلها لأرقام مجردة."
    },
    {
        "user": "زمن التوليد بيزيد جداً كل ما المحادثة تطول، إيه السبب والحل؟",
        "assistant": "زيادة زمن التوليد مع طول المحادثة ناتج عن تضخم طول السياق المدخل؛ لأن الانتباه الذاتي يحسب مصفوفة أطول لكل رمز جديد:\n1. استخدم آلية KV-Cache أثناء التوليد (`use_cache=True`) لتفادي إعادة حساب تضمينات الكلمات السابقة في كل رمز مولد.\n2. طبق استراتيجية نافذة منزلقة (Sliding Window) لقص المحادثات القديمة والاحتفاظ بآخر 4 إلى 6 رسائل فقط.\n3. استخدم FlashAttention-2 لتسريع حسابات السياق الطويل على المعالج الرسومي."
    }
]

STRICT_ONE_SENTENCE = [
    {
        "user": "عرف معامل الـ Learning Rate بعبارة موجزة في جملة واحدة.",
        "assistant": "الـ Learning Rate هو معامل يحدد حجم الخطوة التي يتخذها النموذج لتعديل أوزانه في اتجاه تقليل الخطأ عند كل تحديث."
    },
    {
        "user": "عرف الـ Overfitting في جملة واحدة حاسمة.",
        "assistant": "الـ Overfitting هو حفظ النموذج لتفاصيل وضوضاء بيانات التدريب مما يفقده القدرة على التعميم على أي بيانات جديدة."
    },
    {
        "user": "ما وظيفة دالة التفعيل في الشبكة العصبية بعبارة موجزة؟",
        "assistant": "تضيف دالة التفعيل اللاخطية إلى الشبكة العصبية لتمكينها من تعلم وتمثيل الأنماط المعقدة غير الخطية."
    },
    {
        "user": "ما هو الـ Self-Attention في جملة واحدة؟",
        "assistant": "هو آلية تتيح لكل رمز في الجملة التفاعل وحساب درجات الارتباط مع جميع الرموز الأخرى لتحديد معناه بدقة حسب السياق."
    },
    {
        "user": "ما هو الـ Temperature في النماذج اللغوية في سطر واحد؟",
        "assistant": "هو معامل يتحكم في عشوائية التوليد اللغوي؛ فخفضه يجعل النص حتمياً ومحافظاً، ورفعه يجعله متنوعاً وإبداعياً."
    },
    {
        "user": "عرف تقنية LoRA في جملة واحدة فقط دون استطراد.",
        "assistant": "LoRA هي تقنية تعديل كفؤة للنماذج بتجميد الأوزان الأصلية وتدريب مصفوفات صغيرة منخفضة الرتبة لتوفير الذاكرة والحوسبة."
    },
    {
        "user": "ما الفرق الجوهري بين RAG و Fine-Tuning في جملة واحدة؟",
        "assistant": "الـ RAG يزود النموذج بمعلومات ومستندات خارجية موثقة عند الطلب، بينما الـ Fine-Tuning يغير أسلوب وسلوك ونبرة إجابات النموذج."
    },
    {
        "user": "ما هي الهلوسة في الذكاء الاصطناعي بكلمات موجزة؟",
        "assistant": "هي توليد النموذج لمعلومات أو حقائق غير صحيحة مع صياغتها بأسلوب واثق ومقنع لغوياً."
    },
    {
        "user": "ما هو الـ Vector Database في جملة واحدة؟",
        "assistant": "هي قاعدة بيانات متخصصة في تخزين والبحث فائق السرعة في متجهات التضمين الرقمية بناءً على التشابه الدلالي في المعنى."
    },
    {
        "user": "عرف الـ Cross-Validation في جملة واحدة مباشرة.",
        "assistant": "هو أسلوب تقييم يقسم البيانات إلى K أجزاء ويدور التدريب والاختبار بينها لقياس ثبات وأداء النموذج بموضوعية."
    },
    {
        "user": "ما هو الـ Dropout في سطر واحد؟",
        "assistant": "هو تقنية تنظيمية تسقط عشوائياً نسبة من الخلايا العصبية أثناء التدريب لإجبار الشبكة على تعلم ميزات مستقلة وقوية."
    },
    {
        "user": "عرف تسريب البيانات (Data Leakage) في جملة واحدة حاسمة.",
        "assistant": "هو دخول معلومات غير مقصودة من بيانات الاختبار أو من المستقبل إلى التدريب مما يعطي دقة وهمية تنهار في الإنتاج."
    },
    {
        "user": "ما هي نافذة السياق (Context Window) في عبارة واحدة؟",
        "assistant": "هي الحد الأقصى لعدد الرموز التي يستطيع النموذج اللغوي استيعابها ومعالجتها دفعة واحدة في المحادثة."
    },
    {
        "user": "ما الفرق بين Precision و Recall في جملة واحدة؟",
        "assistant": "الـ Precision يقيس نسبة صحة الإنذارات الصادرة، بينما الـ Recall يقيس نسبة الحالات الفعلية التي نجح النموذج في التقاطها."
    },
    {
        "user": "عرف الـ Underfitting في سطر واحد فقط.",
        "assistant": "الـ Underfitting هو عجز النموذج عن استيعاب الأنماط الأساسية في البيانات بسبب بساطته الزائدة أو قلة تدريبه."
    },
    {
        "user": "ما هو الـ Tokenizer في معالجة اللغات الطبيعية بعبارة موجزة؟",
        "assistant": "هو أداة تقسم النصوص الخام إلى وحدات بنائية رقمية (رموز) تفهمها وتعالجها النماذج اللغوية."
    },
    {
        "user": "ما وظيفة التضمين (Embedding) في سطر واحد؟",
        "assistant": "يحول الكلمات والمفاهيم النصية إلى متجهات رقمية كثيفة تعكس التقارب والتشابه في المعنى الدلالي."
    },
    {
        "user": "ما هو الـ Gradient Descent في جملة واحدة؟",
        "assistant": "هو خوارزمية تحسين رياضية تعدل أوزان النموذج خطوة بخطوة في عكس اتجاه التدرج للوصول لأدنى خطأ ممكن."
    },
    {
        "user": "ما هو الـ Batch Size في عبارة موجزة؟",
        "assistant": "هو عدد عينات التدريب التي يمررها النموذج لحساب الخسارة قبل تحديث أوزانه دفعة واحدة."
    },
    {
        "user": "عرف الـ Baseline Model في سطر واحد.",
        "assistant": "هو نموذج مرجعي بسيط نستخدمه كنقطة بداية أساسية لمقارنة وقياس أي تحسن تقدمه النماذج المعقدة."
    },
    {
        "user": "ما وظيفة الـ Early Stopping في جملة واحدة؟",
        "assistant": "هو آلية توقف التدريب تلقائياً عندما تبدأ خسارة التحقق في الارتفاع لحماية النموذج من الـ Overfitting."
    },
    {
        "user": "ما الفرق بين Hyperparameter و Parameter في سطر واحد؟",
        "assistant": "الـ Parameters يتعلمها النموذج بنفسه أثناء التدريب، بينما الـ Hyperparameters يحددها المطور يدوياً قبل بدء التدريب."
    },
    {
        "user": "ما هو الـ Quantization في جملة واحدة؟",
        "assistant": "هو تقليل دقة الأرقام الحسابية لأوزان النموذج لتخفيض حجم الذاكرة وتسريع الاستدلال بأقل فقد في الدقة."
    },
    {
        "user": "عرف هندسة الميزات (Feature Engineering) في سطر واحد.",
        "assistant": "هي تحويل واستخراج سمات رقمية ذات دلالة قوية من البيانات الخام لرفع كفاءة وسرعة تعلم النماذج."
    },
    {
        "user": "ما هو الـ Prompt Engineering في عبارة واحدة؟",
        "assistant": "هو فن صياغة وهيكلة مدخلات النصوص بدقة لتوجيه النموذج اللغوي نحو إنتاج الإجابة الأنسب للمهمة."
    }
]
