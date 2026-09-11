"""Assemble Dataset V3.2 with Behavioral Contrasts, Neutral System Prompt, and Zero Templating."""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_2.kb_data import CONCEPTS
from src.v3_2.kb_data_part2 import CONCEPTS_PART2
from src.v3_2.kb_data_part3 import CONCEPTS_PART3
from src.v3_2.kb_data_part4 import CONCEPTS_PART4
from src.v3_2.behavioral_contrasts import BEHAVIORAL_CONTRASTS
from src.v3_2.vague_and_dialogue import VAGUE_CLARIFICATIONS, UNCERTAINTY_AND_SCOPE, FOLLOW_UPS
from src.v3_2.developer_scenarios import CODE_SNIPPETS, ENGINEERING_DEBUGGING, STRICT_ONE_SENTENCE
from src.v3_2.pedagogical_contrasts import DECISION_TRADEOFFS, NEGATIVE_CONSTRAINTS, STUDENT_PITFALLS

from src.utils import SYSTEM_PROMPT

SYSTEM_PROMPT_NEUTRAL = SYSTEM_PROMPT

ALL_CONCEPTS = {}
ALL_CONCEPTS.update(CONCEPTS)
ALL_CONCEPTS.update(CONCEPTS_PART2)
ALL_CONCEPTS.update(CONCEPTS_PART3)
ALL_CONCEPTS.update(CONCEPTS_PART4)

# Custom summaries for explicitly requested 4-section structured explanations (strictly 15 concepts)
CUSTOM_STRUCTURED_SUMMARIES = {
    "overfitting": "معالجة Overfitting تهدف لتحسين التعميم، ويجب قياس أثرها على بيانات مستقلة دون ضمان أداء مستقبلي ثابت.",
    "underfitting": "معالجة Underfitting تتطلب زيادة سعة النموذج واستخراج ميزات غير خطية كافية لتمثيل العلاقات الحقيقية في البيانات.",
    "accuracy_and_imbalance": "في البيانات غير المتوازنة، يجب استبدال Accuracy بمقاييس أدق مثل F1-score و Precision و Recall لحماية الفئات النادرة.",
    "learning_rate": "معدل التعلم هو المتحكم في استقرار وسرعة التقارب، واستخدام مجدول تدريجي مع إحماء هو الخيار الأكثر أماناً.",
    "cross_validation": "التحقق المتقاطع يوفر تقييماً إحصائياً نزيهاً وموثوقاً لأداء النموذج ويقلل التباين الناتج عن التقسيم العشوائي الفردي.",
    "dropout": "الـ Dropout يمنع الاعتماد الزائد على ميزات محددة أثناء التدريب فقط، ويُعطل تماماً أثناء الإنتاج للحصول على توقع حتمي.",
    "self_attention": "الانتباه الذاتي يمنح النموذج قدرة فائقة على فهم السياق عبر ربط جميع الكلمات ببعضها بالتوازي ولحظياً.",
    "temperature_and_sampling": "نضبط Temperature عند الصفر للمهام المنطقية والبرمجية الصارمة، ونرفعه للمهام التوليدية الإبداعية لزيادة التنوع.",
    "lora_and_qlora": "LoRA تدرب مصفوفات منخفضة الرتبة مع تجميد الأوزان الأساسية؛ نسبة المعاملات المدربة تعتمد على الرتبة والطبقات المستهدفة.",
    "rag_vs_finetuning": "نستخدم RAG لربط النموذج بالمعلومات والمستندات الحية، ونستخدم Fine-Tuning لضبط السلوك والأسلوب التواصلي المتخصص.",
    "hallucination": "يمكن تقليل بعض الهلوسات باستخدام مصادر موثوقة والتحقق من الإجابات والاعتراف بنقص المعلومات. RAG خيار مساعد وليس ضماناً، وخفض الحرارة لا يضمن صحة الحقائق.",
    "activation_functions": "دوال التفعيل غير الخطية هي المحرك الأساسي لعمق الشبكات العصبية وبدونها تفقد الشبكة قدرتها على تعلم العلاقات المعقدة.",
    "data_leakage": "عزل مجموعة الاختبار تماماً وتطبيق كل خطوات المعالجة المسبقة على التدريب فقط هو خط الدفاع الأول لمنع تسريب البيانات.",
    "tokenization": "التقطيع بالوحدات الفرعية يساعد في تمثيل كلمات غير موجودة كوحدات كاملة في القاموس؛ الكفاءة وعدد الرموز يعتمدان على النص والـ Tokenizer.",
    "quantization_nf4_int8": "التكميم يقلل ذاكرة الأوزان، لكن أثره على جودة الإجابات والسرعة وإجمالي الذاكرة يجب قياسه على المهمة والعتاد المستخدمين."
}

def create_chat_turn(user_msg, assistant_msg, behavior, concept_id=None):
    return {
        "behavior": behavior,
        "concept_id": concept_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_NEUTRAL},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()}
        ]
    }

def build_v3_2_dataset():
    examples = []
    
    # -----------------------------------------------------------------
    # Part 1: Concepts KB with natural, non-combinatorial distribution
    # -----------------------------------------------------------------
    concept_list = list(ALL_CONCEPTS.items())
    
    # Diverse concept-specific intros for troubleshooting to eliminate repeated prefixes
    def format_tb_intro(c, idx):
        name_en = c["name_en"]
        name_ar = c["name_ar"]
        variant = idx % 5
        if variant == 0:
            return f"لتشخيص وعلاج هذه المشكلة في تطبيق {name_en}:"
        elif variant == 1:
            return f"للتعامل مع هذا العَرَض وإصلاح خلل {name_ar}:"
        elif variant == 2:
            return f"إليك الخطوات العملية الموصى بها لحل مشكلة {name_en}:"
        elif variant == 3:
            return f"لمعالجة هذا الخلل واستعادة استقرار عمل {name_en}:"
        else:
            return f"لتصحيح هذا السلوك في نموذجك وضبط {name_ar} بدقة:"
    
    # Diverse connectors for analogies
    def format_analogy(c, idx):
        name_en = c["name_en"]
        name_ar = c["name_ar"]
        sdef = c["short_def"]
        ana = c["analogy"]
        variant = idx % 4
        if variant == 0:
            return f"{ana}\n\nوجه الشبه هنا أن {name_en} يقوم على فكرة مطابقة: {sdef}"
        elif variant == 1:
            return f"{ana}\n\nفي سياق الذكاء الاصطناعي، هذا يماثل وظيفة {name_en} تماماً؛ حيث {sdef}"
        elif variant == 2:
            return f"{ana}\n\nتطبيقاً على {name_en} ({name_ar}): الآلية التقنية تعمل بنفس المنطق عبر {sdef}"
        else:
            return f"{ana}\n\nهذا التشبيه يوضح الفكرة لأن {name_en} يهدف أساساً إلى {sdef}"

    for idx, (cid, c) in enumerate(concept_list):
        name_ar = c["name_ar"]
        name_en = c["name_en"]
        
        # A) Short / Direct Constraint Following (strictly 1-2 lines)
        short_q_templates = [
            f"عرف {name_en} في سطرين مباشرين.",
            f"ما هو {name_en} باختصار شديد في جملة أو جملتين؟",
            f"عرف {name_ar} باختصار دون تفصيل في سطرين.",
            f"ما المقصود بـ {name_en} بعبارة موجزة في سطرين؟"
        ]
        q_short = short_q_templates[idx % len(short_q_templates)]
        ans_short = f"{name_en} ({name_ar}): {c['short_def']}"
        examples.append(create_chat_turn(q_short, ans_short, "short_direct", cid))
        
        # B) Explanation depth: distributed naturally (not all 3 for all 40)
        # We assign beginner to ~25 concepts, intermediate to ~30 concepts, detailed to ~25 concepts
        if idx % 3 != 0:  # 26 concepts get beginner
            beg_q_templates = [
                f"أنا مبتدئ تماماً في البرمجة والذكاء الاصطناعي، كيف تشرح لي {name_en} بأسلوب مبسط جداً؟",
                f"اشرح لي مفهوم {name_en} كأنني طفل في العاشرة دون أي معادلات معقدة.",
                f"كيف تشرح {name_ar} لشخص ليس لديه أي خلفية تقنية؟",
                f"بسط لي فكرة {name_en} بدون مصطلحات صعبة."
            ]
            q_beg = beg_q_templates[idx % len(beg_q_templates)]
            examples.append(create_chat_turn(q_beg, c["beginner"], "beginner", cid))
            
        if idx % 4 != 1:  # 30 concepts get intermediate
            int_q_templates = [
                f"ما هو مفهوم {name_en} وكيف يعمل بشكل عام؟",
                f"اشرح لي تقنية {name_ar} ({name_en}) وأهميتها.",
                f"وضح وظيفة ومفهوم {name_en} في الذكاء الاصطناعي."
            ]
            q_int = int_q_templates[idx % len(int_q_templates)]
            examples.append(create_chat_turn(q_int, c["intermediate"], "intermediate", cid))
            
        if idx % 3 != 2:  # 26 concepts get detailed
            det_q_templates = [
                f"اشرح لي الأساس الرياضي والتقني لمفهوم {name_en} بتفصيل معمق.",
                f"قدم تحليلاً تقنياً متقدماً لكيفية عمل {name_en}.",
                f"ما هي التفاصيل الهندسية والحسابية خلف {name_en}؟"
            ]
            q_det = det_q_templates[idx % len(det_q_templates)]
            examples.append(create_chat_turn(q_det, c["detailed"], "detailed", cid))
            
        # C) Analogy (30 concepts)
        if idx % 4 != 3 and "analogy" in c:
            ana_q_templates = [
                f"أعطني تشبيهاً واقعياً من الحياة اليومية لشرح كيف يعمل {name_en}.",
                f"اشرح لي فكرة {name_ar} باستخدام تشبيه محسوس وبسيط.",
                f"هل هناك تشبيه مجازي يقرب مفهوم {name_en} للذهن؟"
            ]
            q_ana = ana_q_templates[idx % len(ana_q_templates)]
            ans_ana = format_analogy(c, idx)
            examples.append(create_chat_turn(q_ana, ans_ana, "analogy", cid))
            
        # D) Practical Applied Example (35 concepts)
        if idx % 8 != 7 and "practical" in c:
            prac_q_templates = [
                f"وضح بمثال تطبيقي واقعي كيف يمكن لشركة استخدام {name_en} في الإنتاج.",
                f"أعطني مثالاً عملياً لكيفية تطبيق {name_ar} في الأنظمة الواقعية.",
                f"كيف تستفيد التطبيقات التجارية والخدمية من {name_en} عملياً؟"
            ]
            q_prac = prac_q_templates[idx % len(prac_q_templates)]
            examples.append(create_chat_turn(q_prac, c["practical"], "practical_example", cid))
            
        # E) Misconception Correction (explicit rejection first)
        if "misconception" in c:
            misc = c["misconception"]
            misc_q_templates = [
                f"هل صحيح أن {misc['claim']}؟",
                f"يعتقد زميلي أن {misc['claim']}، ما رأيك؟",
                f"ألا تعتقد أن {misc['claim']}؟"
            ]
            q_misc = misc_q_templates[idx % len(misc_q_templates)]
            examples.append(create_chat_turn(q_misc, misc["refutation"], "misconception", cid))
            
        # F) Troubleshooting (with varied natural intros)
        if "troubleshooting" in c:
            tb = c["troubleshooting"]
            tb_q_templates = [
                f"أواجه مشكلة في مشروعي: {tb['symptom']}، ما التشخيص وكيف أصلح ذلك؟",
                f"لاحظت أثناء العمل على {name_en} أن {tb['symptom']}، كيف أشخص الخلل وأعالجه؟",
                f"النموذج يعاني من العَرَض التالي: {tb['symptom']}، ما هي خطوات الحل المقترحة؟"
            ]
            q_tb = tb_q_templates[idx % len(tb_q_templates)]
            intro = format_tb_intro(c, idx)
            ans_tb = f"{intro}\n" + "\n".join(f"- {step}" for step in tb["steps"])
            examples.append(create_chat_turn(q_tb, ans_tb, "troubleshooting", cid))
            
        # G) Comparison (where defined)
        if "comparison_with" in c:
            comp = c["comparison_with"]
            other_id = comp["concept"]
            other_name = ALL_CONCEPTS.get(other_id, {}).get("name_en", other_id)
            comp_q_templates = [
                f"ما الفرق الجوهري بين {name_en} و {other_name} ومتى نفضل كلاً منهما؟",
                f"قارن بين {name_ar} ({name_en}) و {other_name}.",
                f"كيف أفرق بين {name_en} و {other_name} من حيث الاستخدام والهدف؟"
            ]
            q_comp = comp_q_templates[idx % len(comp_q_templates)]
            examples.append(create_chat_turn(q_comp, comp["contrast"], "comparison", cid))
            
        # H) Explicitly Requested Structured Output (ONLY for the 15 selected concepts with custom summaries)
        if cid in CUSTOM_STRUCTURED_SUMMARIES:
            summary = CUSTOM_STRUCTURED_SUMMARIES[cid]
            struct_q_templates = [
                f"اشرح لي تقنية {name_en} بالتفصيل المنظم مع توضيح: التعريف المختصر، الشرح المبسط، المثال العملي، والخلاصة.",
                f"قدم شرحاً مهيكلاً لمفهوم {name_ar} مقسماً إلى الأقسام التالية: تعريف مختصر، شرح مبسط، مثال عملي، الخلاصة.",
                f"اشرح {name_en} بهيكل كامل يشمل: التعريف، الشرح، المثال، والخلاصة."
            ]
            q_struct = struct_q_templates[idx % len(struct_q_templates)]
            ans_struct = (
                f"**تعريف مختصر:**\n{c['short_def']}\n\n"
                f"**شرح مبسط:**\n{c['intermediate']}\n\n"
                f"**مثال عملي:**\n{c['practical']}\n\n"
                f"**الخلاصة:**\n{summary}"
            )
            examples.append(create_chat_turn(q_struct, ans_struct, "explicit_structured", cid))

    # -----------------------------------------------------------------
    # Part 2: Explicit Behavioral Contrasts (60 examples across 12 concepts)
    # -----------------------------------------------------------------
    for row in BEHAVIORAL_CONTRASTS:
        examples.append(create_chat_turn(row["user"], row["assistant"], row["behavior"], row["concept_id"]))

    # -----------------------------------------------------------------
    # Part 3: Vague Clarifications (25 examples)
    # -----------------------------------------------------------------
    for row in VAGUE_CLARIFICATIONS:
        examples.append(create_chat_turn(row["user"], row["assistant"], "vague_clarification"))

    # -----------------------------------------------------------------
    # Part 4: Uncertainty & Scope Limits (12 examples)
    # -----------------------------------------------------------------
    for row in UNCERTAINTY_AND_SCOPE:
        examples.append(create_chat_turn(row["user"], row["assistant"], "uncertainty_scope"))

    # -----------------------------------------------------------------
    # Part 5: Follow-Up Scenarios (12 examples)
    # -----------------------------------------------------------------
    for row in FOLLOW_UPS:
        examples.append(create_chat_turn(row["user"], row["assistant"], "follow_up"))

    # -----------------------------------------------------------------
    # Part 6: Developer Code Implementations (15 examples)
    # -----------------------------------------------------------------
    for row in CODE_SNIPPETS:
        examples.append(create_chat_turn(row["user"], row["assistant"], "code_implementation"))

    # -----------------------------------------------------------------
    # Part 7: Engineering Debugging (12 examples)
    # -----------------------------------------------------------------
    for row in ENGINEERING_DEBUGGING:
        examples.append(create_chat_turn(row["user"], row["assistant"], "troubleshooting"))

    # -----------------------------------------------------------------
    # Part 8: Strict Single Sentence Constraints (25 examples)
    # -----------------------------------------------------------------
    for row in STRICT_ONE_SENTENCE:
        examples.append(create_chat_turn(row["user"], row["assistant"], "short_direct"))

    # -----------------------------------------------------------------
    # Part 9: Pedagogical Decision Tradeoffs (18 examples)
    # -----------------------------------------------------------------
    for row in DECISION_TRADEOFFS:
        examples.append(create_chat_turn(row["user"], row["assistant"], "comparison"))

    # -----------------------------------------------------------------
    # Part 10: Negative Constraints & Style Guidance (15 examples)
    # -----------------------------------------------------------------
    for row in NEGATIVE_CONSTRAINTS:
        examples.append(create_chat_turn(row["user"], row["assistant"], "short_direct"))

    # -----------------------------------------------------------------
    # Part 11: Student Pitfalls & Guidance (18 examples)
    # -----------------------------------------------------------------
    for row in STUDENT_PITFALLS:
        examples.append(create_chat_turn(row["user"], row["assistant"], "misconception"))

    return examples

def main():
    random.seed(42)
    examples = build_v3_2_dataset()
    print(f"\nGenerated {len(examples)} total candidate examples for Dataset V3.2.")
    
    # Save full dataset
    out_dir = ROOT / "data/v3_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "clean_dataset_v3_2.jsonl"
    
    with full_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    print(f"Full V3.2 dataset saved to: {full_path}")

if __name__ == "__main__":
    main()
