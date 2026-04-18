import os
import json
import re
from groq import Groq
from typing import Dict, List, Optional
from src.core.utils import getAllDicts
from src.core.model_loader import model_manager

# ------------------------------
# 1. Tài KG dictionaries
# ------------------------------
all_dicts = getAllDicts('wikidata_big')
ent2id = all_dicts['ent2id']
rel2id = all_dicts['rel2id']
ts2id = all_dicts['ts2id']
id2ent = all_dicts['id2ent']
id2ts = all_dicts['id2ts']
wd_id_to_text = all_dicts['wd_id_to_text']

# ------------------------------
# 2. Kh i t o Groq client
# ------------------------------
# Set API key directly
GROQ_API_KEY = "..."
client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# 3. Hàm trích xu t thành ph n t câu h i (dùng LLM)
# ------------------------------
def extract_components(question: str) -> Dict:
    prompt = f"""You are an expert in temporal knowledge graphs. Extract the following components from the question:
- head: the main subject entity (as a string, exactly as it appears in Wikidata if possible)
- relation: the predicate (e.g., "president", "born", "found")
- time: the year (if any) as a string
- answer_type: either "entity" or "time" (what the question asks for)

Return ONLY a JSON object. Example:
Question: Who was the president of the United States in 2020?
Output: {{"head": "United States", "relation": "president", "time": "2020", "answer_type": "entity"}}

Question: {question}
Output:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ------------------------------
# 4. Ánh x a entity name  ID (fuzzy matching)
# ------------------------------
def resolve_entity(name: str) -> int:
    if not name:
        return 0
    if name in ent2id:
        return ent2id[name]
    name_lower = name.lower()
    for e, eid in ent2id.items():
        if e.lower() == name_lower:
            return eid
    # Tìm trong wd_id_to_text (các tên th c t)
    for qid, text in wd_id_to_text.items():
        if text.lower() == name_lower:
            # qid có d ng 'Q...'
            if qid in ent2id:
                return ent2id[qid]
    return 0

def resolve_relation(name: str) -> int:
    if not name:
        return 0
    if name in rel2id:
        return rel2id[name]
    # Th tìm trong wd_id_to_text (các relation có th có tên)
    for rid, rname in wd_id_to_text.items():
        if rname.lower() == name.lower():
            if rid in rel2id:
                return rel2id[rid]
    return 0

def resolve_time(year_str: str) -> int:
    if not year_str:
        return 0
    try:
        year = int(year_str)
    except:
        return 0
    for (y,_,_), tid in ts2id.items():
        if y == year:
            return tid
    return 0

# ------------------------------
# 5. T o c u trúc câu h i gi ng dataset
# ------------------------------
def build_question_dict(question_text: str, components: Dict) -> Dict:
    head_name = components.get("head", "")
    relation_name = components.get("relation", "")
    time_str = components.get("time", "")
    answer_type = components.get("answer_type", "entity")
    
    head_id = resolve_entity(head_name)
    time_id = resolve_time(time_str)
    relation_id = resolve_relation(relation_name)
    
    # T o entities list (ch head, có th thêm tail n u có)
    entities = [head_name] if head_name else []
    times = [time_str] if time_str else []
    relations = [relation_name] if relation_name else []
    
    # T o template (don gi n)
    if answer_type == "entity":
        template = f"What is the {relation_name} of {head_name}?" + (f" in {time_str}?" if time_str else "")
    else:
        template = f"When did {head_name} {relation_name}?" + (f" in {time_str}?" if time_str else "")
    
    annotation = {
        "head": head_name,
        "time": time_str,
        "relation": relation_name
    }
    
    return {
        "question": question_text,
        "entities": entities,
        "times": times,
        "relations": relations,
        "answer_type": answer_type,
        "template": template,
        "annotation": annotation,
        "paraphrases": [question_text],
        "type": "simple_entity" if answer_type == "entity" else "simple_time",
        "answers": []  # s  i n sau khi predict
    }

# ------------------------------
# 6. Hàm chuyển ID sang tên entity
# ------------------------------
def id_to_entity_name(entity_id: int) -> str:
    """Chuyển entity ID sang tên thực thể từ wd_id_to_text"""
    if entity_id in id2ent:
        ent_wd = id2ent[entity_id]
        return wd_id_to_text.get(ent_wd, f"Entity_{entity_id}")
    return f"Entity_{entity_id}"

# ------------------------------
# 7. D u o n b i m hình TempoQR (t n d ng model_manager)
# ------------------------------
def predict_with_tempoqr(question_dict: Dict) -> List[str]:
    """
    Dùng model_manager.predict  l y top-k câu tr i.
    Tr v list các string (sử dụng id_to_entity_name để chuyển ID sang tên).
    """
    try:
        answers = model_manager.predict(question_dict["question"], k=5)
        # Chuyển các ID trong answers sang tên thực thể
        named_answers = []
        for ans in answers:
            if isinstance(ans, int):
                named_answers.append(id_to_entity_name(ans))
            else:
                named_answers.append(str(ans))
        return named_answers
    except Exception as e:
        print(f"L i khi i TempoQR: {e}")
        return []

# ------------------------------
# 7. Hàm chính: Agent x lý t u u n cu i
# ------------------------------
def answer_question(question: str) -> Dict:
    print(f"\n C u h i: {question}")
    # B c 1: Trích xu t thành ph n
    comp = extract_components(question)
    print(f" Thành ph n trích xu t: {comp}")
    
    # B c 2: T o c u trúc dataset
    qdict = build_question_dict(question, comp)
    print(f" C u trúc câu h i:\n{json.dumps(qdict, indent=2, ensure_ascii=False)}")
    
    # B c 3: D u o n
    predictions = predict_with_tempoqr(qdict)
    print(f" D u o n top-5: {predictions}")
    
    # B c 4: Tr v k t qu
    return {
        "question": question,
        "structure": qdict,
        "predictions": predictions
    }

# ------------------------------
# 8. Ch y th
# ------------------------------
if __name__ == "__main__":
    # API key is set directly in the script
    print(" GROQ_API_KEY is set directly in the script")
    
    test_question = "Who was the president of the United States in 2020?"
    result = answer_question(test_question)
    print("\n K t qu cu i cùng:")
    print(f"   C u h i: {result['question']}")
    print(f"   D u o n: {result['predictions']}")
