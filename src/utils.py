# utils.py (bổ sung)

from keybert import KeyBERT

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config_loader import load_config

config = load_config()


tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["summary_model"])

def extract_summary(text, max_len=128):
    input_ids = tokenizer(
        "Tóm tắt: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids
    output = model.generate(input_ids, max_length=max_len, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary



# Dùng cùng model multilingual để đồng nhất embedding
kw_model = KeyBERT(model=config["model"]["keywords_model"])

def extract_keywords(text, top_k=10):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),    # lấy cụm 1-2 từ
        stop_words=None,                 # không lọc tiếng Việt
        top_n=top_k
    )
    # keywords là list [(từ, điểm)]
    return ", ".join([kw for kw, _ in keywords])
