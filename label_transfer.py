import json

input_file = '/mnt/bd/riskif-peiran/RiskIF/llm_factoscope/features/Llama-2-7b-hf/HaluEval_qa_dataset/store_data.json'
output_file = 'HaluEval_qa_results.json'

# 初始化存储结果的列表
questions = []
model_completions = []

# 逐行读取JSONL文件
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())  # 逐行加载JSON对象
        questions.append(item["input_text"])
        model_completions.append(item["output_answer"])

# 转换为所需的格式
result = {
    "question": questions,
    "model_completion": model_completions
}

# 将结果写入到输出文件中
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
