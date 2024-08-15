import json

def convert_json_file(input_file, output_file):
    output_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f, start=1):
            data = json.loads(line)
            prompt = data.get("question", "")
            answer = data.get("right_answer", "")
            output_data.append({"index": index, "prompt": prompt, "answer": answer})

    with open(output_file, 'w+', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

input_file = "./HaluEval/data/qa_data.json"  
output_file = "./dataset_train/HaluEval_qa_data.json"  
convert_json_file(input_file, output_file)