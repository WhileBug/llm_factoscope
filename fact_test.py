from lib.FactJudger import query_huggingface
from lib.FactJudger import query_huggingface

response = query_huggingface(
    question="Who is current China's Leader",
    answer="Xi Jinping",
    prediction="Xi Jinping"
)
print(response)