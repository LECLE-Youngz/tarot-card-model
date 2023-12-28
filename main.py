import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

model_id = "danurahul/gptneo_tarot"

tokenizer = AutoTokenizer.from_pretrained(
    model_id
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10
)
hf = HuggingFacePipeline(pipeline=pipe)

card = [
    {
        "name": "The Fool",
        "meaning": "Beginnings, innocence, spontaneity, a free spirit"
    },
    {
        "name": "The Magician",
        "meaning": "Power, skill, concentration, action, resourcefulness"
    },
    {
        "name": "The High Priestess",
        "meaning": "Intuition, Higher powers, mystery, subconscious mind"
    }
]

template = """
Generate standardized messages following the below instructions explaining the task.

### Instructions:
You are a Tarot Reader who will receive three tarot cards, then combine the meanings of the cards to make today predictions following a specific topic as requested.

### Card:
Below are three tarot cards and their meanings by the format `name: meanings`. You can use the meanings to make a prediction:
1. {card0_name}: {card0_meaning}
2. {card1_name}: {card1_meaning}
3. {card2_name}: {card2_meaning}


### Input:
Below is a request to your prediction, answer this question with a prediction based on your knowledge and analysis:
{input}

### Response:
Please write a response that appropriately completes the request:
"""
prompt = PromptTemplate.from_template(template)
chain = prompt | hf

print("Generating output...")
input = "how will my love go on in near future?"
print(chain.invoke({
    "input": input, 
    "card0_name": card[0]["name"], "card0_meaning": card[0]["meaning"],
    "card1_name": card[1]["name"], "card1_meaning": card[1]["meaning"],
    "card2_name": card[2]["name"], "card2_meaning": card[2]["meaning"]
}))