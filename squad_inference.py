import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import nltk
from pathlib import Path

model_path = "fine_tuned_breast_cancer_model"  # Replace with the path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Download the NLTK data for identifying prepositions
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Specify the path to your local NLTK data directory
nltk_data_dir = Path("data")

# Load NLTK data from the local directory
nltk.data.path.append(str(nltk_data_dir))

# Prepare the question and context (passage) for inference
question = input("Enter your question: ")

# Function to remove prepositions from the question
def remove_prepositions(text):
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    return " ".join(token for token, pos in tagged_tokens if pos not in ['IN', 'TO', 'PRP', 'PRP$', 'VBZ', 'WRB'])

question = remove_prepositions(question)

context = """Breast cancer risk factors include age, family history, genetic mutations,
hormonal factors, and lifestyle choices. Women over the age of 50 are at higher risk.
Having a first-degree relative with breast cancer increases the risk.
BRCA1 and BRCA2 gene mutations are associated with a higher likelihood of breast cancer.
Exposure to estrogen for an extended period can also increase the risk.
adopting a healthy lifestyle, including regular exercise and a balanced diet,
can help reduce the risk of breast cancer."""

# Tokenize the question and context
inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Perform inference to get the answer
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get the most probable start and end positions for the answer
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Check if the answer is available in the context
    if start_index < end_index:
        # Get the answer span from the context
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index + 1]))
        print("Question:", question)
        print("Answer:", answer)
    else:
        print("Question out of context.")