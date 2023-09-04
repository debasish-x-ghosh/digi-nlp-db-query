from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd

data = pd.read_csv(r"data.csv")
print(data)
data = data.astype(str)

# Load model & tokenizer
model = 'google/tapas-base-finetuned-wtq'
tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model)
tapas_tokenizer = AutoTokenizer.from_pretrained(model)

# Initializing pipeline
nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)


def qa(query,data):
    print('>>>>>')
    print(query)
    result = nlp({'table': data,'query':query})
    answer = result['cells']
    print(answer)

prediction = qa('Which sri lankan batsman has average below 35',data)
print(prediction)


prediction = qa('who is the third highest scorer?',data)
print(prediction)


