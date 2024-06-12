import ast

from fastapi import FastAPI, UploadFile
from io import BytesIO
import pandas as pd

from src.scripts.wrapper import LLMLWrapper

app = FastAPI()

texts_path = 'texts.csv'
rag_texts_path = 'rag_texts.csv'

llml_wrapper = LLMLWrapper()


@app.post("/llm/label_data")
def upload(classes: str, texts: UploadFile, target_column: str):
    classes = process_classes(classes)
    if type(classes) != dict:
        return False

    process_csv(texts)

    preds = llml_wrapper.label_data(classes, texts_path, target_column)

    return preds


@app.post("/llm_rag/label_data")
def upload(classes: str, texts: UploadFile, target_column: str, rag_texts: UploadFile, k: int):
    classes = process_classes(classes)
    if type(classes) != dict:
        return False

    process_csv(texts)
    process_csv(rag_texts, True)

    preds = llml_wrapper.label_data(classes, texts_path, target_column, True, rag_texts_path, k)

    return preds


def process_csv(file, rag=False):
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)

    file_path = texts_path
    if rag:
        file_path = rag_texts_path

    df.to_csv(file_path, index=False)
    buffer.close()
    file.file.close()

def process_classes(classes):
    try:
        classes = ast.literal_eval(classes)
        return classes
    except:
        return classes
