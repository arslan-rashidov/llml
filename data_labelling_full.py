import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from datetime import datetime
import locale
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import warnings

def get_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def save_preds(preds, path, p):
    path = path.split('.')[0] + f'_part_{p}.' + path.split('.')[1]
    with open(path, 'w+') as fp:
        for label in preds:
            fp.write(f"{label}\n")

def get_prompt_for_mistral(classes, with_rag=False):
    prompt_start = "<s>[INST] Ты профессиональный разметчик данных. Тебе нужно размечать тексты которые я присылаю. Я буду присылать Текст, а ты будешь присылать мне подходящую Метку. "
    prompt_label_info = ""
    for class_name, class_label in classes.items():
        prompt_label_part_info = f", напиши метку '{class_label}', если этот текст относится к классу '{class_name}'"
        prompt_label_info += prompt_label_part_info
    prompt_label_info += '. '
    prompt_label_info = "Н" + prompt_label_info[3:]
    prompt_rag_part = "Ты можешь использовать следующие примеры размеченных текстов. Изучи их схожесть и используй их как подсказку, если тексты похожи, и только если они помогут тебе правильно определить метку: {context}. "
    prompt_label_model_answer = "[/INST]Хорошо. Я"
    for class_name, class_label in classes.items():
        prompt_label_model_answer_part = f" буду присылать Метку '{class_label}' если посчитаю что Текст относится к классу '{class_name}',"
        prompt_label_model_answer += prompt_label_model_answer_part
    prompt_label_model_answer = prompt_label_model_answer[:-1] + '.'
    prompt_end = '</s> [INST] Текст:"""Привет. Как дела?""". Метка: [/INST]0</s> [INST] Текст: """{question}""". Метка: [/INST]'
    if with_rag:
        prompt = prompt_start + prompt_label_info + prompt_rag_part + prompt_label_model_answer + prompt_end
        return prompt
    else:
        prompt = prompt_start + prompt_label_info + prompt_label_model_answer + prompt_end
        return prompt

def get_qa_chain(llm, prompt, with_rag=False, rag_texts_path=None, k=None):
    qa_chain = None
    if with_rag:
        loader = CSVLoader(file_path=rag_texts_path,
                           encoding='utf-8')

        data = loader.load()

        modelPath = "intfloat/e5-large-unsupervised"
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False})

        # Using faiss index
        db = FAISS.from_documents(data, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=db.as_retriever(k=k),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    else:
        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            return_final_only=True
        )
    return qa_chain


def label_data(classes, texts_path, target_column, save_preds_path, part_start=None, with_rag=False, rag_texts_path=None, k=None):
    print(cfg)

    locale.getpreferredencoding = lambda: "UTF-8"

    warnings.filterwarnings("ignore")

    model_name = "ybelkada/Mistral-7B-v0.1-bf16-sharded"
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)


    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        # temperature=0.2,
        # repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1,

    )

    text_generation_pipeline.tokenizer.pad_token_id = text_generation_pipeline.tokenizer.pad_token_id

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt_template = get_prompt_for_mistral(classes, with_rag)
    prompt = PromptTemplate.from_template(prompt_template)

    if with_rag:
        qa_chain = get_qa_chain(llm,
                                prompt,
                                with_rag=True,
                                rag_texts_path=rag_texts_path,
                                k=k)
    else:
        qa_chain = get_qa_chain(llm,
                                prompt,
                                with_rag=False)

    texts = list(pd.read_csv(texts_path)[target_column].values)

    batch_start_time = datetime.now()
    preds = []

    if part_start is None:
        part_start = 0
    classes_labels = set([str(el) for el in list(classes.values())])
    for p in range(part_start, len(texts)):
        text_to_label = str(texts[p])[:3700]
        if with_rag:
            input_text = {
                "query": text_to_label,
            }
        else:
            input_text = {
                "question": text_to_label,
            }

        init_text_len = len(text_to_label)
        text_to_label_decrease_step = 1
        while True:
            try:
                response = qa_chain(input_text)
                break
            except Exception as e:
                input_text[list(input_text.keys())[0]] = text_to_label[:-100 * text_to_label_decrease_step]
                text_to_label_decrease_step += 1
                print(f'error found: {str(e)}. text_len = {len(input_text[list(input_text.keys())[0]])}/{init_text_len}')
                continue
        if with_rag:
            label = response['result']
        else:
            label = response['text']
        if label not in classes_labels:
            label = 'unpredicted'
        preds.append(label)
        if len(str(label).strip()) == 0:
            print(response)
        if ((p + 1) % 100 == 0 and p != 0):
            print(f"{str(datetime(year=1, month=1, day=1,hour=(datetime.now().hour + 3) % 24, minute=datetime.now().minute, second=datetime.now().second).time()).split('.')[0]}.{p} done. Batch Time = {datetime.now() - batch_start_time}")
            part_i = (p+1) // 100
            save_preds(preds, save_preds_path, part_i)
            preds = []

            batch_start_time = datetime.now()
        elif p == len(texts) - 1:
            print(
                f"{str(datetime(year=1, month=1, day=1, hour=(datetime.now().hour + 3) % 24, minute=datetime.now().minute, second=datetime.now().second).time()).split('.')[0]}.{p} done. Batch Time = {datetime.now() - batch_start_time}")
            part_i = (p // 100) + 1
            save_preds(preds, save_preds_path, part_i)

            batch_start_time = datetime.now()


import sys

with open(sys.argv[1]) as json_file:
    cfg = json.load(json_file)

label_data(**cfg)