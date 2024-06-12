import locale
import warnings
from datetime import datetime

import transformers

from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

import pandas as pd

from src.scripts.configs.e5_embeddings import embedddings_model_name
from src.scripts.configs.mistral import model_name

from src.scripts.model import get_model
from src.scripts.prompt import get_prompt_for_mistral
from src.scripts.tokenizer import get_tokenizer

locale.getpreferredencoding = lambda: "UTF-8"

warnings.filterwarnings("ignore")


class LLMLWrapper:
    def __init__(self):
        model = get_model(model_name)
        tokenizer = get_tokenizer(model_name)

        text_generation_pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            return_full_text=False,
            max_new_tokens=1
        )

        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    def setup_qa_chain(self, classes, with_rag=False, rag_texts_path=None, k=None):
        prompt_template = get_prompt_for_mistral(classes, with_rag)
        prompt = PromptTemplate.from_template(prompt_template)

        if with_rag:
            loader = CSVLoader(file_path=rag_texts_path,
                               encoding='utf-8')

            data = loader.load()

            embeddings = HuggingFaceEmbeddings(
                model_name=embedddings_model_name,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': False})

            db = FAISS.from_documents(data, embeddings)

            self.qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=db.as_retriever(k=k),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
        else:
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                return_final_only=True
            )

    def label_data(self, classes, texts_path, target_column, with_rag=False, rag_texts_path=None, k=None):
        self.setup_qa_chain(classes, with_rag=with_rag, rag_texts_path=rag_texts_path, k=k)

        texts = list(pd.read_csv(texts_path)[target_column].values)

        batch_start_time = datetime.now()
        preds = []

        classes_labels = set([str(el) for el in list(classes.values())])

        for p in range(len(texts)):
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
                    response = self.qa_chain(input_text)
                    break
                except Exception as e:
                    input_text[list(input_text.keys())[0]] = text_to_label[:-100 * text_to_label_decrease_step]
                    text_to_label_decrease_step += 1
                    print(
                        f'error found: {str(e)}. text_len = {len(input_text[list(input_text.keys())[0]])}/{init_text_len}')
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
                print(
                    f"{str(datetime(year=1, month=1, day=1, hour=(datetime.now().hour + 3) % 24, minute=datetime.now().minute, second=datetime.now().second).time()).split('.')[0]}.{p} done. Batch Time = {datetime.now() - batch_start_time}")

                batch_start_time = datetime.now()

        return preds
