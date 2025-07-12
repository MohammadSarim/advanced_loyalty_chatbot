import gradio as gr
import time
import openai
import os
from langchain.vectorstores import Chroma
from typing import List, Tuple
import re
import ast
import html
from utils.load_config import LoadConfig

APPCFG = LoadConfig()
URL = "https://github.com/MohammadSarim/advanced_loyalty_chatbot"
hyperlink = f"[RAG-GPT user guideline]({URL})"


from openai import OpenAI

class ChatBot:
    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed doc", temperature: float = 0.0) -> Tuple:
        if data_type == "Preprocessed doc":
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(
                    persist_directory=APPCFG.persist_directory,
                    embedding_function=APPCFG.embedding_model
                )
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module. For further information please visit {hyperlink}.")
                )
                return "", chatbot, None

        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(
                    persist_directory=APPCFG.custom_persist_directory,
                    embedding_function=APPCFG.embedding_model
                )
            else:
                chatbot.append(
                    (message, f"No file was uploaded. Please first upload your files using the 'upload' button.")
                )
                return "", chatbot, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)
        print(docs)

        question = "# User new question:\n" + message
        retrieved_content = ChatBot.clean_references(docs)
        chat_history = f"Chat history:\n {str(chatbot[-APPCFG.number_of_q_a_pairs:])}\n\n"
        prompt = f"{chat_history}{retrieved_content}{question}"

        print("========================")
        print(prompt)

        # ✅ Initialize client (for Groq API which mimics OpenAI)
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_API_BASE")
        )

        # ✅ Call LLM using correct method
        response = client.chat.completions.create(
            model=APPCFG.llm_engine,
            messages=[
                {"role": "system", "content": APPCFG.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )

        chatbot.append((message, response.choices[0].message.content))
        time.sleep(2)

        return "", chatbot, retrieved_content

    @staticmethod
    def clean_references(documents: List) -> str:
        server_url = "http://localhost:8000"
        documents = [str(x) + "\n\n" for x in documents]
        markdown_documents = ""
        counter = 1

        for doc in documents:
            match = re.search(r"page_content=(.*?)( metadata=\{.*\})", doc, re.DOTALL)
            if not match:
                print(f"[WARNING] Skipping unmatched document:\n{doc}")
                continue

            content, metadata = match.groups()
            metadata = metadata.split('=', 1)[1]
            metadata_dict = ast.literal_eval(metadata)

            content = bytes(content, "utf-8").decode("unicode_escape")
            content = re.sub(r'\\n', '\n', content)
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            content = html.unescape(content)
            content = content.encode('latin1').decode('utf-8', 'ignore')

            content = re.sub(r'â', '-', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Ã', '×', content)
            content = re.sub(r'ï¬', 'fi', content)
            content = re.sub(r'ï¬', 'fl', content)
            content = re.sub(r'Â·', '·', content)

            pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"
            markdown_documents += f"# Retrieved content {counter}:\n" + content + "\n\n" + \
                f"Source: {os.path.basename(metadata_dict['source'])}" + " | " + \
                f"Page number: {str(metadata_dict['page'])}" + " | " + \
                f"[View PDF]({pdf_url})" "\n\n"
            counter += 1

        return markdown_documents
