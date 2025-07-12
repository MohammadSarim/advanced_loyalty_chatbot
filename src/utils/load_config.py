import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from pyprojroot import here
import shutil
import openai
# LangChain embedding imports
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

load_dotenv()

class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine = app_config["llm_config"]["engine"]
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.max_token = app_config["llm_config"].get("max_token", 4096)

        # Directories
        self.data_directory = str(here(app_config["directories"]["data_directory"]))
        self.persist_directory = str(here(app_config["directories"]["persist_directory"]))
        self.custom_persist_directory = str(here(app_config["directories"]["custom_persist_directory"]))

        # Embeddings
        self.embedding_model_engine = app_config["embedding_model_config"]["engine"]
        self.embedding_model = self._load_embedding_model()

        # Retrieval configs
        self.k = app_config["retrieval_config"]["k"]
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        # Summarizer configs
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = app_config["summarizer_config"]["final_summarizer_llm_system_role"]

        # Memory
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

        # Load credentials if needed
        self._configure_llm_api()

        # Setup directories
        self.create_directory(self.persist_directory)
        self.remove_directory(self.custom_persist_directory)

    def _load_embedding_model(self):
        if self.embedding_model_engine.startswith("text-embedding"):
            return OpenAIEmbeddings()
        else:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbeddings(model_name=model_name)

    def _configure_llm_api(self):
        if self.llm_engine.startswith("gpt") or self.embedding_model_engine.startswith("text-embedding"):
            # OpenAI style
        
            openai.api_type = os.getenv("OPENAI_API_TYPE", "open_ai")
            openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            openai.api_version = os.getenv("OPENAI_API_VERSION", "")
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif self.llm_engine.startswith("llama") or "groq" in os.getenv("GROQ_API_BASE", ""):
            # Groq-style LLM
            os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
            os.environ["OPENAI_API_BASE"] = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
            os.environ["OPENAI_MODEL_NAME"] = os.getenv("GROQ_MODEL", "llama3-8b-8192")
            os.environ["OPENAI_API_TYPE"] = "open_ai"

    def create_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")

