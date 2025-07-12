from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Union

class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB using either OpenAI or HuggingFace embeddings.

    Parameters:
        data_directory (str or List[str]): Directory or list of paths to PDF documents.
        persist_directory (str): Directory where the vector DB will be saved.
        embedding_model_engine (str): Embedding engine identifier (e.g., "text-embedding-ada-002" or HuggingFace model name).
        chunk_size (int): Size of each document chunk.
        chunk_overlap (int): Overlap between document chunks.
    """

    def __init__(
        self,
        data_directory: Union[str, List[str]],
        persist_directory: str,
        embedding_model_engine: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> None:
        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding = self._load_embedding_model()

    def _load_embedding_model(self):
        """
        Load embedding model dynamically based on engine name.
        Returns:
            Embedding object (OpenAI or HuggingFace)
        """
        if self.embedding_model_engine.startswith("text-embedding"):
            print("Using OpenAI Embeddings")
            return OpenAIEmbeddings()
        else:
            model_name = os.getenv("EMBEDDING_MODEL", self.embedding_model_engine)
            print(f"Using HuggingFace Embeddings: {model_name}")
            return HuggingFaceEmbeddings(model_name=model_name)

    def __load_all_documents(self) -> List:
        """
        Load all PDF documents from directory or list of paths.
        Returns:
            List of Document objects
        """
        doc_counter = 0
        docs = []

        if isinstance(self.data_directory, list):
            print("Loading documents from list of files...")
            for doc_path in self.data_directory:
                docs.extend(PyPDFLoader(doc_path).load())
                doc_counter += 1
        else:
            print("Loading documents from directory...")
            document_list = os.listdir(self.data_directory)
            for doc_name in document_list:
                doc_path = os.path.join(self.data_directory, doc_name)
                if doc_path.endswith(".pdf"):
                    docs.extend(PyPDFLoader(doc_path).load())
                    doc_counter += 1

        print(f"Loaded {doc_counter} document(s), {len(docs)} total pages\n")
        return docs

    def __chunk_documents(self, docs: List) -> List:
        """
        Split loaded documents into smaller chunks.
        Returns:
            List of chunked documents.
        """
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunked_documents)} chunks\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        """
        Load, chunk, embed, and save documents into Chroma vector store.
        Returns:
            Chroma DB object
        """
        docs = self.__load_all_documents()
        chunked_documents = self.__chunk_documents(docs)

        print("Saving documents to Chroma VectorDB...")
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        print("VectorDB saved.")
        print(f"Vector count: {vectordb._collection.count()}\n")
        return vectordb
