from langchain.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
from openai import OpenAI
import os


class Summarizer:
    """
    A class for summarizing PDF documents using ChatGPT-compatible engines (like Groq).

    Methods:
        summarize_the_pdf: Summarizes the content of a PDF file using the LLM.
        get_llm_response: Retrieves the response from the LLM for a given prompt.
    """

    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int
    ) -> str:
        """
        Summarizes the content of a PDF file using a ChatGPT-compatible engine.

        Args:
            file_dir (str): Path to the PDF.
            max_final_token (int): Max tokens in final summary.
            token_threshold (int): Token reduction threshold.
            gpt_model (str): Model name (e.g., "llama3-8b-8192").
            temperature (float): Sampling temperature.
            summarizer_llm_system_role (str): Role prompt for each page.
            final_summarizer_llm_system_role (str): Role prompt for final summary.
            character_overlap (int): Overlap between page chunks.

        Returns:
            str: Final summary string.
        """
        docs = PyPDFLoader(file_dir).load()
        print(f"Document length: {len(docs)}")
        max_output_per_page = int(max_final_token / len(docs)) - token_threshold

        full_summary = ""
        print("Generating the summary...")

        if len(docs) > 1:
            for i in range(len(docs)):
                if i == 0:
                    prompt = docs[i].page_content + docs[i+1].page_content[:character_overlap]
                elif i < len(docs) - 1:
                    prompt = (
                        docs[i-1].page_content[-character_overlap:]
                        + docs[i].page_content
                        + docs[i+1].page_content[:character_overlap]
                    )
                else:
                    prompt = docs[i-1].page_content[-character_overlap:] + docs[i].page_content

                role_prompt = summarizer_llm_system_role.format(max_output_per_page)
                summary = Summarizer.get_llm_response(
                    gpt_model=gpt_model,
                    temperature=temperature,
                    llm_system_role=role_prompt,
                    prompt=prompt
                )
                print(f"Page {i+1} summarized.")
                full_summary += summary + "\n\n"
        else:
            full_summary = docs[0].page_content
            print("Single-page document processed.")

        print("\nFull summary token length:",
              count_num_tokens(full_summary, model=gpt_model))

        final_summary = Summarizer.get_llm_response(
            gpt_model=gpt_model,
            temperature=temperature,
            llm_system_role=final_summarizer_llm_system_role,
            prompt=full_summary
        )
        return final_summary

    @staticmethod
    def get_llm_response(gpt_model: str, temperature: float, llm_system_role: str, prompt: str) -> str:
        """
        Gets a response from the language model using OpenAI-compatible API (Groq).

        Args:
            gpt_model (str): Model name.
            temperature (float): Sampling temperature.
            llm_system_role (str): Role instruction for the LLM.
            prompt (str): User prompt/content.

        Returns:
            str: LLM-generated response.
        """
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_API_BASE") 
        )

        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
