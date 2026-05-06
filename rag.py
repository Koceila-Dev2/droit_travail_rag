from groq import Groq
from dotenv import load_dotenv
import json
import os
from vector_db import VectorDB
from config import LLM_MODEL_NAME,EMBEDDING_MODEL_NAME



class RAG:
    def __init__(self, vector_db_name):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.vector_db_object = VectorDB(vector_db_name)


    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()



    def build_context(self, question):
        context = RAG.read_file(file_path="context.txt")

        chuncks = self.vector_db_object.retrieve(question, n=3)[0]

        full_context = context.replace("{{Chuncks}}", str(chuncks))
        
        return full_context




    def answer_question(self, question):

        chat_completion = self.client.chat.completions.create(
            messages=[

                {
                    "role": "system",
                    "content": self.build_context(question),
                },

                {
                    "role": "user",
                    "content": question,
                }
            ],

            model=LLM_MODEL_NAME
        )

        return chat_completion.choices[0].message.content


if __name__ == "__main__":
    rag_object = RAG(vector_db_name="droits_travail_embeddings_db")
    response = rag_object.answer_question(question="quels sont les droits d'un salarié en cas de licenciement ?")
    print(response)
