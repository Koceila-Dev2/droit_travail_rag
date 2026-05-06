from mistralai.client import Mistral
from dotenv import load_dotenv
import pickle
import os

load_dotenv()

mistral_api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=mistral_api_key)

def get_embeddings(texts):
    # Mistral permet d'envoyer des listes de textes
    # embeddings_batch_response = client.embeddings(
    #     model="mistral-embed",
    #     input=texts
    # )
    


    with Mistral(
        api_key=os.environ["MISTRAL_API_KEY"],
    ) as mistral:

        res = mistral.embeddings.create(model="mistral-embed", inputs=texts)
        return [obj.embedding for obj in res.data]

    

if __name__ == "__main__":
    # Exemple : on embedde les 10 premiers chunks
    with open("./data/chunks.pckl", "rb") as f:
        chunks = pickle.load(f)

    sample_texts = [c["content"] for c in chunks[:10]]
    vectors = get_embeddings(sample_texts)
    print(f"Embedding du premier chunk : {vectors[0]}")