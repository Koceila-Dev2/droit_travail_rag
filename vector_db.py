from asyncio import sleep

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
from mistralai.client import Mistral
from dotenv import load_dotenv
import pickle
import os

load_dotenv()

class VectorDB:
	def __init__(self, vector_db_name, chuncks=None, metadatas=None):
		self.vector_db_name = vector_db_name

		if os.path.exists(vector_db_name):
			self.load_vector_db(vector_db_name)
		
		elif chuncks :
			self.create_vector_db(vector_db_name, chuncks, metadatas)

		else:
			raise(Exception("Can't initiate vector db object ! please give a path to a vector db / chuncks."))



	def create_vector_db(self, vector_db_name, chuncks, metadatas):
		print("Création de ma base de donnée vectorielle")
		print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
		# self.sentence_transformer_object = SentenceTransformer(EMBEDDING_MODEL_NAME)

		
		self.chroma = chromadb.PersistentClient(path=vector_db_name)
		collection = self.chroma.get_or_create_collection(
										name=vector_db_name,
										metadata={**metadatas[0], 
        									"embedding_model": EMBEDDING_MODEL_NAME,
        									}
										)

		# la création des embeddings
		embeddings = self.get_embeddings(chuncks)

		# ChromaDB maximum batch size limit 5461 tokens
		chroma_batch_size = 5000 
		for i in range(0, len(chuncks), chroma_batch_size):
			batch_chuncks = chuncks[i:i+chroma_batch_size]
			batch_embeddings = embeddings[i:i+chroma_batch_size]
			batch_metadatas = metadatas[i:i+chroma_batch_size]
			batch_ids = [f"chunck_{id_chunck}" for id_chunck in range(i, i + len(batch_chuncks))]
			
			collection.add(
				ids=batch_ids,
				documents=batch_chuncks,
				embeddings=batch_embeddings,
				metadatas=batch_metadatas
			)



	def load_vector_db(self, vector_db_name):
		print("Chargement de ma base données vectorielle")
		self.chroma = chromadb.PersistentClient(path=vector_db_name)
		collection_info = self.chroma.get_collection(vector_db_name)
		# EMBEDDING_MODEL_NAME = collection_info.metadata["embedding_model"]
		# print(f"Embedding model : {EMBEDDING_MODEL_NAME}")




	def get_embeddings(self, chuncks):
		#pour embed la question : 
		if len(chuncks) == 1:
			with Mistral(api_key=os.environ["MISTRAL_API_KEY"]) as mistral:
				response = mistral.embeddings.create(model="mistral-embed", inputs=chuncks)
				return [obj.embedding for obj in response.data]

		if os.path.exists("embeddings_response_mistral.pckl"):
			print("Chargement des embeddings Mistral depuis le fichier pickle...")
			with open("embeddings_response_mistral.pckl", "rb") as f:
				response = pickle.load(f)
			return [obj.embedding for obj in response]

		with Mistral( api_key=os.environ["MISTRAL_API_KEY"] ) as mistral:

			response = []
			# Split chunks into batches of 40 to avoid "Too many tokens overall"
			batch_size = 40
			for i in range(0, len(chuncks), batch_size):
				batch = chuncks[i:i+batch_size]
				batch_response = mistral.embeddings.create(model="mistral-embed", inputs=batch)
				response.extend(batch_response.data)
				sleep(0.5)

			with open("embeddings_response_mistral.pckl", "wb") as f:
				pickle.dump(response, f)

			return [obj.embedding for obj in response]

		# embeddings = self.sentence_transformer_object.encode(
		# 	chuncks,
		# 	batch_size=64,
		# 	normalize_embeddings=True,
		# 	show_progress_bar=True
		# 	).tolist()
		# return embeddings


	def retrieve(self, question, n=3):
		embedded_question = self.get_embeddings([question])[0]

		collection = self.chroma.get_or_create_collection(self.vector_db_name)

		results = collection.query(query_embeddings=[embedded_question], n_results=n)

		return results["documents"], results["metadatas"]



if __name__ == "__main__":

	with open("./data/chuncks.pckl", "rb") as f:
		raw_chuncks = pickle.load(f)

	chuncks = [c["content"] for c in raw_chuncks]
	metadatas = [c["metadata"] for c in raw_chuncks]

	vector_db_object = VectorDB(vector_db_name="droits_travail_embeddings_db", chuncks=chuncks, metadatas=metadatas)






