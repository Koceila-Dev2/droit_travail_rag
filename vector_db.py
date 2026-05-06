from sentence_transformers import SentenceTransformer
import chromadb
from config import EMBEDDING_MODEL_NAME


import os

class VectorDB:
	def __init__(self, vector_db_name, chuncks=None):

		if os.path.exists(vector_db_name):
			self.load_vector_db(vector_db_name)
		
		elif chuncks :
			self.create_vector_db(vector_db_name, chuncks)

		else:
			raise(Exception("Can't initiate vector db object ! please give a path to a vector db / chuncks."))



	def create_vector_db(self, vector_db_name, chuncks):
		print("Création de ma base de donnée vectorielle")
		print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
		self.sentence_transformer_object = SentenceTransformer(EMBEDDING_MODEL_NAME)

		self.chroma = chromadb.PersistentClient(path=vector_db_name)
		collection = self.chroma.get_or_create_collection(
										name="knowledge",
										metadata={
        									"embedding_model": EMBEDDING_MODEL_NAME,
        									}
										)

		# la création des embeddings
		embeddings = self.get_embeddings(chuncks)

		collection.add(
				ids = [f"chunck_{id_chunck}" for id_chunck in range(len(chuncks))],
				documents=chuncks,
				embeddings=embeddings,
				metadatas=[{"source": "doc1.pdf"} for _ in range(len(chuncks))]
				)



	def load_vector_db(self, vector_db_name):
		print("Chargement de ma base données vectorielle")
		self.chroma = chromadb.PersistentClient(path=vector_db_name)
		collection_info = self.chroma.get_collection("knowledge")	
		EMBEDDING_MODEL_NAME = collection_info.metadata["embedding_model"]
		print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
		self.sentence_transformer_object = SentenceTransformer(EMBEDDING_MODEL_NAME)




	def get_embeddings(self, chuncks):
		embeddings = self.sentence_transformer_object.encode(
			chuncks,
			batch_size=64,
			normalize_embeddings=True,
			show_progress_bar=True
			).tolist()
		return embeddings






	def retrieve(self, question, n=3):
		embedded_question = self.get_embeddings([question])[0]

		collection = self.chroma.get_or_create_collection("knowledge")

		results = collection.query(query_embeddings=[embedded_question], n_results=n)

		return results["documents"], results["metadatas"]



if __name__ == "__main__":

	chuncks = [
		"Le chat bleu de bob s'appelle Henri",
		"La tomate grise de ma tortue s'appelle Fred",
		"Le chien rouge de marie s'appelle Max",
		"La carotte orange de mon lapin s'appelle Léo",
		"Le poisson violet de pierre s'appelle Nemo",
		"La patate jaune de ma souris s'appelle Tom",
		"Le canard rose de sophie s'appelle Donald",
		"L'aubergine verte de mon hamster s'appelle Remy",
		"Le renard blanc de jean s'appelle Rusty",
		"La banane marron de ma chèvre s'appelle George",
		"Le pigeon cyan de lucas s'appelle Sky",
		"La courgette dorée de mon cochon d'inde s'appelle Oscar",
		"Le lion turquoise de marie s'appelle Simba",
		"L'olive rose de ma poule s'appelle Bella",
		"Le serpent argent de thomas s'appelle Silva",
		"La courge bronze de mon escargot s'appelle Speedy",
		"L'aigle pourpre de jessica s'appelle Phoenix",
		"Le melon beige de ma fourmi s'appelle Tiny",
		"Le loup noir de andré s'appelle Shadow",
		"L'ananas écarlate de mon perroquet s'appelle Polly",
		"Le panda lime de caroline s'appelle Bamboo",
		"La betterave indigo de ma baleine s'appelle Moby",
		"Le tigre fuchsia de david s'appelle Stripes",
		"Le radis turquoise de mon cactus s'appelle Prickly",
		"L'ours saumon de émilie s'appelle Teddy",
		"La citrouille prune de ma chauve-souris s'appelle Vlad",
		"Le zèbre abricot de françois s'appelle Stripe",
		"L'asperge cobalt de mon homard s'appelle Crusher",
		"Le chamois jade de sophie s'appelle Buck",
		"La navette spatiale olive de mon astronaute s'appelle Buzz",
		"Le corbeau argent de grégoire s'appelle Quill",
		"La fraise émeraude de ma girafe s'appelle Spots",
		"L'hippopotame ambre de natalie s'appelle Hippo",
		"Le champignon rose clair de mon gnome s'appelle Toadstool",
		"Le papillon gris foncé de victor s'appelle Flutter",
		"La prune corail de mon dragon s'appelle Spike",
		"Le lièvre cendre de véronique s'appelle Hop",
		"Le crabe menthe de mon pirate s'appelle Pinchy",
		"L'orange sarcelle de marie-claire s'appelle Valencia",
		"Le requin nacre de mon plongeur s'appelle Jaws",
		"La myrtille rouille de paul s'appelle Berry",
		"L'éléphant gris bleu de mon cirque s'appelle Dumbo",
		"Le mûrier grenat de mon oiseau s'appelle Violet",
		"Le sapin chartreuse de noel s'appelle Evergreen",
		"Le renard couleur crème de mon forestier s'appelle Ruskin",
		"La noix muscade de mon écureuil s'appelle Nutty",
		"Le brocoli turquoise de mon légume animé s'appelle Broc",
		"L'épinard magenta de mon superhéros s'appelle Power",
		"Le chou-fleur blanc nacré de mon jardin s'appelle Floret",
		"La patate douce cordon bleu de mon chef s'appelle Yam",
		"Le kiwi vert menthe de mon explorateur s'appelle Kiwi",
		"L'artichaut violet profond de mon artiste s'appelle Vincent",
		"Le fenouil aurore de mon docteur s'appelle Anise",
		"La betterave rose bonbon de mon boulanger s'appelle Sugar",
		"Le poivron orange vif de mon cuisinier s'appelle Peppy",
		"Le concombre bleu ciel de mon nageur s'appelle Splash",
		"La laitue vert lime de mon lapin s'appelle Lettuce",
		"Le maïs jaune d'or de mon fermier s'appelle Corn",
		"La tomate rouge cerise de mon jardinier s'appelle Cherry",
		"Le pois vert forest de mon géant s'appelle Pea",
		"L'oignon blanc neige de mon chef s'appelle Onion",
		"L'ail blanc perle de mon vampire s'appelle Garlic",
		"Le gingembre brun cannelle de mon cuisinier s'appelle Spicy",
		"Le céleri vert anis de mon nutritionniste s'appelle Celery",
		"La roquette vert prairie de mon gourmet s'appelle Rocket",
		"L'épautre doré blé de mon boulanger s'appelle Grain",
		"Le riz blanc lait de mon meunier s'appelle Rice",
		"L'orge brun miel de mon brasseur s'appelle Barley",
		"Le seigle gris charbon de mon paysan s'appelle Rye",
		"L'avoine flocon beige de mon sportif s'appelle Oat",
		"Le millet jaune minuscule de mon oiseau s'appelle Milly",
		"Le quinoa blanc crème de ma nutritionniste s'appelle Quinny",
		"Le soja brun chocolat de mon moine s'appelle Soy",
		"Le lin marron moka de mon tisserand s'appelle Flax",
		"Le chanvre vert kaki de mon écologiste s'appelle Hemp",
		"Le pavot bleu nuit de mon peintre s'appelle Poppy",
		"Le safran doré orange de mon cuisinier s'appelle Saffron",
		"Le curcuma jaune intense de mon healer s'appelle Turmeric",
		"Le poivre noir charbon de mon épicier s'appelle Pepper",
		"Le cacao marron foncé de mon chocolatier s'appelle Cocoa",
		"Le café brun noir de mon barista s'appelle Mocha",
		"Le thé vert jade de mon maître s'appelle Green",
		"Le cacao blanc crème de mon pâtissier s'appelle White",
		"La vanille beige ambré de mon confiseur s'appelle Vanilla",
		"La cannelle brun roux de mon boulanger s'appelle Cinnamon",
		"La noix de muscade marron chaud de mon épicier s'appelle Nutmeg",
		"Le clou de girofle brun noir de mon chimiste s'appelle Clove",
		"L'anis étoilé bronze doré de mon alchimiste s'appelle Anise",
		"Le laurier vert sombre de mon cuisinier s'appelle Laurel",
		"Le thym vert délicat de mon herboriste s'appelle Thyme",
		"Le romarin vert intense de mon jardinier s'appelle Rosemary",
	]
	vector_db_object = VectorDB(vector_db_name="infinite_knowledge", chuncks=chuncks)

























	# print(retrieve("Quelle est la couleur du chat de Bob ?", sentence_transformer_object, collection))



