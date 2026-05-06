import streamlit as st
from rag import RAG

# Phase 2 : Configuration de la page
st.set_page_config(
    page_title="Assistant Code du Travail",
    page_icon="⚖️",
    layout="centered"
)

# Sidebar avec les informations et disclaimers
with st.sidebar:
    st.title("⚖️ Assistant Légal RAG")
    st.markdown("""
    Cet assistant utilise l'Intelligence Artificielle pour répondre à vos questions sur le **Code du Travail français**.
    
    Il s'appuie sur une technique de **RAG** (Retrieval-Augmented Generation) pour chercher le contexte pertinent dans une base de données légale avant de vous répondre.
    
    ⚠️ **Avertissement important :** 
    _Les réponses générées par cette Intelligence Artificielle sont fournies à titre purement informatif. Elles peuvent contenir des approximations et ne remplacent en aucun cas la consultation d'un avocat spécialisé ou d'un expert juridique._
    """)
    st.divider()
    st.caption("Propulsé par Groq et ChromaDB")

# Titre principal de l'application
st.title("🤖 Chatbot - Droit du Travail")
st.write("Posez vos questions sur le code du travail et obtenez des réponses basées sur les textes de loi.")

# Phase 3 : Instanciation du modèle RAG (Mise en cache)
# Le décorateur @st.cache_resource permet de ne charger la base Chroma et l'API qu'une seule fois
@st.cache_resource
def load_rag():
    return RAG(vector_db_name="droits_travail_embeddings_db")

# Chargement de l'assistant
try:
    rag_assistant = load_rag()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'assistant RAG : {e}")
    st.stop()

# Phase 4 : Initialisation de l'historique des messages (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant en droit du travail. Quelle est votre question aujourd'hui ?"}
    ]

# Affichage des messages existants dans l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Phase 5 : Zone de saisie utilisateur et gestion du flux
if prompt := st.chat_input("Ex: Quels sont mes droits pour un congé paternité ?"):
    
    # 1. Ajout et affichage immédiat de la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Zone d'affichage pour la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les textes de loi et génération de la réponse..."):
            try:
                # 3. Appel de la logique backend RAG
                response = rag_assistant.answer_question(question=prompt)
                
                # 4. Affichage de la réponse et sauvegarde dans l'historique
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Une erreur est survenue lors de la communication avec l'assistant : {e}")
