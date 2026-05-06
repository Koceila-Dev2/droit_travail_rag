import re
import json
import pickle

def split_text_with_overlap(text, max_words=2500, overlap=300):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    sub_chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        sub_chunks.append(" ".join(words[i:i + max_words]))
    return sub_chunks

def chunk_code_travail(md_file_path):
    chunks = []
    current_context = {
        "partie": "",
        "livre": "",
        "titre_chapitre": ""
    }
    
    current_article_title = ""
    current_article_content = []

    with open(md_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            if line.startswith("# "):
                current_context["partie"] = line.replace("# ", "")
            elif line.startswith("## "):
                current_context["livre"] = line.replace("## ", "")
            elif line.startswith("### "):
                current_context["titre_chapitre"] = line.replace("### ", "")
            
            # détection du début d'un article
            elif line.startswith("#### Article"):
                #on sauvegarde le précédent article si il existe : 
                if current_article_title:
                    context_header = f"{current_context['partie']} > {current_context['livre']} > {current_context['titre_chapitre']}\n{current_article_title}\n"
                    full_content = " ".join(current_article_content)
                    
                    sub_contents = split_text_with_overlap(full_content, max_words=2500, overlap=300)
                    for i, sub_content in enumerate(sub_contents):
                        chunks.append({
                            "metadata": {**current_context, "article": current_article_title, "chunk_id": i, "total_chunks": len(sub_contents)},
                            "content": context_header + sub_content
                        })
                
                current_article_title = line.replace("#### ", "")
                current_article_content = []
            
            else:
                if current_article_title:
                    current_article_content.append(line)

        #sauvegrad dudernier article après la fin du fichier           
        if current_article_title:
            context_header = f"{current_context['partie']} > {current_context['livre']} > {current_context['titre_chapitre']}\n{current_article_title}\n"
            full_content = " ".join(current_article_content)
            
            sub_contents = split_text_with_overlap(full_content, max_words=2500, overlap=300)
            for i, sub_content in enumerate(sub_contents):
                chunks.append({
                    "metadata": {**current_context, "article": current_article_title, "chunk_id": i, "total_chunks": len(sub_contents)},
                    "content": context_header + sub_content
                })

    return chunks

if __name__ == "__main__":
    chunks = chunk_code_travail("./data/code_du_travail_rag.md")

    with open("./data/chuncks.pckl", "wb") as f:
        pickle.dump(chunks, f)
        print("Chunks saved to chunks.pckl")


    print(f"Nombre d'articles extraits : {len(chunks)}")
    if chunks:
        print(f"Exemple :\n{chunks[500]['content']}")