import re
import json
import pickle

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
                    full_text = f"{current_context['partie']} > {current_context['livre']} > {current_context['titre_chapitre']}\n{current_article_title}\n" + " ".join(current_article_content)
                    chunks.append({
                        "metadata": {**current_context, "article": current_article_title},
                        "content": full_text
                    })
                
                current_article_title = line.replace("#### ", "")
                current_article_content = []
            
            else:
                if current_article_title:
                    current_article_content.append(line)

        #sauvegrad dudernier article après la fin du fichier           
        if current_article_title:
            full_text = f"{current_context['partie']} > {current_context['livre']} > {current_context['titre_chapitre']}\n{current_article_title}\n" + " ".join(current_article_content)
            chunks.append({
                "metadata": {**current_context, "article": current_article_title},
                "content": full_text
            })

    return chunks

if __name__ == "__main__":
    chunks = chunk_code_travail("code_du_travail_rag.md")

    with open("chunks.pckl", "wb") as f:
        pickle.dump(chunks, f)
        print("Chunks saved to chunks.pckl")


    print(f"Nombre d'articles extraits : {len(chunks)}")
    if chunks:
        print(f"Exemple :\n{chunks[500]['content']}")