#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import os
import csv
import warnings



import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import numpy as np
from logic_engine import LogicEngine
from tkinter import Tk, filedialog
from image_model.task_c_inference import classify_image
from image_model.task_c_cloud import cloud_classify_image, compare_local_and_cloud



# Silence noisy warnings for demo output (safe; doesn't affect results)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore", message=".*tqdm.*IProgress.*")
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlink.*")
warnings.filterwarnings("ignore", message=".*embeddings.position_ids.*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*You are sending requests to the HF API.*")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
NLTK_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)



# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles=os.path.join(PROJECT_ROOT, "kb", "chatbot.aiml"))

# ---  safety guardrail: avoid nonsense matches using a food/ingredient whitelist ---
# ---  safety guardrail: refuse obvious non-food / nonsense queries ---
GENERIC = {
    "is","are","does","do","contain","contains","free","allergy","safe","can","i","eat",
    "with","without","for","a","an","the","to","of","in","on","my","your"
    }

NONFOOD_BLACKLIST = {
    "plastic","air","water","moon","laptop","phone","car","metal","rock","table"
    }



# ---------------------------
# TASK A: Load CSV Knowledge Base
# ---------------------------
KB_PATH = os.path.join(PROJECT_ROOT, "kb", "allergy_qa_kb.csv")

kb_questions = []
kb_answers = []

try:
    with open(KB_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a:
                kb_questions.append(q)
                kb_answers.append(a)

    print("KB LOADED SUCCESSFULLY")
    print("Number of questions:", len(kb_questions))
except FileNotFoundError:
    print("ERROR: Could not find", KB_PATH)
    print("Make sure allergy_qa_kb.csv is inside the kb folder.")
except KeyError:
    print("ERROR: CSV must have headers: question,answer")

# ---------------------------
# TASK A: Preprocessing + TF-IDF setup
# ---------------------------
_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    """
    Convert text -> normalised string for TF-IDF:
    - lowercase
    - tokenize
    - remove non-alphabetic tokens
    - remove stopwords
    - lemmatise
    """
    text = text.lower().strip()
    tokens = word_tokenize(text)
    cleaned = []
    for t in tokens:
        if t.isalpha() and t not in _stopwords:
            cleaned.append(_lemmatizer.lemmatize(t))
    return " ".join(cleaned)


# ---------------------------
# Task B: Normalise logic for kb
# ---------------------------
def normalise_logic_entity(entity_text: str) -> str:
    """
    Convert Task B entity text into a safe KB symbol.

    Examples:
    'rice milk'   -> 'rice_milk'
    'rice_milk'   -> 'rice_milk'
    'rice milk is' -> 'rice_milk'
    """
    entity = entity_text.strip().lower()

    # Defensive fix for AIML greedy capture
    if entity.endswith(" is"):
        entity = entity[:-3].strip()

    if entity.endswith("_is"):
        entity = entity[:-3].strip()

    entity = entity.replace("-", "_")
    entity = entity.replace(" ", "_")

    return entity


def normalise_logic_property(prop_text: str) -> str:
    """
    Map user-friendly Task B property text to exact KB predicate names.
    """
    prop = prop_text.strip().lower().replace("-", " ").replace("_", " ")

    mapping = {
        "dairy": "Dairy",
        "gluten": "Gluten",
        "egg": "Egg",
        "eggs": "Egg",
        "nut": "Nut",
        "nuts": "Nut",
        "vegan": "Vegan",
        "sesame": "Sesame",
        "seed": "Seed",

        "dairy free": "DairyFree",
        "egg free": "EggFree",
        "gluten free": "GlutenFree",
        "nut free": "NutFree",
        "sesame free": "SesameFree",

        "not dairy free": "NotDairyFree",
        "not egg free": "NotEggFree",
        "not gluten free": "NotGlutenFree",
        "not nut free": "NotNutFree",
        "not sesame free": "NotSesameFree",

        "unsafe for milk allergy": "UnsafeForMilkAllergy",
        "unsafe for coeliac": "UnsafeForCoeliac",
        "unsafe for egg allergy": "UnsafeForEggAllergy",
        "unsafe for nut allergy": "UnsafeForNutAllergy",
        "unsafe for sesame allergy": "UnsafeForSesameAllergy",
    }

    return mapping.get(prop, prop_text.strip().replace(" ", ""))









# Preprocess all KB questions once
kb_questions_proc = [preprocess(q) for q in kb_questions]

# Build TF-IDF model ONCE (first-class: don’t rebuild every user message)
vectorizer = TfidfVectorizer()
kb_tfidf = vectorizer.fit_transform(kb_questions_proc)

# Similarity threshold (tune later)
SIM_THRESHOLD = 0.35




# ---------------------------
# TASK A: Extension: SBERT semantic similarity (embeddings)
# ---------------------------

USE_SBERT = True          # set True for extension mode demo
SBERT_THRESHOLD = 0.55    # tune using your eval set
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

if USE_SBERT:
    print("Loading SBERT model...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # Precompute embeddings for KB questions once (fast at runtime)
    print("Encoding KB questions with SBERT...")
    kb_sbert = sbert_model.encode(
        kb_questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    print("SBERT: ready.")


# ---------------------------
# Task B: Logic KB setup
# ---------------------------
TASK_B_KB_PATH = os.path.join(PROJECT_ROOT, "kb", "task_b_kb.csv")

logic_engine = LogicEngine()

try:
    logic_engine.load_kb_from_csv(TASK_B_KB_PATH)
    kb_ok, kb_msg = logic_engine.kb_is_consistent()

    if kb_ok:
        print("TASK B LOGIC KB LOADED SUCCESSFULLY")
        print("Task B KB statements:", len(logic_engine.kb))
    else:
        print("ERROR: Task B KB is inconsistent.")
        print(kb_msg)

except FileNotFoundError:
    print("ERROR: Could not find", TASK_B_KB_PATH)
    print("Make sure task_b_kb.csv is inside the kb folder.")
except Exception as e:
    print("ERROR loading Task B KB:", e)

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")



# ---------------------------
# Task C
# ---------------------------
def choose_image_file() -> str:
    """
    Open a file dialog for the user to choose an image.
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an image for Task C classification",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path




# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    
    # ---------------------------
    # Step 6.4: Mode toggle for demo (type: MODE TFIDF or MODE SBERT)
    # ---------------------------
    if userInput.strip().upper() == "MODE TFIDF":
        USE_SBERT = False
        print("Mode set to TF-IDF baseline.")
        continue

    if userInput.strip().upper() == "MODE SBERT":
        USE_SBERT = True
        print("Mode set to SBERT extension.")
        continue

    

    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        
        
        elif cmd == 41:
            # Task B: I KNOW THAT <thing> IS <property>
            thing = normalise_logic_entity(params[1])
            prop = normalise_logic_property(params[2])

            stmt = f"{prop}({thing})"

            try:
                added, msg = logic_engine.add_statement_if_consistent(stmt)

                if added:
                    print(f"OK. {msg}")
                else:
                    print(f"Sorry. {msg}")

            except Exception as e:
                print("Sorry, I could not process that statement.")
                print(f"[Task B debug] {e}")

        elif cmd == 42:
            # Task B: CHECK THAT <thing> IS <property>
            thing = normalise_logic_entity(params[1])
            prop = normalise_logic_property(params[2])

            stmt = f"{prop}({thing})"

            try:
                result = logic_engine.check_statement(stmt)

                if result.status == "CORRECT":
                    print("Correct.")
                    print(result.explanation)
                elif result.status == "INCORRECT":
                    print("Incorrect.")
                    print(result.explanation)
                else:
                    print("Sorry, I don't know.")
                    print(result.explanation)

            except Exception as e:
                print("Sorry, I could not process that check.")
                print(f"[Task B debug] {e}")
                
                
                
        elif cmd == 51:
            # Task C: image classification
            image_path = choose_image_file()

            if not image_path:
                print("No image was selected.")
            else:
                try:
                    predicted_label, confidence = classify_image(image_path)

                    safe_response = (
                        f"This looks like {predicted_label.replace('_', ' ')} "
                        f"with confidence {confidence:.2%}. "
                        f"Please note that image classification does not guarantee the exact ingredients, "
                        f"so always check labels or recipes if allergies are important."
                    )

                    print(safe_response)

                except Exception as e:
                    print("Sorry, I could not classify that image.")
                    print(f"[Task C debug] {e}")               
                



        elif cmd == 52:
            # Task C extension: cloud-based image classification
            image_path = choose_image_file()

            if not image_path:
                print("No image was selected.")
            else:
                try:
                    predicted_label, confidence, raw_labels = cloud_classify_image(image_path)

                    if predicted_label == "unknown":
                        print(
                            "Google Cloud Vision returned labels for the image, "
                            "but they did not map clearly onto the coursework food classes."
                        )
                        print("Top raw cloud labels:")
                        for label, score in raw_labels[:5]:
                            print(f" - {label}: {score:.2%}")
                    else:
                        print(
                            f"Google Cloud Vision thinks this looks like "
                            f"{predicted_label.replace('_', ' ')} "
                            f"with confidence {confidence:.2%}. "
                            f"This is a cloud-based second opinion and does not guarantee exact ingredients."
                        )

                except Exception as e:
                    print("Sorry, I could not classify that image with Google Cloud Vision.")
                    print(f"[Task C cloud debug] {e}")

        elif cmd == 53:
            # Task C extension: compare local model with cloud API
            image_path = choose_image_file()

            if not image_path:
                print("No image was selected.")
            else:
                try:
                    result = compare_local_and_cloud(image_path)

                    print(
                        f"Local model: {result['local_label'].replace('_', ' ')} "
                        f"({result['local_confidence']:.2%})"
                    )
                    print(
                        f"Google Cloud Vision: {result['cloud_label'].replace('_', ' ')} "
                        f"({result['cloud_confidence']:.2%})"
                    )
                    print("Top raw cloud labels:")
                    for label, score in result["raw_cloud_labels"]:
                        print(f" - {label}: {score:.2%}")

                except Exception as e:
                    print("Sorry, I could not compare the local and cloud models.")
                    print(f"[Task C compare debug] {e}")



                
                
                
                
        
        elif cmd == 99:
            # params[1] should contain the user query passed from AIML: #99$<text>
            query = params[1] if len(params) > 1 else userInput

            query_proc = preprocess(query)
            query_vec = vectorizer.transform([query_proc])


            sims = cosine_similarity(query_vec, kb_tfidf).flatten()
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
            
            # ---------------------------
            # Step 6.5: SBERT similarity (extension)
            # ---------------------------
            if USE_SBERT:
                q_emb = sbert_model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
                sbert_sims = np.dot(kb_sbert, q_emb)  # cosine because embeddings are normalized
                sbert_best_idx = int(np.argmax(sbert_sims))
                sbert_best_score = float(sbert_sims[sbert_best_idx])

            # Choose method (baseline vs extension)
            if USE_SBERT:
                chosen_idx = sbert_best_idx
                chosen_score = sbert_best_score
                chosen_threshold = SBERT_THRESHOLD
            else:
                chosen_idx = best_idx
                chosen_score = best_score
                chosen_threshold = SIM_THRESHOLD

            # Keep your existing safety guardrail but apply it to chosen_idx/chosen_score
            query_tokens = set(preprocess(query).split())
            match_tokens = set(preprocess(kb_questions[chosen_idx]).split())

            meaningful_query = query_tokens - GENERIC
            meaningful_match = match_tokens - GENERIC
            meaningful_overlap = meaningful_query & meaningful_match

            contains_nonfood = len(meaningful_query & NONFOOD_BLACKLIST) >= 1

            if chosen_score >= chosen_threshold and len(meaningful_overlap) >= 1 and not contains_nonfood:
                print(kb_answers[chosen_idx])
            else:
                print("I’m not confident about that. Could you rephrase using the ingredient/food name? Also check the product label to be safe.")


            
            
            
            
        else:
            print("I did not get that, please try again.")

    else:
        print(answer)
