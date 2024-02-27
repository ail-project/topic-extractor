import json
import PyPDF2
from typing import Union
from tqdm import tqdm
import csv


def load_txt(document_name: str) -> Union[str, None]:
    """
    From a txt, load its content and return it as a string
    """
    try:
        with open(document_name, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"'{document_name}' not found")
        return None
    except Exception as e:
        print(f"Error happened : {e}")
        return None


def load_jsonl(document_name: str) -> Union[str, None]:
    """
    From a json, load its content and return it as a string
    """
    donnees_json = []
    try:
        with open(document_name, "r") as file:
            for ligne in tqdm(file):
                donnees = json.loads(ligne)
                donnees_json.append(donnees)
            return " ".join(ele for ele in donnees_json)
    except FileNotFoundError:
        print(f"'{document_name}' not found")
        return None
    except Exception as e:
        print(f"Error happened : {e}")
        return None


def load_json(document_name: str) -> Union[str, None]:
    """
    From a json, load its content and return it as a string
    """
    try:
        with open(document_name, "r") as file:
            content = json.load(file)
            return content
    except FileNotFoundError:
        print(f"'{document_name}' not found")
        return None
    except Exception as e:
        print(f"Error happened : {e}")
        return None


def load_pdf(document_name: str) -> Union[str, None]:
    """
    From a pdf, load its content and return it as a string
    """
    try:
        with open(document_name, "rb") as file:
            lct = PyPDF2.PdfReader(file)
            contenu = ""
            for page_num in range(len(lct.pages)):
                page = lct.pages[page_num]
                contenu += page.extract_text()
            return contenu

    except FileNotFoundError:
        print(f"'{document_name}' not found")
        return None
    except Exception as e:
        print(f"Error happened : {e}")
        return None


def load_csv(document_name: str) -> Union[str, None]:
    """
    From a csv, load its content and return a string
    """
    data = ""
    try:
        with open(document_name, newline="") as csvfile:
            reader = csv.reader(csvfile)
            # reader = csv.DictReader(csvfile)
            # headers = reader.fieldnames
            # print(f"I found these headers : {headers}")
            for row in reader:
                data += ",".join(row) + "\n"

    except FileNotFoundError:
        print(f"'{document_name}' not found")
        return None
    except Exception as e:
        print(f"Error happened : {e}")
        return None

    return data


def load_document(document_name: str) -> str:
    """
    Load content from a document (path).
    Currently supported: txt, json, jsonl, pdf & csv
    """
    content = None
    print(f"Loading document {document_name}")
    if document_name.endswith(".txt"):
        content = load_txt(document_name=document_name)
    elif document_name.endswith(".json"):
        content = load_json(document_name=document_name)
    elif document_name.endswith(".jsonl"):
        content = load_jsonl(document_name=document_name)
    elif document_name.endswith(".pdf"):
        content = load_pdf(document_name=document_name)
    elif document_name.endswith(".csv"):
        content = load_csv(document_name=document_name)
    else:
        print(f"Document format not supported.")
        return

    if content is None:
        print(f"Error: no loaded content detected")
        return
    else:
        print(f"Content loaded.")
        return content
