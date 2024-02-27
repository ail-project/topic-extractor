from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import inflect
import unicodedata
import contractions
import re
import tqdm


def text_sentence_tokenizer(
    str_document_text: str, line_break: bool = False
) -> list[str]:
    """Method to split the string:document along sentences
    - input : a string (the document)
    - output : a list of string (the sentences)
    """
    if line_break:
        return str_document_text.split("\n")
    else:
        return tokenize.sent_tokenize(str_document_text)


def word_remove_numbers(list_words: list[str]) -> list[str]:
    """
    Method to remove all integer occurence
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_word_noint = list()
    for str_word in list_words:
        if not str_word.isdigit():
            list_word_noint.append(str_word)
    return list_word_noint


def sentence_remove_contraction(list_document_sentences: list[str]) -> list[str]:
    """
    Method to remove contraction in english language (eg : don't -> do not)
    - input : a list of string (the sentences)
    - output : a list of string (the sentences)
    """

    def replace_contraction(str_text) -> str:
        return contractions.fix(str_text)

    list_sentences = list()
    for str_sentence in list_document_sentences:
        try:
            str_nocontraction = replace_contraction(str_sentence)
        except IndexError:
            str_nocontraction = str_sentence
        list_sentences.append(str_nocontraction)
    return list_sentences


def sentence_word_tokenizer(list_document_sentence: list[str]) -> list[str]:
    """
    Method to split a sentence along its words.
    - input : a list of string (the sentences)
    - output : a list of lists of strings (the sentences as a list of words)
    """
    list_tokenized_sentences = list()
    for str_sentence in list_document_sentence:
        list_tokens = word_tokenize(str_sentence)
        list_tokenized_sentences.append(list_tokens)
    return list_tokenized_sentences


def word_normalize_ascii(list_words: list[str]) -> list[str]:
    """
    Method that normalize non ascii alphabet to ascii representation from a list of words
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_words_normalized = list()
    for str_word in list_words:
        str_word_normalized = (
            unicodedata.normalize("NFKD", str_word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        list_words_normalized.append(str_word_normalized)
    return list_words_normalized


def word_lowercase(list_words: list[str]) -> list[str]:
    """
    Method to convert all characters to lowercase from a list of words
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_words_lowercase = list()
    for str_word in list_words:
        str_word_lower = str_word.lower()
        list_words_lowercase.append(str_word_lower)
    return list_words_lowercase


def word_remove_punctuation(list_words: list[str]) -> list[str]:
    """
    Method to remove all punctuation from a list of words
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_words_nopunkt = list()
    for str_word in list_words:
        str_word_nopunkt = re.sub(r"[^\w\s]", "", str_word)
        str_word_nopunkt = str_word_nopunkt.replace("_", "")
        if str_word_nopunkt != "":
            list_words_nopunkt.append(str_word_nopunkt)

    return list_words_nopunkt


def word_replace_numbers(list_words: list[str]) -> list[str]:
    """
    Method to replace all integer occurence with its textual representation (eg : 8 -> eight)
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    inflect_engine = inflect.engine()
    list_word_noint = list()
    for str_word in list_words:
        if str_word.isdigit():
            try:
                str_word_noint = inflect_engine.number_to_words(str_word)
                list_word_noint.append(str_word_noint)
            except:
                pass
        else:
            list_word_noint.append(str_word)
    return list_word_noint


def word_remove_stopwords(list_words: list[str]) -> list[str]:
    """
    Method to remove very common words from a list of tokens
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_words_nostop = list()
    for str_word in list_words:
        if str_word not in stopwords.words("english"):
            list_words_nostop.append(str_word)
    return list_words_nostop


def word_lemma(list_words: list[str]) -> list[str]:
    """
    Method transform a noun, or a verb into its lemma
    A lemma in linguistic is an uninflected form of a word
    Example : inflect, inflects, inflected -> (lemma)inflect
    Example : dog, dogs -> (lemma)dog
    Example : good, better, best -> (lemma)good
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    lemmatizer = WordNetLemmatizer()
    list_lemma = list()
    for str_word in list_words:
        str_lemma = lemmatizer.lemmatize(str_word, pos="v")  # verb
        str_lemma = lemmatizer.lemmatize(str_word, pos="n")  # noun
        list_lemma.append(str_lemma)
    return list_lemma


def word_remove_numbers(list_words: list[str]) -> list[str]:
    """
    Method to remove all integer occurence
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_word_noint = list()
    for str_word in list_words:
        if not str_word.isdigit():
            list_word_noint.append(str_word)
    return list_word_noint


def words_normalization(list_words: list[str], lemma=True) -> list[str]:
    """
    Method that stream a list of words through all words transformation function
    - input : a list of string (the words of a sentence)
    - output : a list of string (the words of a sentence)
    """
    list_words_clean = word_normalize_ascii(list_words)
    list_words_clean = word_lowercase(list_words_clean)
    list_words_clean = word_remove_punctuation(list_words_clean)
    # list_words_clean = .word_replace_numbers(list_words_clean)
    list_words_clean = word_remove_stopwords(list_words_clean)
    if lemma:
        list_words_clean = word_lemma(list_words_clean)
    return list_words_clean


def document_normalization(list_sentences: list[str], lemma=True) -> list:
    """
    Method that stream a list of tokenized sentence through all words transformation function
    - input : a list of lists of string (the tokenized sentences)
    - output : a list of lists of string (the tokenized sentences)
    """
    list_normalized_sentence = list()
    print(f"Working on: {list_sentences[:5]}")
    for list_tokenized_sentence in tqdm.tqdm(list_sentences):
        list_words_clean = words_normalization(list_tokenized_sentence, lemma)
        list_normalized_sentence.append(
            list(filter(None, list_words_clean))
        )  # remove empty list
    return list_normalized_sentence


def default_text_preprocessing(
    str_document_text: str, line_break: bool = False
) -> list[list[str]]:
    """
    Method that stream a text document through all textual transformation
    - input : a string (the document)
    - output : a list of lists of string (the tokenized sentences of the document)

    raw text --> raw text splitted into list of sentences -->
    --> raw list of sentences without conctraction --> splitted into list of words -->
    --> each words get preprocessed --> full preprocessed text reassembled

    """
    print(f"Preprocessing raw document...")
    list_document_sentence = text_sentence_tokenizer(
        str_document_text, line_break=line_break
    )
    print(list_document_sentence[:2])
    print(f"Sentence tokenizer: Done.")
    list_document_sentence = sentence_remove_contraction(list_document_sentence)
    print(f"Removing contraction: Done.")
    list_document_sentence = sentence_word_tokenizer(list_document_sentence)
    print(f"Word tokenizer: Done.")
    list_document_sentence = document_normalization(list_document_sentence)
    print(f"Document normalization: Done.")
    return list_document_sentence


def string_cleaned(str_document: str) -> str:
    """
    Mother method that take as input an raw text (as string) and
    return the preprocessed text (as string)
    """
    str_document_cleaned = str()
    for list_sentence in default_text_preprocessing(str_document):
        for str_word in list_sentence:
            str_document_cleaned += " {}".format(str_word)
    return str_document_cleaned


# def preprocess_data_frame(df):
#     df_cleaned = [cleaned(df[i]) for i in tqdm(range(0, len(df)))]
#     df_cleaned = pd.DataFrame(df_cleaned)
#     print(f"Preprocessed dataset : {df_cleaned.head()}")
#     return df_cleaned


### preprocessing.py ends here
