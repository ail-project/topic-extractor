#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import gensim
import preprocessing
import wordcloud
import matplotlib.pyplot as plt
import plotly.express as px
import utils
import tqdm
import pandas
import pyLDAvis.gensim

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic


class Lda:
    def __init__(self, raw_document: str) -> None:
        self.raw_document = raw_document

    # def preprocess_doc_into_string(self) -> None:
    #     """
    #     Launch the preprocessing and set the result as an attribut [string]
    #     """
    #     self.preprocessed_doc = preprocessing.cleaned(str_document=self.raw_document)

    def preprocess_doc_into_list(self, line_break: bool = False) -> None:
        """
        Launche the preprocssing (based on a string) and set the results an attribut [list]
        """
        self.preprocessed_doc = preprocessing.default_text_preprocessing(
            str_document_text=self.raw_document, line_break=line_break
        )  # list of str

    def make_corpus(self) -> None:
        """
        Make the corpus based on the Gensim Dictionnary of the preprocessed doc
        """
        print(f"Making the corpus..")

        if not self.dictionary:
            print(
                f"No dictionary detected. Please, make the dictionary first with .make_dictionary()"
            )
            return

        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.preprocessed_doc]

    def make_dictionary(self) -> None:
        """
        Make the dictionary using Gensim.
        Must be set up before make_corpus.
        """
        print(f"Making the dictionnary..")
        self.dictionary = gensim.corpora.Dictionary(self.preprocessed_doc)

    def vectorize_data(self):
        """ """
        print(f"Vectorizing the data")
        vectorizer = CountVectorizer(
            analyzer="word",
        )  # simple because data are enough preprocessed
        dt = []
        for list_sentence in self.preprocessed_doc:
            new_sent = str()
            for str_word in list_sentence:
                new_sent += f" {str_word}"
            dt.append(new_sent)

        self.data_vectorized = vectorizer.fit_transform(dt)

    def use_bertopic(self):
        """
        Using all-in-one Bertopic to do the job
        """
        print("Using BerTopic...")
        topic_model = BERTopic(
            language="multilingual", calculate_probabilities=True, verbose=True
        )
        dt = []
        for list_sentence in self.preprocessed_doc:
            new_sent = str()
            for str_word in list_sentence:
                new_sent += f" {str_word}"
            dt.append(new_sent)

        topics, probs = topic_model.fit_transform(dt)
        print(topics)
        freq = topic_model.get_topic_info()
        print(freq.head())
        topic_model.visualize_topics()
        topic_model.visualize_barchart(top_n_topics=5)

    def gridsearch(
        self, limit: int = 40, start: int = 2, step: int = 2, verbose: bool = True
    ):
        """
        Similar to compute_coherence_values, but used GridSearch library instead of coherence score manually.
        This needs to vectorize the data (done automatically here)
        Compute the best model, based on nbr of topic and learning decay of LDA Gensim algo
        """
        print(f"GridSearching the best possible model..")
        search_params = {
            "n_components": range(start, limit, step),
            "learning_decay": [0.5, 0.7, 0.9],
        }
        model = LatentDirichletAllocation(
            max_iter=5, learning_method="online", learning_offset=50.0, random_state=100
        )
        self.best_lda_model = GridSearchCV(
            estimator=model, param_grid=search_params, verbose=3, refit=True
        )
        self.vectorize_data()
        # Define Search Param

        print(f"Fitting the model with vectorized data..")
        self.best_lda_model.fit(self.data_vectorized)

        # Model Parameters
        print("Best Model's Params: ", self.best_lda_model.best_params_)
        # self.best_lda_model = self.best_lda_model.best_estimator_

        if verbose:
            model = gensim.models.ldamodel.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.best_lda_model.best_params_["n_components"],
                random_state=100,
                alpha="auto",
                per_word_topics=True,
            )
            self.visualize_topics(
                lda_model=model,
                corpus=self.corpus,
                dictionary=self.dictionary,
            )

    def compute_coherence_values(
        self, limit: int = 40, start: int = 2, step: int = 2, verbose: bool = True
    ):
        """
        Compute c_v and u_mass coherence for various number of topics and
        return the best lda model to use for the corpus.
        The c_v coherence score is based on the coherence measurement of topics. The higher the better (positive highest possible)
        The u_mass coherence score measures the similarity between words within topics. The closest to 0, the better.

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        print(
            f"Computing coherence values for different number of topics in order to determine the best one to use.."
        )
        results = {}
        cv_coherence = []
        umass_coherence = []
        for num_topics in tqdm.tqdm(range(start, limit, step)):
            model = gensim.models.ldamodel.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=100,
                alpha="auto",
                per_word_topics=True,
            )
            results[model] = []
            coherencemodel = gensim.models.CoherenceModel(
                model=model,
                texts=self.preprocessed_doc,
                dictionary=self.dictionary,
                coherence="u_mass",
            )
            results[model].append(coherencemodel.get_coherence())
            umass_coherence.append(coherencemodel.get_coherence())
            coherencemodel = gensim.models.CoherenceModel(
                model=model,
                texts=self.preprocessed_doc,
                dictionary=self.dictionary,
                coherence="c_v",
            )
            cv_coherence.append(coherencemodel.get_coherence())
            results[model].append(coherencemodel.get_coherence())

        top_3_cv = sorted(zip(cv_coherence, results.keys()), reverse=True)[:3]
        top_3_umass = sorted(zip(umass_coherence, results.keys()), reverse=True)[:3]
        chosen_one = {
            "model": top_3_umass[0][1],
            "u_mass": top_3_umass[0][0],
            "c_v": results[top_3_umass[0][1]][1],
        }

        for ele in top_3_umass:
            if ele[1] in [
                mod[1] for mod in top_3_cv
            ]:  # second element of the tuple, which is the model
                chosen_one = {
                    "model": ele[1],
                    "u_mass": ele[0],
                    "c_v": results[ele[1]][1],
                }

        self.best_lda_model = chosen_one["model"]

        if verbose:
            print(f"============= RESULTS ====================")
            for model, scores in results.items():
                print(f"{model}: u_mass {scores[0]} & c_v {scores[1]}")
            print(f"========================================")
            print(
                f"Top model : c_v {chosen_one['c_v']} & u_mass {chosen_one['u_mass']} & {chosen_one['model'].num_topics} topics"
            )
            df = pandas.DataFrame()
            df["number_of_topics"] = range(start, limit, step)
            df["c_v"] = cv_coherence
            df["u_mass"] = umass_coherence
            fig = px.line(data_frame=df, x=df.number_of_topics, y=[df.c_v, df.u_mass])
            fig.show()
            self._show_topics()
            self.world_cloud()
            self.visualize_topics(
                lda_model=self.best_lda_model,
                corpus=self.corpus,
                dictionary=self.dictionary,
            )

    def _show_topics(self):
        """
        Show the topic selected by our best lda model
        """
        model_topics = self.best_lda_model.show_topics(formatted=False)
        print("================= TOPIC KEYWORDS ========================")
        for ele in model_topics:
            topic_number = ele[0]
            keywords = []
            list_topic_keywords = ele[1]
            for tupl in list_topic_keywords:
                keywords.append(tupl[0])
            print(f"{topic_number} : {keywords}")
        print("=========================================================")

    def visualize_topics(self, lda_model, corpus, dictionary):
        """
        using Gensim to save topics visualization (in html)
        """
        # Visualize the topics
        p = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        name = "lda.html"
        pyLDAvis.save_html(p, name)
        print(f"Visualization saved to {name}")

    def world_cloud(self, num_words: int = 10):
        """
        A pretty cloud with <num_words> words in it in order to show
        top wods within topics.
        """
        for topic_id, topic in enumerate(
            self.best_lda_model.print_topics(
                num_topics=self.best_lda_model.num_topics, num_words=num_words
            )
        ):
            topic_words = " ".join(
                [word.split("*")[1].strip() for word in topic[1].split(" + ")]
            )
            wd = wordcloud.WordCloud(
                width=800, height=800, random_state=21, max_font_size=110
            ).generate(topic_words)
            plt.figure()
            plt.imshow(wd, interpolation="bilinear")
            plt.axis("off")
            plt.title("Topic: {}".format(topic_id))
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Topic extraction")
    parser.add_argument(
        "filename",
        help="Name/Path of the file to which we want to perform Topic Extraction",
    )
    parser.add_argument(
        "--method",
        choices=["gridsearch", "bertopic", "coherence"],
        help="name of the technic to use",
    )
    args = parser.parse_args()

    method = args.method
    # gridsearch by default
    if method is None:
        method = "gridsearch"

    if args.filename is None:
        print("Error: You must provide a filename.")
        exit(1)

    # Vérifie si le fichier existe
    if not os.path.exists(args.filename):
        print(f"Error: The file '{args.filename}' does not exist.")
        exit(1)

    raw_doc = utils.load_document(document_name=args.filename)
    # raw_doc = utils.load_document(document_name="static/en_dissertation_health.pdf")
    if not raw_doc:
        exit()
    lda = Lda(raw_document=raw_doc)
    lda.preprocess_doc_into_list()
    lda.make_dictionary()
    lda.make_corpus()
    if method == "gridsearch":
        print(f"Will use GridSearchCV method")
        lda.gridsearch(limit=30, start=3, step=2)
    elif method == "bertopic":
        print(f"Will use Bertopic automatic method")
        lda.use_bertopic()
    else:
        print(
            f"Will build the best LDA model by evaluating manually the coherence scores"
        )
        lda.compute_coherence_values(limit=30, start=3, step=2)
