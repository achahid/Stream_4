

# pipreqs .
# This will create requirements.txt file at the current directory.

import streamlit_authenticator as stauth
import nltk
import streamlit_ext as ste
import nltk_download_utils
from googletrans import Translator
import pandas as pd
import time
import warnings
import datetime
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from deep_translator import GoogleTranslator
import sys
# import huggingface_hub.snapshot_download
from huggingface_hub import snapshot_download

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
import re
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer, util

#
import xlsxwriter
from io import BytesIO

import streamlit as st



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pd.options.mode.chained_assignment = None

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")



#### FUNCTIONS #####

def translate_to_english(text_list):
    # translator = Translator(service_urls=['translate.google.com'])
    translator = Translator(service_urls=['translate.google.com'],
                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)', proxies=None, timeout=None)
    translated_text = []
    count = 0
    for text in text_list:
        try:
            result = translator.translate(text, dest='en').text
            translated_text.append(result)
            count += 1
            if count % 1000 == 0:
                print('time to sleep 5 sec')
                time.sleep(5)
        except Exception as e:
            translated_text.append("Translation failed")
    return translated_text

# IMPORTANT FUNCTIONS:
def data_preprocessing(df):
    # Make all column to lower case.
    df.columns = map(str.lower, df.columns)
    if 'keyword' not in df.columns:
        # print('ERROR: PLEASE CHECK IF YOUR DATA CONTAINS keyword COLUMN')
        st.error('Please ensure that your data includes the column **KEYWORD**', icon="🚨")
        sys.exit(1)

    if 'id' not in df.columns:
        df['id'] = range(len(df))
        print('id is added to the data')

    if 'keyword_eng' not in df.columns:
        with st.spinner('**The keywords are in the process of being translated to ENGLISH. Please hold on ...** '):
            df = df[['id', 'keyword']].copy()
            df.dropna(inplace=True)
            # Adding 'digit-' prefix for the rows that contains digits only as GoogleTranslator can not
            # translate digits only.
            df["keyword"] = df["keyword"].apply(lambda x: 'digit-' + x if x.isdigit() else x)
            # print("The keywords are in the process of being translated to ENGLISH. Please hold on ... ")
            my_list = df["keyword"].to_list()
            df["Keyword_eng"] = translate_to_english(my_list)
            # df["keyword_eng"] = df["keyword"].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
            # remove the added prefix from the rows
            df["keyword_eng"] = df["keyword_eng"].apply(lambda x: x.replace("digit-", "") if x.startswith("digit-") else x)
            df["keyword"] = df["keyword"].apply(lambda x: x.replace("digit-", "") if x.startswith("digit-") else x)
        st.success('**The translation process is finished, we are now moving on to the clustering process.***')

    # Splits the data into short and long tail keywords:
    df_org = df.copy()
    df['keyword_eng'] = df['keyword_eng'].astype(str)
    df['keyword'] = df['keyword'].astype(str)

    df_new = df[['id', 'keyword', 'keyword_eng']].copy()
    # Remove the next words.
    keywords_quistions = ['what is', 'why is', 'what', 'why', 'how much', 'how long', 'how many',
                          'how do', 'how to', 'when', 'when is', 'where is']

    df_new = keyowrds_removal(df_new, keywords_quistions)

    df_new['strings_only'] = df_new['keyword'].str.replace('\d+', '', regex=True)
    df_new.insert(2, "words", df_new['strings_only'].str.split(), True)
    df_new.insert(3, 'amount', [len(v) for v in df_new.words], True)
    short_tail_df = df_new[df_new.amount < 2]
    short_tail_df = short_tail_df.drop(['strings_only', 'words', 'amount'], axis=1)
    short_tail_df.reset_index(drop=True, inplace=True)
    ID = short_tail_df.id
    long_tail_df = df_new[~df_new.id.isin(ID)][['id', 'keyword', 'keyword_eng']].copy()

    print(" *** There were {} SHORT TAIL keywords, and {} LONG TAIL keywords".format(short_tail_df.shape[0],
                                                                                     long_tail_df.shape[0]))
    return long_tail_df, short_tail_df, df_org

def keyowrds_removal(df, list_to_remove):
    for key in range(len(list_to_remove)):
        df['keyword_eng'] = df['keyword_eng'].str.replace(list_to_remove[key], '')
    return df

def stemmList(list):
    # the stemmer requires a language parameter
    snow_stemmer = SnowballStemmer(language='english')
    # porter_stemmer = PorterStemmer()

    nltk.download('punkt')
    stemmed_list = []
    for l in list:
        words = l.split(" ")
        stem_words = []
        # print(l)
        for word in words:
            x = snow_stemmer.stem(word)
            stem_words.append(x)
        key = " ".join(stem_words)
        # print(key)
        stemmed_list.append(key)
    return stemmed_list

def labelling_clusters(df, cluster_num, n):
    df_cluster = df[df["cluster"] == cluster_num]
    keywords_list = df_cluster.keyword_eng.to_list()
    words = [word_tokenize(i) for i in keywords_list]
    words_list = sum(words, [])
    # Remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    clean_words = [word for word in words_list if word not in stop_words]
    clean_words_0 = [re.sub('[^a-zA-Z0-9]+', "", i) for i in clean_words]
    clean_words_1 = [item for item in clean_words_0 if not item.isdigit()]
    clean_words_2 = [x for x in clean_words_1 if x]

    # Make the words singular
    singular_words = [wnl.lemmatize(wrd) for wrd in clean_words_2]
    singular_words_lower = list(map(lambda x: x.lower(), singular_words))

    # Calculate the frequency of each word
    fdist = nltk.FreqDist(singular_words_lower)
    # Rank the words by frequency
    keywords = sorted(fdist, key=fdist.get, reverse=True)
    keywords_1 = [' '.join(keywords[:n])]
    return keywords_1

def clusters_generator_cosine(df,  labels):
    print('*** Keyword clustering using SENTENCE TRANSFORMERS...')
    df.reset_index(drop=True, inplace=True)
    sentences1 = labels
    sentences2 = df.keyword_eng
    id = df.id

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    score = []
    SEN_1 = []
    SEN_2 = []
    ID = []

    # Output the pairs with their score
    for i in range(len(sentences2)):
        for j in range(len(sentences1)):
            score.append(cosine_scores[j][i].item())
            SEN_1.append(sentences1[j])
            SEN_2.append(sentences2[i])
            ID.append(id[i])

    # initialize data of lists.
    data = {'id': ID,
            'semantic_score': score,
            'labels': SEN_1,
            'keyword_eng': SEN_2
            }

    df = pd.DataFrame(data)
    dt = df.loc[df.groupby(['keyword_eng', 'id'])['semantic_score'].idxmax()]
    dt.sort_values('labels', inplace=True)
    return (dt)

def CLUSTERING_K_MEANS(processed_df, long_tail_df, short_tail_df, start_cluster, end_cluster, steps, cutoff):


    global num_cl
    textlist = long_tail_df.keyword_eng.to_list()
    textlist_stem = stemmList(textlist)
    text_data = pd.DataFrame(textlist_stem)
    # Bag of words
    vectorizer_cv = CountVectorizer(analyzer='word')
    X_cv = vectorizer_cv.fit_transform(textlist_stem)
    dic = {}
    LABELS = {}

    for cl_num in range(start_cluster, end_cluster, steps):

        try:
            kmeans = KMeans(n_clusters=cl_num, random_state=10)
            kmeans.fit(X_cv)
            result = pd.concat([text_data, pd.DataFrame(X_cv.toarray(), columns=vectorizer_cv.get_feature_names_out())],
                               axis=1)
            result['cluster'] = kmeans.predict(X_cv)
            result.rename(columns={0: 'Keyword_ENG_stemmed'}, inplace=True)
            df_results = result[['Keyword_ENG_stemmed', 'cluster']].copy()
            df_results.insert(0, "id", long_tail_df.id.values, True)
            df_results.insert(1, "keyword_eng", textlist, True)

            for num_cl in range(cl_num + 1):
                keyword_label = labelling_clusters(df_results, cluster_num=num_cl, n=2)
                cl_lables = ''.join(keyword_label)
                df_results.loc[df_results.cluster == num_cl, 'labels'] = cl_lables

            # Similarity score calculation between LABELS AND KEYWORD ENGLISH.
            # to have an idea how far a keyword from specific cluster.
            sentences1 = df_results.labels
            sentences2 = df_results.keyword_eng

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            y = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
                y.append(cosine_scores[i][i].item())

            df_results['semantic_score'] = y

            df_results.drop(['Keyword_ENG_stemmed', 'cluster'], inplace=True, axis=1)
            labels = df_results.labels.unique()
            df_clusters = clusters_generator_cosine(short_tail_df, labels=labels)
            clusters_short = df_clusters[df_results.columns.values.tolist()]
            df_clusters_all = pd.concat([df_results, clusters_short], ignore_index=True)
            # Generating some statistics:
            z = df_clusters_all.groupby(['labels'])['semantic_score'].mean()
            A = z[z < cutoff]
            final_clusters = topics_generator(df=long_tail_df, df_clusters=df_clusters_all, clusters_labels=labels)
            # Adding columns van original data to the results
            df_org = processed_df.drop(['keyword_eng'], axis=1)
            final_clusters = final_clusters.merge(df_org, on='id', how='left')
            dic[cl_num] = final_clusters
            LABELS[cl_num] = A.index.values

            # print("DATA WITH {} CLUSTERS WAS GENERATED. HOWEVER CLUSTERS {} WERE NOISY".format(num_cl, A.index.values))
            # logger.info(  "DATA WITH {} CLUSTERS WAS GENERATED. HOWEVER CLUSTERS {} WERE NOISY".format(num_cl, A.index.values))

        except Exception as e:
            print(e)
            continue

    return (dic,LABELS)

def CLUSTERING_TRANSFOMERS_K_MEANS(processed_df, long_tail_df, short_tail_df,clusters_amount,cutoff):


    ID = long_tail_df.id.to_list()
    textlist = long_tail_df.keyword_eng.to_list()
    textlist_stem = stemmList(textlist)
    text_data = pd.DataFrame(textlist_stem)
    # Bag of words
    vectorizer_cv = CountVectorizer(analyzer='word')
    X_cv = vectorizer_cv.fit_transform(textlist_stem)

    kmeans = KMeans(n_clusters=clusters_amount, random_state=10)
    kmeans.fit(X_cv)
    result = pd.concat([text_data, pd.DataFrame(X_cv.toarray(), columns=vectorizer_cv.get_feature_names_out())],
                       axis=1)
    result['cluster'] = kmeans.predict(X_cv)
    cluster_assignment = result.cluster

    clustered_sentences = {}
    clustered_sentences_id = {}
    LABELS ={}
    dic = {}

    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        if cluster_id not in clustered_sentences_id:
            clustered_sentences_id[cluster_id] = []

        clustered_sentences[cluster_id].append(textlist[sentence_id])
        clustered_sentences_id[cluster_id].append(ID[sentence_id])


    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()
    df_results = pd.melt(dft, value_vars=dft.columns)
    df_results.dropna(inplace=True)
    df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)

    your_id = pd.DataFrame.from_dict(clustered_sentences_id, orient='index')
    dft_id = your_id.transpose()
    df_id = pd.melt(dft_id, value_vars=dft.columns)
    df_id.dropna(inplace=True)

    df_results['id'] = df_id['value'].astype(int)

    for num_cl in range(clusters_amount + 1):
        keyword_label = labelling_clusters(df_results, cluster_num=num_cl, n=2)
        cl_lables = ''.join(keyword_label)
        df_results.loc[df_results.cluster == num_cl, 'labels'] = cl_lables

    sentences1 = df_results.labels.to_list()
    sentences2 = df_results.keyword_eng.to_list()

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    y = []
    # Output the pairs with their score
    for i in range(len(sentences1)):
        y.append(cosine_scores[i][i].item())

    df_results['semantic_score'] = y
    df_results.drop(['cluster'], inplace=True, axis=1)
    labels = df_results.labels.unique()

    df_clusters = clusters_generator_cosine(short_tail_df, labels=labels)
    clusters_short = df_clusters[df_results.columns.values.tolist()]
    df_clusters_all = pd.concat([df_results, clusters_short], ignore_index=True)
    # Generating some statistics:
    z = df_clusters_all.groupby(['labels'])['semantic_score'].mean()
    A = z[z < cutoff]

    final_clusters = topics_generator(df=long_tail_df, df_clusters=df_clusters_all, clusters_labels=labels)
    # Adding columns van original data to the results
    df_org = processed_df.drop(['keyword_eng'], axis=1)
    final_clusters = final_clusters.merge(df_org, on='id', how='left')
    LABELS[clusters_amount]  = A.index.values
    dic[clusters_amount] = final_clusters


    return (dic, LABELS)

def re_classification(df, cut_off):
    """
    This function can be used  to reclassify cases that were not classified well.
    :param df: classified data frame
    :param cut_off: cut off statistic larger than 0.5  and <=1 .
    :return: re classified data frame.
    """
    labels = df.labels.unique()
    df_cutoff = df[df.semantic_score < cut_off]

    if df_cutoff.shape[0] > 0:
        print(" These are the keywords with a SEMANTIC SCORE lower than {}".format(cut_off))
        print(df_cutoff)
        df_0 = clusters_generator_cosine(df_cutoff, labels)
        df_new = df[~df.id.isin(df_0.id)]  # remove the reclassified keywords.
        reclassified_df = pd.concat([df_0, df_new], ignore_index=True)
        return reclassified_df
    else:
        print("There were no cases with a SEMANTIC SCORE lower than {}. Try a higher cutoff value ".format(cut_off))
        return df


def topics_generator(df, df_clusters, clusters_labels):
    """
    This function is
    :param df: keywords dataframe
    :param df_clusters:
    :param clusters_labels:
    :return:
    """

    # model_name_topics = 'paraphrase-MiniLM-L6-v2'
    model_name_topics = 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name_topics)
    corpus_embeddings = embedder.encode(clusters_labels)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(clusters_labels[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()

    df_1 = pd.melt(dft, value_vars=dft.columns)
    df_1.dropna(inplace=True)
    df_1.rename(columns={'variable': 'TOPICS', 'value': 'SUB_TOPICS'}, inplace=True)

    df_clusters.rename(columns={'labels': 'SUB_TOPICS'}, inplace=True)
    df_clusters_pre = df_clusters.merge(df_1, on='SUB_TOPICS', how='left')
    df_clusters_final = df_clusters_pre.merge(df[['id', 'keyword']], on='id', how='left')
    df_clusters_final = df_clusters_final[['id', 'keyword_eng', 'semantic_score', 'SUB_TOPICS', 'TOPICS']]
    topics = df_clusters_final.TOPICS.max()
    print('*** GENERATING {} TOPICS USING AGGLOMERATIVE CLUSETERING '.format(topics))
    return df_clusters_final


def re_scoring(df):
    sentences1 = df.TOPICS
    sentences2 = df.keyword_eng

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    scores = []
    # Output the pairs with their score
    for i in range(len(sentences1)):
        scores.append(cosine_scores[i][i].item())

    df['semantic_score'] = scores
    # z = df.groupby(['cluster'])['Semantic_score'].mean()
    # A = z[z < 0.5]
    # print("THIS CLUSTERS {} WERE NOISY".format(A.index.values))
    # logger.info("THIS CLUSTERS {} WERE NOISY".format(A.index.values))

    return (df)


def info(df, cutoff):
    z = df.groupby(['cluster'])['Semantic_score'].mean()
    A = z[z < cutoff]
    # x = ("FOR THIS DATA, THE NEXT CLUSTERS {} WERE NOISY".format(A.index.values))
    x = A.index.values
    return (x)


def dfs_xlsx(data_list):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    for sheet_name, df in data_list.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def is_noun(pos):
    if pos[0] == 'N':
        return True
    return False

def filter_nouns(words):
    # Convert the list of words into a single string
    text = " ".join(words)

    # Tokenize the string
    tokens = word_tokenize(text)

    # Part of speech tagging
    pos_tags = nltk.pos_tag(tokens)

    # Filtering nouns
    filtered_nouns = [word for (word, pos) in pos_tags if is_noun(pos)]
    return filtered_nouns

def keywords_to_lables(df):
    keywords_list = df.keyword_eng.to_list()
    words = [word_tokenize(i) for i in keywords_list]
    words_list = sum(words, [])
    # Remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    clean_words = [word for word in words_list if word not in stop_words]
    clean_words_0 = [re.sub('[^a-zA-Z0-9]+', "", i) for i in clean_words]
    clean_words_1 = [item for item in clean_words_0 if not item.isdigit()]
    clean_words_2 = [x for x in clean_words_1 if x]

    # Make the words singular
    wnl = WordNetLemmatizer()
    singular_words = [wnl.lemmatize(wrd) for wrd in clean_words_2]
    singular_words_lower = list(map(lambda x: x.lower(), singular_words))
    filtered_key_nouns = list(set(singular_words_lower))
    return filtered_key_nouns

def topics_generator_transformer(df_clusters, clusters_labels):
    """
    This function is
    :param df: keywords dataframe
    :param df_clusters:
    :param clusters_labels:
    :return:
    """
    # df = long_tail_df; df_clusters = df_clusters_all; clusters_labels = new_labels
    model_name_topics = 'paraphrase-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name_topics)
    corpus_embeddings = embedder.encode(clusters_labels)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(clusters_labels[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()

    df_1 = pd.melt(dft, value_vars=dft.columns)
    df_1.dropna(inplace=True)
    df_1.rename(columns={'variable': 'TOPICS', 'value': 'SUB_TOPICS'}, inplace=True)

    df_clusters.rename(columns={'labels': 'SUB_TOPICS'}, inplace=True)
    # df_clusters_pre = df_clusters.merge(df_1, on='SUB_TOPICS', how='left')
    df_clusters_final = df_clusters.merge(df_1, on='SUB_TOPICS', how='left')
    # df_clusters_final = df_clusters_pre.merge(df[['id', 'keyword']], on='id', how='left')# to add the original keyword columns
    # df_clusters_final = df_clusters_final[['id', 'keyword_eng', 'semantic_score', 'SUB_TOPICS', 'TOPICS']]
    topics = df_clusters_final.TOPICS.max()
    print('*** GENERATING {} TOPICS USING AGGLOMERATIVE CLUSETERING '.format(topics))
    return df_clusters_final

# Comment : CLUSTERING_AGGLOMERATIVE will not work when we test it on more than 10k keywords.
def Agglomerative_Clustering(text_data, n_clusters):
    # Create a TF-IDF representation of the data
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

    # Perform Agglomerative Clustering on the TF-IDF data
    agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agglomerative_clustering.fit_predict(tfidf_matrix.toarray())

    # Create a dataframe with the text data and the assigned clusters
    df = pd.DataFrame({'keyword_eng': text_data, 'cluster': clusters})
    return df


def CLUSTERING_AGGLOMERATIVE(processed_df, long_tail_df, short_tail_df, start_cluster, end_cluster, steps, cutoff):
    keywords = long_tail_df.keyword_eng.to_list()
    dic = {}
    LABELS = {}

    for cl_num in range(start_cluster, end_cluster, steps):

        try:
            df_results = Agglomerative_Clustering(keywords, n_clusters=cl_num)
            df_results.insert(0, "id", long_tail_df.id.values, True)

            for num_cl in range(cl_num + 1):
                keyword_label = labelling_clusters(df_results, cluster_num=num_cl, n=2)
                cl_lables = ''.join(keyword_label)
                df_results.loc[df_results.cluster == num_cl, 'labels'] = cl_lables

            # Similarity score calculation between LABELS AND KEYWORD ENGLISH.
            # to have an idea how far a keyword from specific cluster.
            sentences1 = df_results.labels.to_list()
            sentences2 = df_results.keyword_eng.to_list()

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            y = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
                y.append(cosine_scores[i][i].item())

            df_results['semantic_score'] = y

            df_results.drop(['cluster'], inplace=True, axis=1)
            labels = df_results.labels.unique()
            df_clusters = clusters_generator_cosine(short_tail_df, labels=labels)
            clusters_short = df_clusters[df_results.columns.values.tolist()]
            df_clusters_all = pd.concat([df_results, clusters_short], ignore_index=True)
            # Generating some statistics:
            z = df_clusters_all.groupby(['labels'])['semantic_score'].mean()
            A = z[z < cutoff]
            final_clusters = topics_generator(df=long_tail_df, df_clusters=df_clusters_all, clusters_labels=labels)
            # Adding columns van original data to the results
            df_org = processed_df.drop(['keyword_eng'], axis=1)
            final_clusters = final_clusters.merge(df_org, on='id', how='left')
            dic[cl_num] = final_clusters
            LABELS[cl_num] = A.index.values

        except Exception as e:
            print(e)
            continue

    return (dic, LABELS)


def option_to_model(level_number,options):
  try:
    return options[level_number]
  except Exception as e:
    return e

# These are some model options for sentence transformers:f
option_models = {
    "<select>": '<select>',
    "General Base": 'all-mpnet-base-v2',
    "General Roberta": 'all-distilroberta-v1',
    "General miniML_L12": 'all-MiniLM-L12-v2',
    "General miniML_L6": 'all-MiniLM-L6-v2',
    "Medics": 'pritamdeka/S-PubMedBert-MS-MARCO',
    'Education and training': 'bert-base-nli-mean-tokens',
    'Finance':'roberta-base-nli-mean-tokens',
}

html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;"> KEYWORDS CLUSTERING APP</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)


# provide a color for buttons.
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)


# --- USER AUTHENTICATION ---
names = ['User Unknown','Lars van Tulden', 'Helena Geginat', 'abdelhak chahid','Michael van den Reym','Mitchell Pijl','Nhu Nguyen']
usernames = ['admin','ltulden', 'hgeginat', 'achahid','mreym','mpijl','nnguyen']
passwords = ['io123#$','123#$123', '123#$123', '123#$123','123#$123','123#$123','123#$123']

hashed_passwords = stauth.Hasher(passwords).generate()
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
name, authentication_status, username = authenticator.login('Login', 'sidebar')



if st.session_state["authentication_status"]:

    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    st.sidebar.text('version Jan 2023')

    st.warning("Please ensure that your data includes the column **KEYWORD** :eye-in-speech-bubble: ")
    uploaded_file_cl = st.file_uploader("Upload data", type=['csv'])

    min_value = 2
    max_value = 0
    if uploaded_file_cl is not None:

        keywords_df = pd.read_csv(uploaded_file_cl,encoding='latin-1')
        max_value = np.trunc(keywords_df.shape[0] - 2).astype(int)
        # long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
        st.dataframe(keywords_df)


    load_K_means = st.button('GENERATE CLUSTERS: K-MEANS' )

    if load_K_means:
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
        with st.spinner('**The K-MEANS clustering algorithm is currently in operation. Please hold on ...**'):

            model_name = 'all-MiniLM-L6-v2'
            model = SentenceTransformer(model_name)

            # long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
            max_cluster = np.trunc(keywords_df.shape[0] * 0.1).astype(int)
            min_cluster = np.trunc(max_cluster / 2).astype(int)
            steps = np.trunc((max_cluster - min_cluster) / 3).astype(int)

            cut_off = 0.5
            data_list, labs = CLUSTERING_K_MEANS(processed_data, long_tail_df, short_tail_df, start_cluster=min_cluster,
                                           end_cluster = max_cluster, steps=steps, cutoff=cut_off)


            preffix = 'CLUSTER_id_'# ff
            new_dict = {(preffix + str(key)): value for key, value in data_list.items()}
            data_list = new_dict

            new_labs = {(preffix + str(key)): value for key, value in labs.items()}
            labs = new_labs
            noisy_clusters = pd.DataFrame.from_dict(labs, orient='index')
            noisy_clusters = noisy_clusters.transpose()
            noisy_clusters = noisy_clusters.fillna(value='')
            data_list['Noisy_clusters'] = noisy_clusters
            df_xlsx = dfs_xlsx(data_list)

            st.write("""
            <p style="background-color: #FEC929; color: black; padding: 10px;"> 
            Further examination is recommended for the subsequent clusters..
            </p>
            """, unsafe_allow_html=True)
            # st.balloons()

            st.dataframe(noisy_clusters)
            st.subheader("Download data")
            ste.download_button(label='Download Results',
                               data=df_xlsx,
                               file_name='K_MEANS_clustering.xlsx')

    ### TRANSFOMERS :

    model_name = ["<select>", "General Base", "General Roberta", "General miniML_L12", "General miniML_L6",
                  "Medics", "Education and training", "Finance"]

    select_box = ste.selectbox('Select a model Transformer', options=model_name)
    selected_option = option_to_model(select_box,option_models)
    num_clusters = ste.number_input(label = 'Insert amount of clusters', min_value = min_value, max_value = max_value)
    num_clusters = int(num_clusters)
    st.write('Amount of clusters is ', num_clusters)

    load_transformers = st.button('GENERATE CLUSTERS: TRANSFORMERS')

    if load_transformers and select_box != '<select>':

        st.write('You selected model:', selected_option)
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)

        with st.spinner('**The Model Transformers clustering algorithm is currently running. Please hold on...**'):

            model = SentenceTransformer(selected_option)

            # long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
            # max_cluster = np.trunc(keywords_df.shape[0] * 0.1).astype(int)
            cut_off = 0.5

            data_list, labs = CLUSTERING_TRANSFOMERS_K_MEANS(processed_data, long_tail_df, short_tail_df,
                                                             clusters_amount = num_clusters, cutoff = cut_off)

            preffix = 'CLUSTER_id_'
            new_dict = {(preffix + str(key)): value for key, value in data_list.items()}
            data_list = new_dict

            new_labs = {(preffix + str(key)): value for key, value in labs.items()}
            labs = new_labs
            noisy_clusters = pd.DataFrame.from_dict(labs, orient='index')
            noisy_clusters = noisy_clusters.transpose()
            noisy_clusters = noisy_clusters.fillna(value='')
            data_list['Noisy_clusters'] = noisy_clusters
            df_xlsx = dfs_xlsx(data_list)

            st.write("""
                        <p style="background-color: #FEC929; color: black; padding: 10px;">
                        Further examination is recommended for the subsequent clusters..
                        </p>
                        """, unsafe_allow_html=True)
            # st.balloons()
            st.dataframe(noisy_clusters)
            st.subheader("Download data")
            ste.download_button(label='Download Results',
                               data=df_xlsx,
                               file_name='Transformers_clustering.xlsx')

    # st.subheader("Download data")
    # score_model = results.to_csv(index=False).encode('utf-8')
    # if st.download_button("Download results as CSV", score_model, "Clusters_transformers.csv", "text/csv",
    #                       key='download-tools-csv'):
    #     st.write("Download clicked!")



elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')


if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')














