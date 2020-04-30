"""P1T6_Analisis_sentimientos.ipynb

Original file is located at
    https://colab.research.google.com/drive/1wvbWYAH1MjYI4r5px4nSYmqtLZyxzWzW

## Importar librerias necesarias
"""

import os
import tweepy as tw
import pandas as pd

"""# Accediendo a la API de Twitter en Python """

consumer_key = 'nkjbpKyMEGEO8Ezy84figKZ0v'
consumer_secret = 'eFdTEbucPvRNtXH74L98SjIg51Olc4LGcuTwL5hdhPBUfiu0Vq'
access_token = '1176236145551826951-CyxNooeUEUscCt8j5Z5LOlDTtUzUJv'
access_token_secret = 'mhsQ6rwRlXXtYGUhNgxfyh9P7PN6MwXxaGQEk4qOWJmAI'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

"""# Buscar Tweets"""

search_words = '#15Abr'
date_since = '2020-04-01'

new_search = search_words + " -filter:retweets"

tweets = tw.Cursor(api.search, q=new_search, lang='es', since=date_since).items(1001)

"""# Visualizar Tweets"""

dtweets = [[tweet.user.screen_name, tweet.user.location, tweet.text] for tweet in tweets]
dtweets[:10]

"""#Visualizando Tweets recolectados en un dataframe"""

tw_dataframe = pd.DataFrame(data=dtweets, columns=["user", "location", "text"])
tw_dataframe[:10]

"""# Guardamos la data en un csv."""

tw_dataframe.to_csv("/content/drive/My Drive/Analisis_sentimientos_Twitter/TWEETS_LGGG_15Abr.csv", index=False,
                    encoding='utf-8')
# tw_dataframe = pd.read_csv(file_name, encoding='utf-8')
# tw_dataframe.head(10)

# Visualizar el datafreme
dataframe = tw_dataframe
dataframe.head(10)

"""# Procesamiento"""

import re  # librería para la búsqueda y manipulación de cadenas
from nltk import TweetTokenizer  # librería para tokenizar(separar y clasificar palabras)
from nltk.stem import SnowballStemmer  # algoritmo para clasificación de palabras

# variables para mejorar la escritura (opcional)
NORMALIZE = 'normalize'
REMOVE = 'remove'
MENTION = 'twmention'
HASHTAG = 'twhashtag'
URL = 'twurl'
LAUGH = 'twlaugh'
# definir que el algoritmo de clasificación use el idioma español
_stemmer = SnowballStemmer('spanish')

# definir una variable para la funcion de tokenizar (opcional)
_tokenizer = TweetTokenizer().tokenize

# variable para definir si quiero normalizar: normalize o eliminar: remove los hashtags, menciones y urls en los tweets
_twitter_features = "normalize"

# variable para definir si se desea tener convertir o no a la raiz de la palabra.
_stemming = False

"""# Definir listas de conversión"""

# lista de conversión para quitar las tildes a las vocales.
DIACRITICAL_VOWELS = [('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ü', 'u')]

# lista para corregir algunas palabras coloquiales / jerga en español (obviamente faltan más)
SLANG = [('d', 'de'), ('[qk]', 'que'), ('xo', 'pero'), ('xa', 'para'), ('[xp]q', 'porque'), ('es[qk]', 'es que'),
         ('fvr', 'favor'), ('(xfa|xf|pf|plis|pls|porfa)', 'por favor'), ('dnd', 'donde'), ('tb', 'también'),
         ('(tq|tk)', 'te quiero'), ('(tqm|tkm)', 'te quiero mucho'), ('x', 'por'), ('\+', 'mas')]

"""# Método para normalización de risas"""


def normalize_laughs(message):
    message = re.sub(r'\b(?=\w*[j])[aeiouj]{4,}\b', LAUGH, message, flags=re.IGNORECASE)
    message = re.sub(r'\b(?=\w*[k])[aeiouk]{4,}\b', LAUGH, message, flags=re.IGNORECASE)
    message = re.sub(r'\b(juas+|lol)\b', LAUGH, message, flags=re.IGNORECASE)
    return message


"""# Método para eliminar o normalizar menciones, hashtags y URLS"""


def process_twitter_features(message, twitter_features):
    message = re.sub(r'[\.\,]http', '. http', message, flags=re.IGNORECASE)
    message = re.sub(r'[\.\,]#', '. #', message)
    message = re.sub(r'[\.\,]@', '. @', message)

    if twitter_features == REMOVE:
        # eliminar menciones, hashtags y URL
        message = re.sub(r'((?<=\s)|(?<=\A))(@|#)\S+', '', message)
        message = re.sub(r'\b(https?:\S+)\b', '', message, flags=re.IGNORECASE)
    elif twitter_features == NORMALIZE:
        # cuando sea necesario se normalizaran las menciones, hashtags y URL
        message = re.sub(r'((?<=\s)|(?<=\A))@\S+', MENTION, message)
        message = re.sub(r'((?<=\s)|(?<=\A))#\S+', HASHTAG, message)
        message = re.sub(r'\b(https?:\S+)\b', URL, message, flags=re.IGNORECASE)

    return message


"""# Método global"""


def preprocess(message):
    # convertir a minusculas
    message = message.lower()

    # eliminar números, retorno de linea y el tan odios retweet (de los viejos estilos de twitter)
    message = re.sub(r'(\d+|\n|\brt\b)', '', message)

    # elimar vocales con signos diacríticos (posible ambigüedad)
    for s, t in DIACRITICAL_VOWELS:
        message = re.sub(r'{0}'.format(s), t, message)

    # eliminar caracteres repetidos
    message = re.sub(r'(.)\1{2,}', r'\1\1', message)

    # normalizar las risas
    message = normalize_laughs(message)

    # traducir la jerga y terminos coloquiales sobre todo en el español
    for s, t in SLANG:
        message = re.sub(r'\b{0}\b'.format(s), t, message)

    # normalizar/eliminar hashtags, menciones y URL
    message = process_twitter_features(message, _twitter_features)

    # Convertir las palabras a su raiz ( Bonita, bonito) -> bonit
    if _stemming:
        message = ' '.join(_stemmer.stem(w) for w in _tokenizer(message))

    return message


# Imprimir metodo global

print(preprocess(dataframe['text'].loc[1]))
print("\n")
print(dataframe.loc[1])

"""# Aplicamos preprocesamiento al CSV y creamos un nuevo CSV limpio"""

dataframe['text'] = dataframe['text'].apply(preprocess)
print(dataframe.loc[1])

dataframe.to_csv("/content/drive/My Drive/Analisis_sentimientos_Twitter/TWEETS_LGGG_15Abr_clean.csv", index=False,
                 encoding='utf-8')

"""# Conectar a drive"""

from google.colab import drive

drive.mount('/content/drive')

"""#Descargamos la libreria de stopwords en español"""

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('spanish')

"""#Convertir cada uno de los tweets en un vector donde cada uno de los elementos es una palabra o símbolo gramatical"""


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


"""#Función para extraer un documento del dataset"""

print("p2.2: funcion para extraer un documento del dataset  ")


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# p2.3: función que tomara una secuencia de documentos y devolverá un número particular de documentos
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


"""# Serializamos (congelamos) el modelo para usarlo fuera de google colaboratory"""

import pickle
import os

# creo una carpeta en mi google drive para guardar los archivos serializados
dest = os.path.join('/content/drive/My Drive/IA/Analisis_sentimientos_Twitter/twitterclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
# convertimos el clasificador y el stopword en archivo/objectos pkl
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
# Es importante recordar que deben verificar que los dos archivos esten en su drive

"""# Cambiamos la basepath (directorio por defecto) de Python a la carpeta de Twitterclassifier"""

import os

os.chdir('/content/drive/My Drive/Analisis_sentimientos_Twitter/twitterclassifier')

"""#Deserializamos los estimadores"""

import pickle
import re
import os
from vectorizer import vect

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

"""# Instalar libreria pyprind"""

pip
install
pyprind

"""#RECORREMOS LOS TWEETS DESCARGADOS Y LOS CLASIFICAMOS"""

import numpy as np
import pandas as pd
import pyprind

df = pd.read_csv("/content/drive/My Drive/Analisis_sentimientos_Twitter/TWEETS_LGGG_15Abr_clean.csv", encoding='utf-8')
# creamos una columna llamada Sentimient donde guardaremos la predicción
df['sentiment'] = ''
# creamos una columna llamada Probability donde guardaremos la acertabilidad que dio el clasificador
df['probability'] = 0
# conversión de sentimientos (numeros a palabras)= NONE->-1 | NEU -> 0 | P->1 | N->2
label = {-1: 'Sin sentimiento', 0: 'Neutro', 1: 'Positivo', 2: 'Negativo'}
for rowid in range(len(df.index)):
    text = df['text'][rowid]
    textConvert = vect.transform([text])
    df['sentiment'][rowid] = label[clf.predict(textConvert)[0]]
    df['probability'][rowid] = np.max(clf.predict_proba(textConvert)) * 100
    pbar.update()
# df.head(20)
df.to_csv('/content/drive/My Drive/Analisis_sentimientos_Twitter/TWEETS_LGGG_15Abr_analysis.csv', index=False,
          encoding='utf-8')

"""# Generar gráficos estadísticos"""

import matplotlib.pyplot as plt

# sentimientos = df["sentiment"].unique()
df.groupby('sentiment')['location'].nunique().plot(kind='bar')
print(df.groupby(['sentiment']).size())
# df.groupby(['sentiment']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
