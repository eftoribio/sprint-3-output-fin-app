import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# nltk.download('punkt') # Downloads the Punkt tokenizer models
# nltk.download('stopwords') # Downloads the list of stopwords
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
###

stopwords = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'u',
 'says',
 'said',
 'says ',
 'said ',
 'would',
 'also',
 'even',
 'like']

api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)
SKLLMConfig.set_openai_key(api_key)
client = OpenAI(api_key=api_key)

###

stop_words = stopwords.words('english') + ['u', 'says', 'said', 'says ', 'said ', 'would', 'also', 'even', 'like']
def generate_wordcloud(text, cmap='viridis', stop_words=stop_words):
    try:
        titles = df['paragraph'].str.cat(sep=' ').lower()
        wordcloud = WordCloud(width = 800, height = 400,
                            background_color ='white',
                            stopwords = stop_words,
                            min_font_size = 10,
                            colormap=cmap).generate(text)
        plt.figure(figsize = (6, 6), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
        return wordcloud
    except Exception as e:
        st.write("Sorry, no articles for this topic.")
        return None
    
def summarize(text, selected_topic):
    system_prompt = 'You are an economist, finance expert, and personal finance advocate. \
        You are tasked to read through articles and simply explain the news articles to a layman in the context of the provided topic.'

    main_prompt = f"""
    ###TASK###
    - Provide a comprehensive but simple summary of the news articles in less than 100 words, in such a way that a layman could easily understand. Integrate the the topic name {selected_topic} into your summary.
    - Explain how these news articles could affect the everyday life of the reader. 
    - Provide practical finance tips and advice taking into consideration the news. 
       
    ### GENERAL GUIDELINES ###
    - In the first two tasks, write it in paragraph form. in the last task, list down your answers.
    - Keep a conversational tone. imagine you are talking as a personal finance advocate talking to a group of people.
    


    ###ARTICLE###
    """

    response = client.chat.completions.create(
            model='gpt-3.5-turbo', 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
    return response.choices[0].message.content

st.set_page_config(layout='wide')
st.title('Finance News Explainer')

## TOPIC
topics = ['stocks', 'retirement', 'insurance', 'loans', 'credit cards', 'mortgage']
selected_topic = st.selectbox("Select a topic", topics)
# st.write(f"Displaying news articles on {selected_month} {selected_year}")

## READ DATA
df = pd.read_csv("data/combined_data-st.csv").sort_values(
    'date', ascending=False
)
df['date'] = pd.to_datetime(df['date'])

## FILTER DATA ACC TO INPUT
df = df[df['gpt_topics'].apply(lambda topics: selected_topic in topics)]

col1, col2 = st.columns(2)

## POS
with col1:
    st.subheader('Positive News')
    pos = df[df['gpt_sentiment'] == 'Positive']
    titles = pos['paragraph'].str.cat(sep=' ').lower()
    wordcloud = generate_wordcloud(titles)
    if wordcloud is not None:
        st.image(wordcloud.to_array(), use_column_width=True)

## NEG  
with col2:  
    st.subheader('Negative News')
    neg = df[df['gpt_sentiment'] == 'Negative']
    titles = neg['paragraph'].str.cat(sep=' ').lower()
    wordcloud = generate_wordcloud(titles, cmap='inferno')
    if wordcloud is not None:
        st.image(wordcloud.to_array(), use_column_width=True)

## SUMMARIZER
st.subheader('Summarize')
test = summarize(df['paragraph'], selected_topic)
st.write(test)