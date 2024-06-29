import streamlit as st
import pandas as pd
from openai import OpenAI
import random
import re
from annotated_text import annotated_text

api_key = st.secrets["api_key"]
client = OpenAI(api_key=api_key)

def extract_keywords_with_definitions(text):
    system_prompt = 'You are a financial news analyst assistant tasked to extract keywords related to finance from news articles and provide simple definitions for each keyword.'
    main_prompt = """
    ###TASK###
    - Extract the five most crucial finance-related keywords from the news article.
    - For each keyword, provide a simple definition as if explaining to a five-year-old, in the context of the article provided.
    - Return the results as a Python dictionary, where each key is a keyword and its value is the simple definition in the context of the article.
    - Example: {"stock": "A tiny piece of a company that you can buy", "ETF": "A basket of different stocks you can buy all at once", "bitcoin": "A special kind of computer money", "mutual funds": "A collection of investments that many people put money into together", "bond": "A way to lend money to a company or government and get paid back later with extra"}
    ###ARTICLE###
    """
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
        keywords_with_definitions = eval(response.choices[0].message.content)
        return keywords_with_definitions
    except:
        return "Unable to extract keywords, please try again later"
    
def generate_summary(text, keywords):
    system_prompt = 'You are a financial news analyst assistant tasked to summarize articles and link key concepts together.'
    main_prompt = f"""
    ###TASK###
    - Summarize the given article in about 50 words.
    - Use and link together the following keywords in your summary: {', '.join(keywords)}
    - The summary should be easy to understand for a general audience.
    ###ARTICLE###
    """
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
        summary = response.choices[0].message.content
        return summary
    except:
        return "Unable to generate summary, please try again later"

st.set_page_config(layout="wide")
st.title('Tagging articles with their most relevant keywords')
df = pd.read_csv("data/combined_data.csv").sort_values(
    'date', ascending=False
)

title = st.selectbox(
    'Select article title', df['title'], index=None
)

if title:
    article = df[df['title']==title].iloc[0]
                        
    st.header(f"[{article['title']}]({article['link']})")
    st.caption(f"__Published date:__ {article['date']}")
    st.caption('**TOP KEYWORDS**')
    keywords_dict = extract_keywords_with_definitions(article['paragraph'])
    
    # Create a row of columns for the keywords
    cols = st.columns(len(keywords_dict))
    
    # Display each keyword in its own column with an expander
    for i, (keyword, definition) in enumerate(keywords_dict.items()):
        with cols[i]:
            with st.expander(keyword):
                st.caption('DEFINITION')
                st.write(definition)

    # Generate and display summary
    st.subheader('Article Summary')
    summary = generate_summary(article['paragraph'], keywords_dict.keys())
    st.write(summary)

    st.subheader('Full article content')    
    st.write(article['paragraph'])