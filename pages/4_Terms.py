import streamlit as st
import pandas as pd
import openai
import random

# Load the CSV file
file_path = 'data/Financial_Terms_short.csv'
data = pd.read_csv(file_path)

api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)

# Function to get a random term
def get_random_term(data):
    random_row = data.sample(n=1).iloc[0]
    term = random_row['Term']
    definition = random_row['Definition']
    return term, definition

# Function to get related links using ChatGPT
def get_related_links(term):
    system_prompt = 'You are a helpful assistant tasked with providing a relevant link to an article, website, or resource related to financial terms, prioritizing Investopedia links.'

    main_prompt = f"""
    ###TASK###
    - Provide one relevant link to an article, website, or resource related to the financial term '{term}', prioritizing links from Investopedia. Your response should be the link only and remove filler words
    """

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": main_prompt}
            ]
        )
        links = response.choices[0].message.content.strip().split('\n')
        return links

    except Exception as e:
        st.error(f"Error fetching related links: {e}")
        return []

# Function to get feedback and explanation from ChatGPT
def get_feedback_and_explanation(term, user_answer):
    feedback_prompt = f"""
    You are a helpful assistant. Prrovide constructive feedback on the  user answer and explain the term '{term}' like a 12-year-old. Please do not mention "12 year old" when explaining it to the user.

    ###ANSWER###
    {user_answer}
    """

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "Provide constructive feedback based on the user answer and help the user under stand the Financial Terms."},
                {"role": "user", "content": feedback_prompt}
            ]
        )
        feedback = response.choices[0].message.content.strip()
        return feedback

    except Exception as e:
        st.error(f"Error fetching feedback and explanation: {e}")
        return ""

# Streamlit app
st.set_page_config(layout="wide")
st.title("Finance Term of the Day ")

# Check if term and definition are already in session state
if 'term' not in st.session_state or 'definition' not in st.session_state:
    term, definition = get_random_term(data)
    st.session_state['term'] = term
    st.session_state['definition'] = definition
else:
    term = st.session_state['term']
    definition = st.session_state['definition']

st.subheader(f"Term: {term}")
st.write(f"Definition: {definition}")

st.subheader("Related Link")
links = get_related_links(term)
for link in links:
    st.write(link)

# st.write("Refresh the page for a new term!")

# Question related to the term of the day
st.subheader("Question about the Term")
question = f"What does the term '{term}' mean to you? Please explain in your own words."

st.write(question)
user_answer = st.text_area("Your Answer", "")

if st.button("Submit Answer"):
    feedback = get_feedback_and_explanation(term, user_answer)
    st.subheader("Feedback and Explanation")
    st.write(feedback)
