import streamlit as st
from transformers import pipeline
import wikipedia
import re
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_summerizer():
	return pipeline("summarization", 
                    model="t5-base", 
                     tokenizer="t5-base", 
                      framework="tf")
summarizer = load_summerizer()


max_length = st.sidebar.slider("max length", 15, 150, 45, 1)
user_input = st.text_input("WikiPedia Page", "Barak Obama")


wiki_output = wikipedia.summary(user_input, 
                 auto_suggest=True, 
                  redirect=True)
wiki_output = " ".join(re.findall("[a-zA-Z0-9]+", wiki_output))


summary_text = summarizer(wiki_output, max_length=max_length, 
                          min_length=5, 
                          do_sample=False)[0]['summary_text']

st.write(summary_text)
st.button("Re-run")
