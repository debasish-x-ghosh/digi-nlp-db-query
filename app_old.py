import torch
torch.__version__

import streamlit as st

from transformers import pipeline
import pandas as pd
import os

#App UI starts here 
st.header("Chatbot using LLM")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Tabs
tab1, tab2 = st.tabs(["NLP-DataQuery", "Project2"])

with tab1:  
        st.header('NLP-DataQuery')
        st.subheader('Drill down all options')
        
with tab2:
        st.header('Project2')
        st.subheader('Drill down all options')

#!! TAB1 starts
#Gets the user input
if(pipeline != None):
    tqa = pipeline(task="table-question-answering", 
                    model="google/tapas-base-finetuned-wtq")

    table = pd.read_csv("data.csv")
    table = table.astype(str)
    tab1.dataframe(table)
    print(table)


def load_answer(question):
    print(question)
    #!!-Mutiple Question starts
    query = question.split(',')
    print("query: ")
    print(len(query))
    finalanswer = ""
    print("-----------------------answer----------------------")
    
    if(len(query) == 1):
        #!!-Single Question starts 
        finalanswer = tqa(table=table, query=query)["answer"] 
        return finalanswer
        #!!-Single Question ends
    else:
        #!!-Mutiple Question starts
        answer = tqa(table=table, query=query)
        for ans in answer:
            print("*****************")
            print(ans["answer"])
            finalanswer += "ANSWER: " + ans["answer"]  + ', '
        return finalanswer
        #!!-Mutiple Question ends
    

#!! TAB1 starts
#Gets the user input
def get_text():
    input_text = tab1.text_input("Please ask: (PS: add multiple questions with ',' separation)", key="input")
    return input_text

user_input = get_text()
if(user_input != ""):
    response = load_answer(user_input)

submit = tab1.button('get answer')  

#If generate button is clicked
if submit:
    tab1.subheader("Answer:")
    tab1.write(response)

#!! TAB1 ends