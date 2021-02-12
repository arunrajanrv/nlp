import pandas as pd
import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
import re
nltk.download('stopwords')





BOW = pickle.load(open('count_vec.pkl', 'rb'))
RF=pickle.load(open('Random_Forest_Classifier.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))


#tockenization and removing punctutations
def message_text_process(data):
    
    no_punctuation = [char for char in data if char]
    no_punctuation = "".join(no_punctuation)
    return [word for word in no_punctuation.split() if word.lower()
            not in stopwords.words("english")]








#Random Forest Classifier
def predict_(data):
    
    
    prediction = RF.predict(data)
    
    return prediction
    
    





def main():
    
        pattern = r"\d*\/\d*\/[a-zA-z]*:|\d*\/\d*\/\d*\(\w*\):|\d*\|\w*\|\w*:|\(\w*\):|\d*\-\d*\-\d*\(\w*\):|\d*\/\d*\w*:|\d*\/\w*:|\d*\/\d*\/\w*\|:|\d*\/\d*\ *\(\w*\)|\d*\ *\w*\ *\|\ *\w*\ *:|\d*\/\d*\/\d*\(\w*:|\d*\/\w*\/\w*|\d*\/\d*\/\d*\|\(\w*\)|\(\w*\)|\d*\-\d*\-\d*|^\ *|^\:"        
    
        st.title("Customer Relationship Management")       
        
            
        message = st.text_input("Customer Response")
    
        data = [message]
        data = pd.DataFrame(data)
        
        data = data.apply(lambda x: re.sub(pattern," ",str(x)))
        
        data = data.apply(message_text_process).astype(str)    
        data = BOW.transform(data)
        data = tfidf.transform(data)
            
            
        

        
        if st.button("Predict"):
            
            
            
            result=predict_(data)
            
            if result[0] == 'not converted':
                st.success("This lead will not going get converted probabily")
                
            else:
                
                st.success("This lead will probabily going to get converted! Wish you luck!")
            
            
            if st.button("About"):
                st.text("Lets Learn")
                st.text("Built with Streamlit")
            
    
            
            
            
            
            
            
            


if __name__=='__main__':
        
            
        main()