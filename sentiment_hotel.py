import pickle
import streamlit as st

# Load the vectorizer and the model
v = pickle.load(open('vectorizer.pkl', 'rb'))
loaded_model = pickle.load(open('trained_model.pkl', 'rb'))

st.title("Hotel Sentiment Analysis")

def main():
    input_text = st.text_input('Review')

    if st.button('Result'):
        # Transform the input text using the loaded vectorizer
        rev_vec = v.transform([input_text])
        prediction = loaded_model.predict(rev_vec)

        # Print the prediction
        #st.write(prediction)
        st.write('The class is ', prediction[0])


if __name__ == '__main__':
    main()
