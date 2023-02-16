



import streamlit as st
import streamlit_authenticator as stauth
import pathlib
from bs4 import BeautifulSoup
import logging
import shutil


option_models = {
    "<select>" : '<select>',
    "General miniML_L6": 'all-MiniLM-L6-v2 ',
    "General miniML_L12": 'all-MiniLM-L12-v2 ',
    "General Base": 'all-mpnet-base-v2',
    "General Roberta": 'all-distilroberta-v1',
    "General Bert": 'bert-base-nli-mean-tokens',
    "Medics": 'pritamdeka/S-PubMedBert-MS-MARCO',
    'Education and training':'bert-base-nli-mean-tokens',
    'Finance':'bert-base-finance-uncased',

}



def option_to_model(level_number):
  try:
    return option_models[level_number]
  except Exception as e:
    return e


@st.cache(allow_output_mutation=True)
def Pageviews():
    return []

pageviews=Pageviews()
pageviews.append('dummy')



# --- USER AUTHENTICATION ---
names = ['Lars van Tulden', 'Helena Geginat', 'abdelhak chahid','Michael van den Reym']
usernames = ['ltulden', 'hgeginat', 'achahid','mreym']
passwords = ['123#$123', '123#$123', '123#$123','123#$123']

hashed_passwords = stauth.Hasher(passwords).generate()
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'some_cookie_name', 'some_signature_key',
                                    cookie_expiry_days=30)
name, authentication_status, username = authenticator.login('Login', 'sidebar')



if st.session_state["authentication_status"]:

    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    st.sidebar.text('version Jan 2023')

    try:
        st.sidebar.text('Page viewed = {} times.'.format(len(pageviews)))
    except ValueError:
        st.sidebar.text('Page viewed = {} times.'.format(1))




    # Create a list of options for the selectbox
    options = ["<select>", "General miniML_L6", "General miniML_L12", "General Base", "General Roberta", "General Bert",
               "Medics", "Education and training", "Finance"]

    # Create the selectbox and assign the result to a variable
    select_box = st.selectbox('Select an option', options)

    selected_option = option_to_model(select_box)


    # Check the value of the selectbox and run code based on the selection
    if select_box != '<select>':
        st.write('You selected {}'.format(selected_option))
        x=2
        y=3
        st.write(x+y)



elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')


if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

google_analytics_js = """
<!-- Global site tag (gtag.js) - Google Analytics -->
    <script
      async
      src="https://www.googletagmanager.com/gtag/js?id=UA-168384497-1"
    ></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag() {
        dataLayer.push(arguments);
      }
      gtag("js", new Date());
      gtag("config", "UA-168384497-1");
    </script>
    """
st.components.v1.html(google_analytics_js)