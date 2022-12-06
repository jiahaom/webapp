import streamlit as st
import hashlib
import datetime


st.write("""# Password Manager""")
st.write(
        """
-   This app is an easy-to-use reversible password generator.
-   Due to the balance between forgetting and protection, passwords are generated from essential information.
	    """
    )
with st.expander("To-Do-List"):
    st.markdown('''
    ✅Password Generator;

    📌Excel Encryption;

    ''')
st.write("""## Password Generator""")
col1, col2, col3 = st.columns(3)
with col1:
    pw_case = st.text_input('Case Number','0')
    pw_case = int(float(pw_case))
with col2:
    pw_date = st.date_input(
        "File create Date",
        datetime.date(2022, 8, 1)).strftime("%Y%m%d")
    pw_date = int(pw_date)
with col3:
    pw_author= st.text_input('Author Name', 'Guido van Rossum', help = 'Full name on MS Teams')
    pw_author= sum(list(pw_author.encode('utf-8')))

pw = st.button('Generate Password')
if pw:
    # temp_pw = hashlib.sha3_256(b'dsadas').hexdigest()
    temp_pw = str(hex(pw_case+pw_date+pw_author))
    temp_pw = hashlib.sha3_256(temp_pw.encode('utf-8')).hexdigest()
#     temp_pw = temp_pw[0:4] +'_'+ temp_pw[4:8]+'_'+ temp_pw[8:12]
    temp_pw = temp_pw[0:4].title() +'_'+ temp_pw[4:8]+'_'+ temp_pw[8:12].title()
	
    temp_pw = 'Your revertible Password is: \n' + str (temp_pw)
    st.info(temp_pw, icon = "ℹ️")

st.write(""" ---""")
# st.write("""## Excel Encryption""")
