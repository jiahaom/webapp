import streamlit as st
import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import matplotlib.pyplot as plt
import torch


# Title
st.write("""# Recommendation Algorithm""")
st.caption("""
- This app could be used to find the most matched item for the 1st list from the 2nd list.
- Instead of Ctrl/Command + F, the Deep Learning/Neural Network will build a vector system based on NLP, which could enable computers to understand inguistic similarity (synonym).
- More explanations could be found on [Fish&Chips VS AMD chip](https://medium.com/@jiahao.meng/how-deep-learning-impacts-our-daily-work-nlp-for-text-matching-a20bc4a746dd)
- Thanks to [Alex](https://www.linkedin.com/in/alexander-lewis-25942282) and [Nelson](https://www.linkedin.com/in/nelson-chiu-43316b1a4)'s Ideas 🍻
""")

with st.expander("To-Do-List"):
    st.markdown('''
    ✅Add Visualization；

    ✅Add Model Selection；

    ✅Add To-Do-List；

    ✅Publish App；

    📌Excel Support;

    📌Optimaise Visualization;

    📌Add Matching Candidates;


    ''')

# File upload
st.write("""## Uncertain work""")
Uni = st.file_uploader("Choose a table that needs to match (.csv needed)")
if Uni is not None:
    Uni = pd.read_csv(Uni)
    Uni_num = st.sidebar.slider("#Rows Display 1", 5, min(100,len(Uni)))
    st.info('There are {0} rows in the list.'.format(len(Uni)), icon="ℹ️")
    st.write(Uni.head(Uni_num))
    Uni_opt = st.sidebar.multiselect(
        'Which column would you like to use',
        Uni.columns, help="could be single or multiple selection")
    st.write ("""--- """)
    st.sidebar.write ("""--- """)


st.write("""## Libretto""")
Book = st.file_uploader("Choose a Lookup field (.csv needed)")
if Book is not None:
    Book = pd.read_csv(Book)
    Book_num = st.sidebar.slider("#Rows Display 2", 5, min(100,len(Book)))
    st.info('There are {0} rows in the list.'.format(len(Book)), icon="ℹ️")
    st.write(Book.head(Book_num))
    Book_opt = st.sidebar.multiselect(
        'Which column would you like to use',
        Book.columns, help="could be single and multiple selection")
        
    col_opt = st.sidebar.multiselect(
        'Which Metadata would you keep?',
        Book.columns, help="could be single or multiple selection")
    st.sidebar.write ("""--- """)
    visual_check = st.checkbox('Need Visualization?')
    st.write ("""--- """)
    if visual_check:
        with st.spinner('Let\'s Go!'):
            time.sleep(1)
        # Visualization
        st.write ("""## Visualization""")

        font_size = st.sidebar.slider("Font Size", 38, 49)
        Book_visual = Book.replace(0, np.nan)

        ## Need min distinct to limit!!!
        group_col = st.sidebar.selectbox(
        'How would you like to groupby?',
        Book_visual.columns)
        st.sidebar.write ("""--- """)

        Book_visual = (Book_visual.groupby(group_col).count() / Book_visual.count()).drop(group_col, axis = 1)
        Book_visual = Book_visual.fillna(0)
        book_value = []
        for i in Book_visual.columns:
            book_value.append(Book_visual[i].values)

        book_dic = {}
        for i, j in zip (Book_visual.columns, book_value):
            book_dic[i] = list(j)

        def table_visual(dataset, category_names):

            labels = list(dataset.keys())
            data = np.array(list(dataset.values()))
            data_cum = data.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, data.shape[1]))

            fig, ax = plt.subplots(figsize=(30,20))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())
            plt.rc('xtick', labelsize=font_size)
            plt.rc('ytick', labelsize=font_size)

            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
                xcenters = starts + widths / 2

                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                for y, (x, c) in enumerate(zip(xcenters, widths)):
                    ax.text(x, y, str(int(c*100))+'%', ha='center', va='center',
                            color=text_color, fontsize=font_size/1.5)
            ax.legend(bbox_to_anchor=(1.05, 1),
                      loc='upper left', fontsize = 25)

            return fig, ax

        dataset = book_dic
        category_names = list(Book_visual.index)

        pic, _ = table_visual(dataset, category_names)
        # plt.show()
        st.pyplot(pic)


# Preprocessing
if (Book is not None) and (Uni is not None):
    if (len (Uni_opt) and len(Book_opt) and len(col_opt)) != 0:
        Uni = Uni.applymap(str)
        Book = Book.applymap(str)

        if len(Book_opt) < 2:
            Book_embed = Book[Book_opt[0]].values.tolist()
        else:
            Book_embed = Book[Book_opt].values.tolist()

        if len(Uni_opt) < 2:
            Uni_embed = Uni[Uni_opt[0]].values.tolist()
        else:
            Uni_embed = Uni[Uni_opt].values.tolist()

        # https://www.sbert.net/docs/pretrained_models.html
        model_choice = st.sidebar.radio("AI Engine",("all-MiniLM-L12-v2",
                                                    "all-mpnet-base-v2",
                                                    "all-distilroberta-v1",
                                                    "all-MiniLM-L6-v2"))
        clicked = st.sidebar.button ("Run")
        st.sidebar.write ("""--- """)

        if torch.cuda.is_available():
            model = SentenceTransformer('sentence-transformers/'+ model_choice, device='cuda')
        else:
            model = SentenceTransformer('sentence-transformers/'+ model_choice, device='cpu')


# Deep Learning
        if clicked:

            st.warning('Deep Learning coud take several miniutes due to the file size.', icon="⚠️")
            max_index = []
            similarity = []




            Book_embed = model.encode(Book_embed)
            Uni_embed = model.encode(Uni_embed)

            for i in Uni_embed:
                # find the most similar vector
                values = cosine_similarity([i], Book_embed).tolist()[0]
                # obtain the index of the vector
                max_index.append(values.index(max(values)))
                # append similarity
                similarity.append(max(values))
            result = Uni.copy()
            result['max_index'] = max_index

            max_sim = []
            max_index_list = result['max_index'].tolist()
            result.drop(['max_index'], axis = 1, inplace = True)

            result[col_opt] = "NULL"
            temp = 0
            for i in max_index_list:
                for j in col_opt:
                    result.loc[temp,j] = Book.iloc[i][j]
                temp += 1

            result['Similarity'] = similarity

# Final Result
            st.balloons()
            st.success('Match found!', icon="✅")

            st.write(result.head(min(100,len(result))))

# Result Export
            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(result)

            st.download_button(
                label="Save results as CSV",
                data=csv,
                file_name='Final_list.csv',
                mime='text/csv',
            )
