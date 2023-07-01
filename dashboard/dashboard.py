import streamlit as st
from functions import load_client_info, filter_dataset, reload_scatter_plot, initialize_dashboard, \
    build_gauge, reset_filter


st.set_page_config(layout="wide", initial_sidebar_state='collapsed')


def main():
    initialize_dashboard()

    st.markdown("""
                <style>
                div[data-testid='metric-container'] {
                border: 1px solid rgba(28, 131, 225, 0.1);
                border-radius: 5px;
                padding: 5% 5% 5% 10%;
                background-color: rgba(28, 131, 225, 0.1);
                }
                </style>""", unsafe_allow_html=True)

    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    title_col1, title_col2 = st.columns([0.7, 0.3])
    title_col1.title('Dashboard Client ')
    with title_col2.container():
        placeholder = st.empty()

    with st.form("Form"):
        st.selectbox("Select an ID", st.session_state.ids, key='client_select_box')
        btn = st.form_submit_button("Submit", on_click=load_client_info,
                                    kwargs={'client_id': st.session_state.client_select_box})

    tab1, tab2, tab3, tab4 = st.tabs(['Score', 'Info perso', 'Comparer', 'Historique demandes API'])

    with tab1:
        if btn or st.session_state.load_state:
            st.session_state.load_state = True
            placeholder.header(':open_file_folder: :blue[Dossier : ' + str(st.session_state.selected_client) + ']')
            if st.session_state.missing_id:
                st.write('Id not found in database :ghost:')
            else:
                if st.session_state.prediction == 'Crédit accordé':
                    st.balloons()
                    st.markdown("<h1 style='text-align: center; font-family:Corbel Light;'> Crédit<span "
                                "style='color:#09ab3b'> accordé </span></h1>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center; font-family:Corbel Light;'> Crédit<span "
                                "style='color:#f63366'> refusé </span></h1>",
                                unsafe_allow_html=True)
                st.divider()
                st.subheader('Détails')
                tab1_col1, tab1_col2 = st.columns([0.3, 0.7])
                if st.session_state.prediction == 'Crédit accordé':
                    tab1_col1.plotly_chart(build_gauge('green'), use_container_width=True)
                else:
                    tab1_col1.plotly_chart(build_gauge('#f63366'), use_container_width=True)

                tab1_col1.metric('Type de crédit', st.session_state.contract_type)
                tab1_col2.pyplot(st.session_state.waterfall_plot, use_container_width=True)

                tab1_col1_1, tab1_col1_2 = tab1_col1.columns(2)
                tab1_col1_1.metric('Montant du crédit', st.session_state.amt_credit)
                tab1_col1_2.metric('Montant des annuités', st.session_state.amt_annuity)

    with tab2:
        if btn or st.session_state.load_state:
            if st.session_state.missing_id:
                st.write('Identifiant non trouvé dans la base de données :ghost:')
            else:
                tab2_col1, tab2_col2 = st.columns([0.6, 0.4])
                st.write("""<style>
                            #[data-testid="stMetricDelta"] svg {display: none;}
                            #</style>
                            #""", unsafe_allow_html=True)
                tab2_col1.subheader('Informations personnelles')

                tab2_col1_1, tab2_col1_2, tab2_col1_3 = tab2_col1.columns(3)

                tab2_col1_1.metric('Genre', st.session_state.gender)
                tab2_col1_2.metric('Age', st.session_state.age)
                tab2_col1_3.metric('Nombre d\'enfants', st.session_state.kids)

                tab2_col1.metric('Situation familiale', st.session_state.family)

                tab2_col1_11, tab2_col1_12 = tab2_col1.columns(2)
                tab2_col1_11.metric('Emploi', st.session_state.occupation)
                tab2_col1_12.metric('Revenu total', st.session_state.income)

                # delta='revenu client moyen ' + str(st.session_state.income_mean)
                tab2_col1.metric('Logement', st.session_state.housing)
                tab2_col1.metric('Education', st.session_state.education)

                # tab2_col1.metric('Type de revenus', st.session_state.income_type)

                tab2_col2.subheader('Distributions')
                tab2_col2.pyplot(st.session_state.donut_gender, use_container_width=True)
                with tab2_col2.container():
                    st.divider()
                tab2_col2.pyplot(st.session_state.hist_age, use_container_width=True)

    with tab3:
        with st.form("Form1"):
            st.selectbox("Feature X", st.session_state.numeric_features, key='x_var')
            st.selectbox("Feature Y", st.session_state.numeric_features, key='y_var')

            btn_var = st.form_submit_button("Submit", on_click=reload_scatter_plot)

        with st.expander('Filtrer les données'):
            with st.container():
                st.write('Par dossiers clients similaires')
                st.number_input('Enter a number of neighbors',
                                min_value=1, max_value=st.session_state.number_of_clients - 1,
                                value=10,
                                key='n_neighbors')

            with st.container():
                st.write('Par Genre')
                st.checkbox('M', value=True, key='m')
                st.checkbox('F', value=True, key='f')

            with st.container():
                st.write('Par Age')
                st.slider('Select a range of values', 10, 100, (10, 100), key='age_slider')

            tab3_col1, tab3_col2 = st.columns([0.13, 0.87])
            with tab3_col1.container():
                btn_filter = st.button('Apply Filter', on_click=filter_dataset)
            with tab3_col2.container():
                btn_remove_filter = st.button('Remove Filter', on_click=reset_filter)

        if btn_var or btn_filter or btn_remove_filter:
            st.plotly_chart(st.session_state.scatter, use_container_width=True)

    with tab4:
        st.dataframe(st.session_state.requests_history, hide_index=True, use_container_width=True)


if __name__ == '__main__':
    main()
