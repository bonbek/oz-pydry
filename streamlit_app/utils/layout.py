import streamlit as st
import config

def base_styles():
    with open("styles.css", "r") as f:
        style = f.read()
        st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

def show_members():

    team = ''.join([member.sidebar_markdown() for member in config.TEAM_MEMBERS])

    st.sidebar.markdown(f"""
    <div id="side-footer">
        <div class="team">
            <div class="team-label">Team members</div>
            {team}
        </div>
        <div class="promo">{config.PROMOTION}</div>
    </div>
    """, unsafe_allow_html=True)