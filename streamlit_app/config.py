"""

Config file for Streamlit App

"""

from utils.member import Member


TITLE = "Gonna Rain Tomorow ?"

TEAM_MEMBERS = [
    Member(
        name="Aalae Benki",
        researchgate_url="https://www.researchgate.net/profile/Aalae-Benki",
        github_url="https://github.com/abenki87",
    ),
    Member(name="Christopher Corbin", linkedin_url="https://www.linkedin.com/in/christopher-corbin-865911146/",
        github_url="https://github.com/tofferPika"),
    Member(name="Romaric Reynier", linkedin_url="linkedin.com/in/romaric-reynier-069ab617b",
        github_url="https://github.com/Romaric-Reynier")
]

PROMOTION = "Bootcamp Data Scientist - February 2023"

MAPBOX_TOKEN = 'pk.eyJ1IjoidG9mZmVyIiwiYSI6ImNqOG85ZWZuYTAxM2wycXJzMzhnZjF5ODYifQ.SqyxswaJV3siPVqlzqwoaQ'