from crewai import Crew
from trip_agents import TripAgents, StreamToExpander
from trip_tasks import TripTasks
import streamlit as st
import datetime
import sys
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_icon="✈️", layout="wide")


class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range
        self.output_placeholder = st.empty()

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range
        )

        plan_task = tasks.plan_task(
            travel_concierge_agent,
            self.origin,
            self.interests,
            self.date_range
        )

        crew = Crew(
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True
        )

        result = crew.kickoff()
        self.output_placeholder.markdown(result)

        return result


if __name__ == "__main__":
    st.title("AI Travel Agent")
    st.subheader("Let AI agents plan your next vacation!",
                 divider="rainbow", anchor=False)

    import datetime

    today = datetime.datetime.now().date()
    next_year = today.year + 1
    jan_16_next_year = datetime.date(next_year, 1, 10)

    with st.sidebar:
        st.header("👇 Enter your trip details")
        with st.form("my_form"):
            location = st.text_input(
                "Current location", placeholder="San Mateo, CA")
            cities = st.text_input(
                "Destination", placeholder="Bali, Indonesia")
            date_range = st.date_input(
                "Travel date range",
                min_value=today,
                value=(today, jan_16_next_year + datetime.timedelta(days=6)),
                format="MM/DD/YYYY",
            )
            interests = st.text_area("High level interests and hobbies or extra details about your trip?",
                                     placeholder="2 adults who love swimming, dancing, hiking, and eating")

            submitted = st.form_submit_button("Submit")

if submitted:
    with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
        with st.container(height=500, border=False):
            sys.stdout = StreamToExpander(st)
            trip_crew = TripCrew(location, cities, date_range, interests)
            result = trip_crew.run()
        status.update(label="✅ Trip Plan Ready!",
                      state="complete", expanded=False)

    st.subheader("Here is your Trip Plan", anchor=False, divider="rainbow")
    st.markdown(result)