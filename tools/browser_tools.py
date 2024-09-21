import json
import os
import google.generativeai as genai
import requests
import time  # Import time for sleep
from crewai import Agent, Task
from langchain.tools import tool
from unstructured.partition.html import partition_html
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini/gemini-1.5-pro")

class BrowserTools:

    @tool("Scrape website content")
    def scrape_and_summarize_website(website):
        """Scrapes and summarizes a website's content."""
        
        # URL for browserless API call
        url = f"https://chrome.browserless.io/content?token={os.getenv('BROWSERLESS_API_KEY')}"
        payload = json.dumps({"url": website})
        headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
        
        # Make the request with rate limit
        response = requests.request("POST", url, headers=headers, data=payload)
        elements = partition_html(text=response.text)
        
        # Process the content and chunk it into manageable pieces
        content = "\n\n".join([str(el) for el in elements])
        content_chunks = [content[i:i + 8000] for i in range(0, len(content), 8000)]
        
        summaries = []
        
        # Loop over each chunk and summarize
        for chunk in content_chunks:
            agent = Agent(
                role='Principal Researcher',
                goal='Do amazing research and summarize based on the content.',
                llm=llm,
                backstory="You're a Principal Researcher at a big company, and you need to summarize the topic.",
                allow_delegation=False
            )
            task = Task(
                agent=agent,
                description=(
                    f'Analyze and summarize the content below, ensuring to include the most relevant information in the summary. Return only the summary, nothing else.\n\nCONTENT\n----------\n{chunk}'
                )
            )
            
            # Execute the task
            summary = task.execute()
            summaries.append(summary)
            
            # Sleep to respect rate limit (Gemini Pro allows 2 requests/minute)
            time.sleep(30)  # Sleep for 30 seconds to avoid rate-limiting
        
        return "\n\n".join(summaries)
