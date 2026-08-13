import os
import json
import requests
from datetime import datetime
from typing import Type

import streamlit as st
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tavily import TavilyClient


# ----------------------------------------------------------------------
# 1) Page Configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Procurement Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Procurement Assistant")

st.caption(
    "Powered by CrewAI — Search, compare, and generate professional "
    "procurement reports seamlessly using free tiers."
)


# ----------------------------------------------------------------------
# 2) Sidebar: API Keys + Company & Request Details
# ----------------------------------------------------------------------

with st.sidebar:

    st.header("🔑 API Keys (Free Tiers)")

    st.markdown(
        "- **Groq**: Get a free key from "
        "[console.groq.com/keys](https://console.groq.com/keys)\n"
        "- **Tavily**: Get a free key from "
        "[app.tavily.com](https://app.tavily.com)"
    )

    # Safely read Streamlit secrets
    try:
        default_groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_groq_key = ""

    try:
        default_tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    except Exception:
        default_tavily_key = ""

    groq_key = st.text_input(
        "GROQ_API_KEY",
        value=default_groq_key,
        type="password",
    )

    tavily_key = st.text_input(
        "TAVILY_API_KEY",
        value=default_tavily_key,
        type="password",
    )

    st.divider()

    st.header("🏢 Company & Order Details")

    company_name = st.text_input(
        "Company Name",
        "Constant Tech Solutions"
    )

    procurement_need = st.text_input(
        "Product Needed",
        "Laptops for Engineers"
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=15
    )

    budget = st.number_input(
        "Budget per Unit (USD)",
        min_value=1,
        value=1500
    )

    must_have = st.text_area(
        "Must-Have Specifications (One per line)",
        "RAM 16GB or higher\n"
        "SSD 512GB or higher\n"
        "Processor Intel i7 / Ryzen 7\n"
        "At least 2 years warranty",
    )

    priority = st.text_input(
        "Comparison Priorities (Comma-separated)",
        "Price, Specifications, Warranty, Brand Reputation"
    )

    run_button = st.button(
        "🚀 Run Agents",
        type="primary",
        use_container_width=True
    )


# ----------------------------------------------------------------------
# 3) Tavily Search Tool
# ----------------------------------------------------------------------

class TavilySearchInput(BaseModel):

    query: str = Field(
        ...,
        description="Search query for the product, price, or specifications."
    )


class TavilySearchTool(BaseTool):

    name: str = "tavily_search"

    description: str = (
        "An internet search tool to find products, suppliers, prices, "
        "and specifications. Use a clear search query."
    )

    args_schema: Type[BaseModel] = TavilySearchInput

    tavily_api_key: str = ""

    def _run(self, query: str) -> str:

        try:

            client = TavilyClient(
                api_key=self.tavily_api_key
            )

            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )

            formatted = []

            for r in results.get("results", []):

                content = r.get("content", "")

                formatted.append(
                    f"- Title: {r.get('title', 'Unknown')}\n"
                    f"  URL: {r.get('url', 'N/A')}\n"
                    f"  Snippet: {content[:400]}"
                )

            if formatted:
                return "\n\n".join(formatted)

            return "No results found."

        except Exception as e:

            return (
                f"Tavily search failed: {str(e)}"
            )


# ----------------------------------------------------------------------
# 4) Simple Web Scraper
# ----------------------------------------------------------------------

class SimpleScrapeInput(BaseModel):

    url: str = Field(
        ...,
        description="The URL of the product page to scrape."
    )


class SimpleScraperTool(BaseTool):

    name: str = "simple_scraper"

    description: str = (
        "A free tool to extract text from a webpage (no paid APIs). "
        "Provide the target product page URL."
    )

    args_schema: Type[BaseModel] = SimpleScrapeInput

    def _run(self, url: str) -> str:

        try:

            headers = {
                "User-Agent":
                "Mozilla/5.0 (compatible; ProcurementBot/1.0)"
            }

            resp = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            resp.raise_for_status()

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )

            for tag in soup(
                ["script", "style", "nav", "footer", "header"]
            ):
                tag.decompose()

            text = " ".join(
                soup.get_text(
                    separator=" "
                ).split()
            )

            return text[:3000]

        except Exception as e:

            return (
                f"Failed to extract data from {url}: {str(e)}"
            )


# ----------------------------------------------------------------------
# 5) Crew Setup and Execution
# ----------------------------------------------------------------------

def run_crew(
    company_context: dict,
    groq_key: str,
    tavily_key: str
):

    # --------------------------------------------------------------
    # Validate API Keys
    # --------------------------------------------------------------

    if not groq_key or not groq_key.strip():
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    if not tavily_key or not tavily_key.strip():
        raise ValueError(
            "TAVILY_API_KEY is missing."
        )

    # --------------------------------------------------------------
    # Set environment variables
    # --------------------------------------------------------------

    os.environ["GROQ_API_KEY"] = groq_key.strip()
    os.environ["TAVILY_API_KEY"] = tavily_key.strip()

    # --------------------------------------------------------------
    # LLM
    # --------------------------------------------------------------

    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_key.strip(),
        temperature=0.3,
        max_tokens=2048,
    )

    # --------------------------------------------------------------
    # Tools
    # --------------------------------------------------------------

    search_tool = TavilySearchTool(
        tavily_api_key=tavily_key.strip()
    )

    scrape_tool = SimpleScraperTool()

    # --------------------------------------------------------------
    # Agent 1 - Search Agent
    # --------------------------------------------------------------

    search_agent = Agent(

        role="Product Search Specialist",

        goal=(
            "Find the best available products and suppliers "
            "matching company requirements."
        ),

        backstory=(
            "An expert in online product and price research "
            "who filters results and retrieves relevant links."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=True,

        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Agent 2 - Data Collection Agent
    # --------------------------------------------------------------

    scraper_agent = Agent(

        role="Data Collection Specialist",

        goal=(
            "Extract precise data including price, specifications, "
            "and warranty from product pages."
        ),

        backstory=(
            "A specialist in structured data extraction "
            "from web pages."
        ),

        tools=[scrape_tool],

        llm=llm,

        verbose=True,

        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Agent 3 - Procurement Analyst
    # --------------------------------------------------------------

    analyst_agent = Agent(

        role="Procurement Analyst",

        goal=(
            "Compare and rank products based on price, "
            "specifications, warranty, and overall value."
        ),

        backstory=(
            "An experienced procurement analyst who evaluates "
            "offers objectively and provides data-driven recommendations."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Agent 4 - Report Writer
    # --------------------------------------------------------------

    report_agent = Agent(

        role="Procurement Report Writer",

        goal=(
            "Write a professional procurement report "
            "formatted in HTML."
        ),

        backstory=(
            "An executive report writer who transforms technical "
            "analyses into clear, structured reports."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Company Context
    # --------------------------------------------------------------

    context_str = json.dumps(
        company_context,
        ensure_ascii=False,
        indent=2
    )

    # --------------------------------------------------------------
    # Task 1 - Search
    # --------------------------------------------------------------

    search_task = Task(

        description=(
            f"Based on the following company requirements:\n"
            f"{context_str}\n\n"

            "Search for at least 5 products matching the "
            "specifications from different sources. "

            "Collect the product name, URL, and source. "

            "Focus on real products and real suppliers. "
            "Do not invent product information."
        ),

        expected_output=(
            "A list of at least 5 products, each containing "
            "a title, URL, source, and a short description."
        ),

        agent=search_agent,
    )

    # --------------------------------------------------------------
    # Task 2 - Scrape
    # --------------------------------------------------------------

    scrape_task = Task(

        description=(
            "From the product links provided by the Search Agent, "
            "extract the current price, specifications, and "
            "warranty period if available for each product. "

            "If information is unavailable, explicitly say "
            "'Not available' instead of inventing information."
        ),

        expected_output=(
            "A structured data layout for each product containing: "
            "Name, Price, Specifications, Warranty, and URL."
        ),

        agent=scraper_agent,

        context=[search_task],
    )

    # --------------------------------------------------------------
    # Task 3 - Analysis
    # --------------------------------------------------------------

    analysis_task = Task(

        description=(

            f"Compare the products based on company priorities: "
            f"{company_context['priority_order']} "

            f"and budget: "
            f"{company_context['budget_per_unit_usd']} USD per unit. "

            "Rank the products from best to lowest value. "

            "Consider the required specifications, price, "
            "warranty, and brand reputation. "

            "Clearly explain why the first-ranked product "
            "is recommended. "

            "Do not invent missing data."
        ),

        expected_output=(
            "A final ranking of products with a brief rationale "
            "for each ranking and a clear final recommendation."
        ),

        agent=analyst_agent,

        context=[scrape_task],
    )

    # --------------------------------------------------------------
    # Task 4 - Report
    # --------------------------------------------------------------

    report_task = Task(

        description=(

            "Write a complete final Procurement report in full "
            "HTML format including <html>, <head>, and <body>. "

            "The report must contain: "

            "1. Title "
            "2. Executive Summary "
            "3. Company Requirements "
            "4. Comparison Table "
            "5. Final Ranking "
            "6. Final Recommendation with Rationale "
            "7. Procurement Notes "
            "8. Date "

            "Use clean inline CSS inside <style> "
            "for a professional business layout. "

            "Use only information available in the previous tasks. "
            "Do not invent prices, specifications, or warranties."
        ),

        expected_output=(
            "Complete and ready-to-render/download HTML code."
        ),

        agent=report_agent,

        context=[analysis_task],
    )

    # --------------------------------------------------------------
    # Crew
    # --------------------------------------------------------------

    crew = Crew(

        agents=[
            search_agent,
            scraper_agent,
            analyst_agent,
            report_agent
        ],

        tasks=[
            search_task,
            scrape_task,
            analysis_task,
            report_task
        ],

        process=Process.sequential,

        verbose=True,
    )

    # --------------------------------------------------------------
    # Run Crew
    # --------------------------------------------------------------

    try:

        result = crew.kickoff()

        return result

    except Exception as e:

        error_text = str(e)

        # Make the most common errors easier to understand

        if "401" in error_text or "invalid_api_key" in error_text.lower():

            raise RuntimeError(
                "❌ Groq API key is invalid or expired. "
                "Create a new key from Groq Console and paste it "
                "into GROQ_API_KEY."
            ) from e

        if "403" in error_text:

            raise RuntimeError(
                "❌ Groq rejected the request (403). "
                "Check your Groq project/model permissions."
            ) from e

        if "429" in error_text or "rate limit" in error_text.lower():

            raise RuntimeError(
                "❌ Groq rate limit reached. "
                "Wait a little and try again."
            ) from e

        if "BadRequestError" in error_text:

            raise RuntimeError(
                "❌ Groq rejected the request as a Bad Request. "
                "Check the API key and try again. "
                f"\n\nOriginal error: {error_text}"
            ) from e

        raise RuntimeError(
            f"❌ CrewAI failed while running the agents.\n\n"
            f"{error_text}"
        ) from e


# ----------------------------------------------------------------------
# 6) Main Page Logic
# ----------------------------------------------------------------------

if run_button:

    if not groq_key or not tavily_key:

        st.error(
            "Please enter both GROQ_API_KEY and "
            "TAVILY_API_KEY in the sidebar."
        )

    else:

        company_context = {

            "company_name":
                company_name,

            "procurement_need":
                procurement_need,

            "quantity":
                quantity,

            "budget_per_unit_usd":
                budget,

            "must_have_specs":
                [
                    s.strip()
                    for s in must_have.split("\n")
                    if s.strip()
                ],

            "priority_order":
                [
                    p.strip()
                    for p in priority.split(",")
                    if p.strip()
                ],
        }

        # ----------------------------------------------------------
        # Run Agents
        # ----------------------------------------------------------

        try:

            with st.status(
                "🤖 Agents at work... This may take about a minute.",
                expanded=True
            ) as status:

                st.write(
                    "🔍 Search Agent is looking for products..."
                )

                result = run_crew(
                    company_context,
                    groq_key,
                    tavily_key
                )

                status.update(
                    label="✅ Report is ready!",
                    state="complete"
                )

        except Exception as e:

            st.error(
                "The application could not complete the request."
            )

            st.code(
                str(e),
                language="text"
            )

            st.stop()

        # ----------------------------------------------------------
        # Clean HTML Output
        # ----------------------------------------------------------

        html_output = str(result)

        if "```html" in html_output:

            html_output = (
                html_output
                .split("```html", 1)[1]
                .split("```", 1)[0]
                .strip()
            )

        elif "```HTML" in html_output:

            html_output = (
                html_output
                .split("```HTML", 1)[1]
                .split("```", 1)[0]
                .strip()
            )

        elif "```" in html_output:

            html_output = (
                html_output
                .split("```", 1)[1]
                .split("```", 1)[0]
                .strip()
            )

        # ----------------------------------------------------------
        # Display Report
        # ----------------------------------------------------------

        st.subheader("📄 Final Report")

        st.components.v1.html(
            html_output,
            height=600,
            scrolling=True
        )

        # ----------------------------------------------------------
        # Download Report
        # ----------------------------------------------------------

        filename = (
            f"procurement_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        )

        st.download_button(
            "⬇️ Download Report (HTML)",
            data=html_output,
            file_name=filename,
            mime="text/html",
            use_container_width=True,
        )

else:

    st.info(
        "Fill in your company details and free API keys "
        "in the sidebar, then click 'Run Agents'."
    )
