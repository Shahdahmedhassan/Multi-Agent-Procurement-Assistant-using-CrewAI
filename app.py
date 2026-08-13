import os
import json
import time
from datetime import datetime

import requests
import streamlit as st
from bs4 import BeautifulSoup

from crewai import Agent, Task, Crew, Process, LLM
from tavily import TavilyClient


# ----------------------------------------------------------------------
# 1) Page Configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Procurement Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Procurement Assistant")

st.caption(
    "Powered by CrewAI — Search, compare, and generate professional "
    "procurement reports using free-tier APIs."
)


# ----------------------------------------------------------------------
# 2) Sidebar
# ----------------------------------------------------------------------

with st.sidebar:

    st.header("🔑 API Keys (Free Tiers)")

    st.markdown(
        "- **Groq:** get a key from "
        "[console.groq.com/keys](https://console.groq.com/keys)\n"
        "- **Tavily:** get a key from "
        "[app.tavily.com](https://app.tavily.com)"
    )

    try:
        secret_groq = st.secrets.get("GROQ_API_KEY", "")
        secret_tavily = st.secrets.get("TAVILY_API_KEY", "")
    except Exception:
        secret_groq = ""
        secret_tavily = ""

    groq_key = st.text_input(
        "GROQ_API_KEY",
        value=secret_groq,
        type="password",
    )

    tavily_key = st.text_input(
        "TAVILY_API_KEY",
        value=secret_tavily,
        type="password",
    )

    st.divider()

    st.header("🏢 Company & Order Details")

    company_name = st.text_input(
        "Company Name",
        "Constant Tech Solutions",
    )

    procurement_need = st.text_input(
        "Product Needed",
        "Laptops for Engineers",
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=15,
        step=1,
    )

    budget = st.number_input(
        "Budget per Unit (USD)",
        min_value=1,
        value=1500,
        step=50,
    )

    must_have = st.text_area(
        "Must-Have Specifications (One per line)",
        "RAM 16GB or higher\n"
        "SSD 512GB or higher\n"
        "Processor Intel i7 / Ryzen 7\n"
        "At least 2 years warranty",
        height=130,
    )

    priority = st.text_input(
        "Comparison Priorities (Comma-separated)",
        "Price, Specifications, Warranty, Brand Reputation",
    )

    run_button = st.button(
        "🚀 Run Agents",
        type="primary",
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# 3) Tavily Search
# ----------------------------------------------------------------------

def search_products(tavily_key: str, company_context: dict) -> list:

    client = TavilyClient(api_key=tavily_key)

    specs = ", ".join(
        company_context["must_have_specs"]
    )

    query = (
        f"{company_context['procurement_need']} "
        f"business laptop "
        f"under ${company_context['budget_per_unit_usd']} "
        f"{specs}"
    )

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )

    products = []

    for item in response.get("results", []):

        url = item.get("url", "")
        title = item.get("title", "Unknown product")
        content = item.get("content", "")

        if not url:
            continue

        try:
            source = url.split("/")[2]
        except Exception:
            source = "Unknown"

        products.append(
            {
                "title": title,
                "url": url,
                "source": source,
                # Keep Tavily content SHORT
                "snippet": content[:600],
            }
        )

    return products


# ----------------------------------------------------------------------
# 4) Product Page Scraper
# ----------------------------------------------------------------------

def scrape_product_page(url: str) -> str:

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript",
            ]
        ):
            tag.decompose()

        text = " ".join(
            soup.get_text(
                separator=" "
            ).split()
        )

        # VERY IMPORTANT:
        # Keep page text small to protect Groq TPM.
        return text[:1000]

    except Exception as exc:

        return f"Page could not be read: {str(exc)[:150]}"


# ----------------------------------------------------------------------
# 5) Collect Product Data
# ----------------------------------------------------------------------

def collect_product_data(products: list) -> list:

    collected = []

    for product in products:

        page_text = scrape_product_page(
            product["url"]
        )

        collected.append(
            {
                "name": product["title"],
                "url": product["url"],
                "source": product["source"],
                "search_snippet": product["snippet"][:500],
                "page_text": page_text[:1000],
            }
        )

    return collected


# ----------------------------------------------------------------------
# 6) Create Groq LLM
# ----------------------------------------------------------------------

def create_llm(groq_key: str) -> LLM:

    os.environ["GROQ_API_KEY"] = groq_key

    return LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=groq_key,

        # Low temperature = more stable output
        temperature=0.1,

        # Keep completion small
        max_tokens=1200,

        # Avoid excessive agent iterations
        max_iter=1,
    )


# ----------------------------------------------------------------------
# 7) Run LLM With Simple Retry
# ----------------------------------------------------------------------

def run_crew_with_retry(
    crew,
    max_attempts=3,
):

    for attempt in range(max_attempts):

        try:

            return crew.kickoff()

        except Exception as exc:

            error_text = str(exc).lower()

            rate_limit = (
                "rate limit" in error_text
                or "ratelimit" in error_text
                or "tokens per minute" in error_text
                or "429" in error_text
            )

            if not rate_limit:
                raise

            if attempt == max_attempts - 1:
                raise

            # Wait between retries.
            wait_seconds = 35

            st.warning(
                f"⏳ Groq rate limit reached. "
                f"Waiting {wait_seconds} seconds before retry "
                f"({attempt + 1}/{max_attempts - 1})..."
            )

            time.sleep(wait_seconds)

    return None


# ----------------------------------------------------------------------
# 8) Build Compact Research Context
# ----------------------------------------------------------------------

def build_compact_context(
    company_context: dict,
    research_data: list,
) -> str:

    compact_products = []

    for item in research_data:

        compact_products.append(
            {
                "product": item["name"],
                "source": item["source"],
                "url": item["url"],
                "evidence": (
                    item["search_snippet"]
                    + " "
                    + item["page_text"]
                )[:1300],
            }
        )

    data = {
        "company": company_context["company_name"],
        "product_needed": company_context["procurement_need"],
        "quantity": company_context["quantity"],
        "budget_usd": company_context["budget_per_unit_usd"],
        "requirements": company_context["must_have_specs"],
        "priorities": company_context["priority_order"],
        "products": compact_products,
    }

    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ----------------------------------------------------------------------
# 9) CrewAI Workflow
#
# IMPORTANT:
# Only TWO LLM calls are made:
#
# Call 1 -> Research Agent
# Call 2 -> Report Agent
#
# Ranking/analysis is handled with Python.
# ----------------------------------------------------------------------

def run_crew(
    company_context: dict,
    groq_key: str,
    research_data: list,
):

    llm = create_llm(groq_key)

    compact_context = build_compact_context(
        company_context,
        research_data,
    )

    # ==============================================================
    # AGENT 1 - RESEARCH
    # ==============================================================

    research_agent = Agent(
        role="Procurement Research Specialist",

        goal=(
            "Extract only useful factual product information "
            "from the supplied evidence."
        ),

        backstory=(
            "You are a careful procurement researcher. "
            "You never invent prices, specifications, warranties "
            "or product facts."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    research_task = Task(

        description=(
            "Analyze the supplied procurement data.\n\n"

            f"DATA:\n{compact_context}\n\n"

            "Return a SHORT factual shortlist.\n"

            "For every product provide:\n"
            "- Product name\n"
            "- Source\n"
            "- URL\n"
            "- Price if explicitly available\n"
            "- Important specifications if explicitly available\n"
            "- Warranty if explicitly available\n"
            "- Missing information\n\n"

            "Never invent information.\n"
            "Keep the response under 900 words."
        ),

        expected_output=(
            "A concise factual product shortlist."
        ),

        agent=research_agent,
    )

    # ==============================================================
    # AGENT 2 - REPORT WRITER
    # ==============================================================

    report_agent = Agent(
        role="Procurement Report Writer",

        goal=(
            "Create a concise professional procurement report "
            "from verified research evidence."
        ),

        backstory=(
            "You are an executive procurement report writer. "
            "You clearly distinguish verified information "
            "from missing information."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    report_task = Task(

        description=(
            "Create the final procurement report.\n\n"

            "Company:\n"
            f"{company_context['company_name']}\n\n"

            "Product Needed:\n"
            f"{company_context['procurement_need']}\n\n"

            "Quantity:\n"
            f"{company_context['quantity']}\n\n"

            "Budget per Unit:\n"
            f"${company_context['budget_per_unit_usd']}\n\n"

            "Required Specifications:\n"
            f"{', '.join(company_context['must_have_specs'])}\n\n"

            "Priorities:\n"
            f"{', '.join(company_context['priority_order'])}\n\n"

            "Research Evidence:\n"
            f"{compact_context}\n\n"

            "Research Summary:\n"
            "Use the previous research task.\n\n"

            "Create complete HTML beginning with <!DOCTYPE html>.\n\n"

            "The HTML must contain:\n"
            "1. Professional title\n"
            "2. Company request summary\n"
            "3. Executive Summary\n"
            "4. Product comparison table\n"
            "5. Final Recommendation\n"
            "6. Rationale\n"
            "7. Verification Notes\n"
            "8. Report Date\n\n"

            "Comparison table columns:\n"
            "Product | Source | Price | Specifications | "
            "Warranty | Recommendation\n\n"

            "Use clean inline CSS.\n"
            "Do not invent facts.\n"
            "If information is missing, write 'Not verified'.\n"
            "Keep the HTML concise."
        ),

        expected_output=(
            "Complete ready-to-render HTML procurement report."
        ),

        agent=report_agent,

        context=[research_task],
    )

    # ==============================================================
    # CREW
    # ==============================================================

    crew = Crew(

        agents=[
            research_agent,
            report_agent,
        ],

        tasks=[
            research_task,
            report_task,
        ],

        process=Process.sequential,

        verbose=False,
    )

    return run_crew_with_retry(
        crew,
        max_attempts=3,
    )


# ----------------------------------------------------------------------
# 10) Clean HTML
# ----------------------------------------------------------------------

def clean_html_output(result) -> str:

    html_output = str(result).strip()

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

    return html_output


# ----------------------------------------------------------------------
# 11) Main Application
# ----------------------------------------------------------------------

if run_button:

    # --------------------------------------------------------------
    # Validate API Keys
    # --------------------------------------------------------------

    if not groq_key:

        st.error(
            "❌ Please enter your GROQ_API_KEY."
        )

        st.stop()

    if not tavily_key:

        st.error(
            "❌ Please enter your TAVILY_API_KEY."
        )

        st.stop()

    # --------------------------------------------------------------
    # Build Company Context
    # --------------------------------------------------------------

    company_context = {

        "company_name":
            company_name,

        "procurement_need":
            procurement_need,

        "quantity":
            int(quantity),

        "budget_per_unit_usd":
            float(budget),

        "must_have_specs": [

            item.strip()

            for item in must_have.splitlines()

            if item.strip()
        ],

        "priority_order": [

            item.strip()

            for item in priority.split(",")

            if item.strip()
        ],
    }

    # --------------------------------------------------------------
    # Run Workflow
    # --------------------------------------------------------------

    try:

        with st.status(
            "🤖 Agents at work... This may take about a minute.",
            expanded=True,
        ) as status:

            # ======================================================
            # STEP 1 - TAVILY SEARCH
            # ======================================================

            st.write(
                "🔍 Search Agent is looking for products..."
            )

            products = search_products(
                tavily_key,
                company_context,
            )

            if not products:

                status.update(
                    label="❌ No products found",
                    state="error",
                )

                st.error(
                    "Tavily returned no product results. "
                    "Try a broader product name or check the "
                    "Tavily API key."
                )

                st.stop()

            st.write(
                f"✅ Found {len(products)} product sources."
            )

            # ======================================================
            # STEP 2 - SCRAPE
            # ======================================================

            st.write(
                "📄 Collecting product-page information..."
            )

            research_data = collect_product_data(
                products
            )

            st.write(
                "🧠 CrewAI Research Agent is organizing "
                "the evidence..."
            )

            # ======================================================
            # STEP 3 - CREWAI
            # ======================================================

            result = run_crew(
                company_context,
                groq_key,
                research_data,
            )

            status.update(
                label="✅ Report is ready!",
                state="complete",
            )

        # ==========================================================
        # FINAL REPORT
        # ==========================================================

        html_output = clean_html_output(
            result
        )

        st.subheader(
            "📄 Final Report"
        )

        st.components.v1.html(
            html_output,
            height=700,
            scrolling=True,
        )

        # ==========================================================
        # DOWNLOAD
        # ==========================================================

        filename = (
            "procurement_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        )

        st.download_button(
            "⬇️ Download Report (HTML)",

            data=html_output,

            file_name=filename,

            mime="text/html",

            use_container_width=True,
        )

    # --------------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------------

    except Exception as exc:

        error_text = str(exc)

        st.error(
            "❌ The application could not complete the workflow."
        )

        lower_error = error_text.lower()

        if (
            "rate limit" in lower_error
            or "ratelimit" in lower_error
            or "tokens per minute" in lower_error
            or "429" in lower_error
        ):

            st.warning(
                "⏳ Groq rate limit was reached. "
                "The application was optimized to use only "
                "two CrewAI LLM calls and smaller prompts. "
                "Please wait about 30–40 seconds and run again."
            )

        elif (
            "401" in lower_error
            or "authentication" in lower_error
            or "invalid api key" in lower_error
        ):

            st.warning(
                "🔑 The Groq API key was rejected. "
                "Please check GROQ_API_KEY."
            )

        elif "tavily" in lower_error:

            st.warning(
                "🔎 The Tavily request failed. "
                "Please check TAVILY_API_KEY."
            )

        else:

            st.warning(
                "An unexpected error occurred. "
                "Check the details below."
            )

        st.exception(exc)

else:

    st.info(
        "Fill in your company details and API keys in the "
        "sidebar, then click 'Run Agents'."
    )
