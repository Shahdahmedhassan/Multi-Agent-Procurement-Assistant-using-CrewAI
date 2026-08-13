import os
import json
import time
from datetime import datetime

import requests
import streamlit as st
from bs4 import BeautifulSoup

from crewai import Agent, Task, Crew, Process, LLM
from tavily import TavilyClient


# ==============================================================
# 1) PAGE CONFIGURATION
# ==============================================================

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


# ==============================================================
# 2) SIDEBAR
# ==============================================================

with st.sidebar:

    st.header("🔑 API Keys (Free Tiers)")

    st.markdown(
        "- **Groq:** get a key from "
        "[console.groq.com/keys](https://console.groq.com/keys)\n\n"
        "- **Tavily:** get a key from "
        "[app.tavily.com](https://app.tavily.com/)"
    )

    # ----------------------------------------------------------
    # Read Streamlit Secrets safely
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Company Details
    # ----------------------------------------------------------

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


# ==============================================================
# 3) TAVILY PRODUCT SEARCH
# ==============================================================

def search_products(
    tavily_key: str,
    company_context: dict,
) -> list:

    client = TavilyClient(api_key=tavily_key)

    specs = ", ".join(
        company_context["must_have_specs"]
    )

    query = (
        f"{company_context['procurement_need']} "
        f"business laptop "
        f"budget under ${company_context['budget_per_unit_usd']} "
        f"{specs} "
        f"buy online"
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

        title = item.get(
            "title",
            "Unknown product",
        )

        content = item.get(
            "content",
            "",
        )

        if not url:
            continue

        try:
            source = (
                url.split("/")[2]
                if "://" in url
                else "Unknown"
            )
        except Exception:
            source = "Unknown"

        products.append(
            {
                "title": title,
                "url": url,
                "source": source,
                "snippet": content[:900],
            }
        )

    return products


# ==============================================================
# 4) PRODUCT PAGE SCRAPER
# ==============================================================

def scrape_product_page(url: str) -> str:

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/131.0.0.0 Safari/537.36"
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

        # Keep the page context small.
        return text[:2500]

    except Exception as exc:

        return (
            f"Could not scrape page: {str(exc)[:300]}"
        )


# ==============================================================
# 5) COLLECT PRODUCT DATA
# ==============================================================

def collect_product_data(
    products: list,
) -> list:

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
                "search_snippet": product["snippet"],
                "page_text": page_text,
            }
        )

    return collected


# ==============================================================
# 6) CREATE CREWAI LLM
#
# IMPORTANT:
# Do NOT add max_iter here.
#
# max_iter is NOT a Groq API parameter.
# Adding it to LLM() can cause:
#
# property 'max_iter' is unsupported
# ==============================================================

def create_llm(groq_key: str) -> LLM:

    os.environ["GROQ_API_KEY"] = groq_key

    return LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0.1,
        max_tokens=700,
    )


# ==============================================================
# 7) RUN CREW
# ==============================================================

def run_crew(
    company_context: dict,
    groq_key: str,
    research_data: list,
):

    llm = create_llm(groq_key)

    # ----------------------------------------------------------
    # Keep data small to reduce Groq TPM usage
    # ----------------------------------------------------------

    compact_data = []

    for item in research_data[:5]:

        compact_data.append(
            {
                "name": item["name"],
                "url": item["url"],
                "source": item["source"],
                "search_snippet": item[
                    "search_snippet"
                ][:500],
                "page_text": item[
                    "page_text"
                ][:1200],
            }
        )

    context_json = json.dumps(
        company_context,
        ensure_ascii=False,
    )

    research_json = json.dumps(
        compact_data,
        ensure_ascii=False,
    )

    # ==========================================================
    # AGENT 1
    # ==========================================================

    research_agent = Agent(
        role="Procurement Research Specialist",

        goal=(
            "Organize product evidence into a factual shortlist. "
            "Never invent prices, specifications, warranty details, "
            "or product information."
        ),

        backstory=(
            "You are an experienced procurement researcher. "
            "You carefully distinguish verified facts from missing "
            "information."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # ==========================================================
    # AGENT 2
    # ==========================================================

    analyst_agent = Agent(
        role="Procurement Analyst",

        goal=(
            "Compare products against company requirements and "
            "rank them according to price, specifications, warranty, "
            "and business value."
        ),

        backstory=(
            "You are a professional procurement analyst. "
            "You make practical purchasing recommendations using "
            "only available evidence."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # ==========================================================
    # AGENT 3
    # ==========================================================

    report_agent = Agent(
        role="Procurement Report Writer",

        goal=(
            "Create a concise professional procurement report in "
            "complete HTML."
        ),

        backstory=(
            "You are an executive business report writer who "
            "converts procurement analysis into clear reports "
            "for management."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # ==========================================================
    # TASK 1 - RESEARCH
    # ==========================================================

    research_task = Task(

        description=(
            "Review the company requirements and collected product "
            "evidence below.\n\n"

            f"COMPANY REQUIREMENTS:\n"
            f"{context_json}\n\n"

            f"PRODUCT EVIDENCE:\n"
            f"{research_json}\n\n"

            "Create a factual shortlist of up to 5 products.\n\n"

            "For every product provide:\n"
            "- Product name\n"
            "- Source\n"
            "- URL\n"
            "- Price if explicitly available\n"
            "- Key specifications if explicitly available\n"
            "- Warranty if explicitly available\n"
            "- Missing information\n\n"

            "IMPORTANT:\n"
            "Do not invent information. "
            "If something cannot be verified, write "
            "'Not verified'."
        ),

        expected_output=(
            "A concise factual shortlist of up to 5 products "
            "with verified evidence and missing information."
        ),

        agent=research_agent,
    )

    # ==========================================================
    # TASK 2 - ANALYSIS
    # ==========================================================

    analysis_task = Task(

        description=(
            "Analyze the research shortlist from the previous task.\n\n"

            f"COMPANY REQUIREMENTS:\n"
            f"{context_json}\n\n"

            f"PRIORITIES:\n"
            f"{company_context['priority_order']}\n\n"

            f"BUDGET PER UNIT:\n"
            f"${company_context['budget_per_unit_usd']}\n\n"

            "Rank the products from best to worst.\n\n"

            "For each product explain briefly:\n"
            "- Whether it appears to meet the requirements\n"
            "- Price suitability\n"
            "- Specification suitability\n"
            "- Warranty suitability\n"
            "- Main advantage\n"
            "- Main concern\n\n"

            "Clearly mark any product where important information "
            "could not be verified.\n\n"

            "Do not invent facts."
        ),

        expected_output=(
            "A concise ranked procurement analysis containing "
            "a recommended product and alternatives."
        ),

        agent=analyst_agent,

        context=[research_task],
    )

    # ==========================================================
    # TASK 3 - FINAL REPORT
    # ==========================================================

    report_task = Task(

        description=(
            "Create the final procurement report using the "
            "research and analysis from the previous tasks.\n\n"

            "The output MUST be complete HTML.\n\n"

            "It MUST start with:\n"
            "<!DOCTYPE html>\n\n"

            "Include:\n"
            "1. Report title\n"
            "2. Company information\n"
            "3. Procurement request\n"
            "4. Quantity\n"
            "5. Budget per unit\n"
            "6. Must-have specifications\n"
            "7. Executive Summary\n"
            "8. Product Comparison Table\n"
            "9. Final Recommendation\n"
            "10. Rationale\n"
            "11. Verification Notes\n"
            "12. Report Date\n\n"

            "The comparison table must contain:\n"
            "Product | Source | Price | Specifications | "
            "Warranty | Recommendation\n\n"

            "Use clean inline CSS.\n"
            "Use a professional business layout.\n"
            "Do not invent facts.\n"
            "If information is unavailable, write "
            "'Not verified'.\n\n"

            f"Report Date: "
            f"{datetime.now().strftime('%Y-%m-%d')}"
        ),

        expected_output=(
            "Complete ready-to-render HTML procurement report."
        ),

        agent=report_agent,

        context=[analysis_task],
    )

    # ==========================================================
    # CREW
    # ==========================================================

    crew = Crew(

        agents=[
            research_agent,
            analyst_agent,
            report_agent,
        ],

        tasks=[
            research_task,
            analysis_task,
            report_task,
        ],

        process=Process.sequential,

        verbose=False,
    )

    return crew.kickoff()


# ==============================================================
# 8) RETRY FUNCTION
# ==============================================================

def run_crew_with_retry(
    company_context: dict,
    groq_key: str,
    research_data: list,
):

    max_attempts = 2

    for attempt in range(max_attempts):

        try:

            return run_crew(
                company_context,
                groq_key,
                research_data,
            )

        except Exception as exc:

            error_text = str(exc).lower()

            is_rate_limit = (
                "rate limit" in error_text
                or "ratelimit" in error_text
                or "429" in error_text
                or "tokens per minute" in error_text
                or "tpm" in error_text
            )

            if not is_rate_limit:
                raise

            if attempt == max_attempts - 1:
                raise

            # Wait before retrying.
            time.sleep(35)


# ==============================================================
# 9) CLEAN HTML
# ==============================================================

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


# ==============================================================
# 10) MAIN APPLICATION
# ==============================================================

if run_button:

    # ----------------------------------------------------------
    # Validate API Keys
    # ----------------------------------------------------------

    if not groq_key:

        st.error(
            "Please enter GROQ_API_KEY."
        )

        st.stop()

    if not tavily_key:

        st.error(
            "Please enter TAVILY_API_KEY."
        )

        st.stop()

    # ----------------------------------------------------------
    # Build Company Context
    # ----------------------------------------------------------

    company_context = {

        "company_name": company_name,

        "procurement_need": procurement_need,

        "quantity": int(quantity),

        "budget_per_unit_usd": float(
            budget
        ),

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

    # ----------------------------------------------------------
    # Run Workflow
    # ----------------------------------------------------------

    try:

        with st.status(
            "🤖 Agents at work... This may take about a minute.",
            expanded=True,
        ) as status:

            # ==================================================
            # STEP 1 - SEARCH
            # ==================================================

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
                    "Try a broader product name or check "
                    "your Tavily API key."
                )

                st.stop()

            st.write(
                f"✅ Found {len(products)} product sources."
            )

            # ==================================================
            # STEP 2 - SCRAPING
            # ==================================================

            st.write(
                "📄 Collecting product-page information..."
            )

            research_data = collect_product_data(
                products
            )

            # ==================================================
            # STEP 3 - CREWAI
            # ==================================================

            st.write(
                "🧠 CrewAI Research Agent is organizing "
                "the evidence..."
            )

            result = run_crew_with_retry(
                company_context,
                groq_key,
                research_data,
            )

            status.update(
                label="✅ Report is ready!",
                state="complete",
            )

        # ======================================================
        # CLEAN RESULT
        # ======================================================

        html_output = clean_html_output(
            result
        )

        # ======================================================
        # SHOW REPORT
        # ======================================================

        st.subheader(
            "📄 Final Report"
        )

        st.components.v1.html(
            html_output,
            height=700,
            scrolling=True,
        )

        # ======================================================
        # DOWNLOAD REPORT
        # ======================================================

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

    # ==========================================================
    # ERROR HANDLING
    # ==========================================================

    except Exception as exc:

        error_text = str(exc)

        st.error(
            "❌ The application could not complete the workflow."
        )

        lower_error = error_text.lower()

        # ------------------------------------------------------
        # MAX_ITER ERROR
        # ------------------------------------------------------

        if (
            "max_iter" in lower_error
            and "unsupported" in lower_error
        ):

            st.warning(
                "The Groq request contains an unsupported "
                "'max_iter' parameter. Make sure the deployed "
                "app.py is the latest version and that there is "
                "no old LiteLLM monkey-patch in the file."
            )

        # ------------------------------------------------------
        # RATE LIMIT
        # ------------------------------------------------------

        elif (
            "rate limit" in lower_error
            or "ratelimit" in lower_error
            or "429" in lower_error
            or "tokens per minute" in lower_error
        ):

            st.warning(
                "Groq reached its free-tier token limit. "
                "The app has been optimized to use smaller "
                "requests. Wait about 30–40 seconds and run "
                "the workflow again."
            )

        # ------------------------------------------------------
        # AUTH ERROR
        # ------------------------------------------------------

        elif (
            "401" in lower_error
            or "authentication" in lower_error
            or "invalid api key" in lower_error
        ):

            st.warning(
                "The Groq API key was rejected. "
                "Check GROQ_API_KEY and make sure the complete "
                "key was entered."
            )

        # ------------------------------------------------------
        # TAVILY ERROR
        # ------------------------------------------------------

        elif "tavily" in lower_error:

            st.warning(
                "The Tavily request failed. "
                "Check TAVILY_API_KEY and try again."
            )

        # ------------------------------------------------------
        # GENERAL ERROR
        # ------------------------------------------------------

        else:

            st.warning(
                "An unexpected error occurred. "
                "Check the technical details below."
            )

        st.exception(exc)

# ==============================================================
# INITIAL PAGE
# ==============================================================

else:

    st.info(
        "Fill in your company details and API keys in the "
        "sidebar, then click 'Run Agents'."
    )
