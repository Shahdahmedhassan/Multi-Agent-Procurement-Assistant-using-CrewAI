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
#    IMPORTANT:
#    Search is done directly with Tavily instead of giving a Tavily
#    tool to CrewAI. This avoids the CrewAI/LiteLLM native-tool path
#    that was causing the previous BadRequestError.
# ----------------------------------------------------------------------
def search_products(tavily_key: str, company_context: dict) -> list:
    client = TavilyClient(api_key=tavily_key)

    specs = ", ".join(company_context["must_have_specs"])

    query = (
        f"{company_context['procurement_need']} "
        f"for business engineers, "
        f"budget under ${company_context['budget_per_unit_usd']} per unit, "
        f"{specs}, buy online"
    )

    response = client.search(
        query=query,
        search_depth="advanced",
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

        products.append(
            {
                "title": title,
                "url": url,
                "source": item.get("url", "").split("/")[2]
                if "://" in item.get("url", "")
                else "Unknown",
                "snippet": content[:1200],
            }
        )

    return products


# ----------------------------------------------------------------------
# 4) Simple Product Page Scraper
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
            timeout=12,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            ["script", "style", "nav", "footer", "header", "noscript"]
        ):
            tag.decompose()

        text = " ".join(
            soup.get_text(separator=" ").split()
        )

        return text[:5000]

    except Exception as exc:
        return f"Could not scrape page: {exc}"


# ----------------------------------------------------------------------
# 5) Build product research data
# ----------------------------------------------------------------------
def collect_product_data(products: list) -> list:
    collected = []

    for product in products:
        page_text = scrape_product_page(product["url"])

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


# ----------------------------------------------------------------------
# 6) CrewAI LLM
#
# We intentionally do NOT monkey-patch litellm.completion here.
# The old patch was wrapping itself repeatedly on Streamlit reruns,
# which caused the repeated _completion_no_cache_breakpoint traceback.
#
# We also use the smaller/faster Groq model to reduce free-tier
# pressure.
# ----------------------------------------------------------------------
def create_llm(groq_key: str) -> LLM:
    os.environ["GROQ_API_KEY"] = groq_key

    return LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0.2,
        max_tokens=1800,
    )


# ----------------------------------------------------------------------
# 7) CrewAI Multi-Agent Workflow
#
# Search and scraping happen outside CrewAI.
# CrewAI agents are responsible for:
#   1. Research organization
#   2. Procurement analysis
#   3. Professional report writing
#
# This keeps CrewAI away from native tool-calling and avoids the
# previous Groq/LiteLLM cache_breakpoint problem.
# ----------------------------------------------------------------------
def run_crew(company_context: dict, groq_key: str, research_data: list):
    llm = create_llm(groq_key)

    # Keep the context reasonably small for free-tier usage.
    compact_data = []

    for item in research_data:
        compact_data.append(
            {
                "name": item["name"],
                "url": item["url"],
                "source": item["source"],
                "search_snippet": item["search_snippet"][:700],
                "page_text": item["page_text"][:1800],
            }
        )

    context_json = json.dumps(
        company_context,
        ensure_ascii=False,
        indent=2,
    )

    research_json = json.dumps(
        compact_data,
        ensure_ascii=False,
        indent=2,
    )

    # --------------------------------------------------------------
    # Agent 1 - Research Organizer
    # --------------------------------------------------------------
    research_agent = Agent(
        role="Procurement Research Specialist",
        goal=(
            "Organize the collected online product evidence into a "
            "clean, factual shortlist. Never invent missing prices, "
            "specifications, warranty information, or URLs."
        ),
        backstory=(
            "You are an experienced procurement researcher. "
            "You carefully separate confirmed facts from missing data."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Agent 2 - Procurement Analyst
    # --------------------------------------------------------------
    analyst_agent = Agent(
        role="Procurement Analyst",
        goal=(
            "Compare products against the company's requirements and "
            "priorities, then rank the available options using only "
            "the supplied evidence."
        ),
        backstory=(
            "You are a procurement analyst who evaluates price, "
            "specifications, warranty, and overall business value."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Agent 3 - Report Writer
    # --------------------------------------------------------------
    report_agent = Agent(
        role="Procurement Report Writer",
        goal=(
            "Create a professional procurement report in HTML that "
            "is clear enough for a manager to review and approve."
        ),
        backstory=(
            "You are an executive report writer who turns procurement "
            "analysis into concise, professional business reports."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # --------------------------------------------------------------
    # Task 1
    # --------------------------------------------------------------
    research_task = Task(
        description=(
            "Company requirements:\n"
            f"{context_json}\n\n"
            "Collected online product evidence:\n"
            f"{research_json}\n\n"
            "Create a factual shortlist of up to 5 products. For each "
            "product include: product name, source, URL, price if "
            "explicitly available, specifications if explicitly "
            "available, warranty if explicitly available, and any "
            "missing information. Do not invent values."
        ),
        expected_output=(
            "A structured shortlist of up to 5 products with factual "
            "evidence and clearly marked missing information."
        ),
        agent=research_agent,
    )

    # --------------------------------------------------------------
    # Task 2
    # --------------------------------------------------------------
    analysis_task = Task(
        description=(
            "Using the research shortlist from the previous task, "
            "compare the products against these requirements:\n"
            f"{context_json}\n\n"
            f"Priorities: {company_context['priority_order']}\n"
            f"Budget per unit: "
            f"${company_context['budget_per_unit_usd']}\n\n"
            "Rank the products from best to lowest option. Explain "
            "briefly why each product received its rank. Clearly flag "
            "products whose price or required specifications could "
            "not be verified."
        ),
        expected_output=(
            "A ranked procurement analysis with a clear winner, "
            "alternatives, and reasons."
        ),
        agent=analyst_agent,
        context=[research_task],
    )

    # --------------------------------------------------------------
    # Task 3
    # --------------------------------------------------------------
    report_task = Task(
        description=(
            "Write the final procurement report using the research "
            "and analysis from the previous tasks.\n\n"
            "The report MUST be complete HTML beginning with "
            "<!DOCTYPE html> and containing <html>, <head>, and <body>.\n\n"
            "Include:\n"
            "1. Report title\n"
            "2. Company and procurement request summary\n"
            "3. Executive Summary\n"
            "4. Comparison table with Product, Source, Price, "
            "Key Specifications, Warranty, and Recommendation\n"
            "5. Final Recommendation\n"
            "6. Rationale\n"
            "7. Important verification notes\n"
            f"8. Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            "Use clean inline CSS and a professional business layout. "
            "Do not invent any product facts. If a value is unknown, "
            "write 'Not verified'."
        ),
        expected_output=(
            "Complete ready-to-render HTML procurement report."
        ),
        agent=report_agent,
        context=[analysis_task],
    )

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


# ----------------------------------------------------------------------
# 8) Helpers
# ----------------------------------------------------------------------
def clean_html_output(result) -> str:
    html_output = str(result).strip()

    if "```html" in html_output:
        html_output = (
            html_output.split("```html", 1)[1]
            .split("```", 1)[0]
            .strip()
        )
    elif "```HTML" in html_output:
        html_output = (
            html_output.split("```HTML", 1)[1]
            .split("```", 1)[0]
            .strip()
        )
    elif "```" in html_output:
        html_output = (
            html_output.split("```", 1)[1]
            .split("```", 1)[0]
            .strip()
        )

    return html_output


# ----------------------------------------------------------------------
# 9) Main Page Logic
# ----------------------------------------------------------------------
if run_button:
    if not groq_key or not tavily_key:
        st.error(
            "Please enter both GROQ_API_KEY and TAVILY_API_KEY "
            "in the sidebar."
        )
        st.stop()

    company_context = {
        "company_name": company_name,
        "procurement_need": procurement_need,
        "quantity": int(quantity),
        "budget_per_unit_usd": float(budget),
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

    try:
        with st.status(
            "🤖 Agents at work... This may take about a minute.",
            expanded=True,
        ) as status:

            st.write("🔍 Search Agent is looking for products...")

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
                    "Try a broader product name or check the Tavily key."
                )
                st.stop()

            st.write(
                f"✅ Found {len(products)} product sources."
            )

            st.write(
                "📄 Collecting product-page information..."
            )

            research_data = collect_product_data(products)

            st.write(
                "🧠 CrewAI Research Agent is organizing the evidence..."
            )

            result = run_crew(
                company_context,
                groq_key,
                research_data,
            )

            status.update(
                label="✅ Report is ready!",
                state="complete",
            )

        html_output = clean_html_output(result)

        st.subheader("📄 Final Report")

        st.components.v1.html(
            html_output,
            height=700,
            scrolling=True,
        )

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

    except Exception as exc:
        error_text = str(exc)

        st.error("❌ The application could not complete the workflow.")

        if "429" in error_text or "rate limit" in error_text.lower():
            st.warning(
                "Groq rate limit was reached. The app now uses the "
                "lighter llama-3.1-8b-instant model and only three "
                "CrewAI LLM calls, but the API key can still have "
                "its own usage limit. Wait briefly and run again."
            )
        elif "401" in error_text or "authentication" in error_text.lower():
            st.warning(
                "The Groq API key was rejected. Create/check the key "
                "and paste the complete key into GROQ_API_KEY."
            )
        elif "tavily" in error_text.lower():
            st.warning(
                "The Tavily request failed. Check TAVILY_API_KEY "
                "and try again."
            )

        st.exception(exc)

else:
    st.info(
        "Fill in your company details and API keys in the sidebar, "
        "then click 'Run Agents'."
    )
