
import os
import json
import time
import re
from datetime import datetime
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from tavily import TavilyClient
from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# 1) PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi-Agent Procurement Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Procurement Assistant")
st.caption(
    "CrewAI + Tavily + Groq | Lightweight free-tier version "
    "designed to minimize Groq token usage."
)


# ============================================================
# 2) SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🔑 API Keys")

    st.markdown(
        "- **Groq:** `console.groq.com/keys`\n"
        "- **Tavily:** `app.tavily.com`"
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
        "🚀 Run Procurement Analysis",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 3) HELPERS
# ============================================================
def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "Unknown"


def clean_text(text: str, limit: int = 900) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def extract_price(text: str) -> str:
    if not text:
        return "Not verified"

    patterns = [
        r"(?:USD|US\$|\$)\s?\d{2,5}(?:[,.]\d{2})?",
        r"\d{2,5}(?:[,.]\d{2})?\s?(?:USD|US\$)",
        r"(?:price|cost|sale price)\s*[:\-]?\s*(?:USD|US\$|\$)?\s?\d{2,5}(?:[,.]\d{2})?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(0)[:80]

    return "Not verified"


def scrape_product_page(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/131 Safari/537.36"
                )
            },
            timeout=10,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            ["script", "style", "nav", "footer", "header", "noscript",
             "svg", "form"]
        ):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:3500]

    except Exception as exc:
        return f"Page could not be collected: {exc}"


# ============================================================
# 4) TAVILY SEARCH
# ============================================================
def search_products(tavily_key: str, context: dict) -> list:
    client = TavilyClient(api_key=tavily_key)

    specs = ", ".join(context["must_have_specs"])

    query = (
        f"{context['procurement_need']} "
        f"business professional laptop "
        f"under ${context['budget_per_unit_usd']} "
        f"{specs}"
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
        if not url:
            continue

        products.append(
            {
                "name": item.get("title", "Unknown product"),
                "url": url,
                "source": domain_from_url(url),
                "snippet": clean_text(item.get("content", ""), 900),
            }
        )

    # Remove duplicate URLs while preserving order.
    unique = []
    seen = set()

    for product in products:
        if product["url"] not in seen:
            unique.append(product)
            seen.add(product["url"])

    return unique[:5]


# ============================================================
# 5) COLLECT SMALL EVIDENCE PACK
# ============================================================
def collect_evidence(products: list) -> list:
    evidence = []

    for product in products:
        page_text = scrape_product_page(product["url"])

        combined = f"{product['snippet']} {page_text}"

        evidence.append(
            {
                "name": product["name"],
                "source": product["source"],
                "url": product["url"],
                "price_hint": extract_price(combined),
                # Keep the LLM input deliberately small.
                "evidence": clean_text(combined, 1200),
            }
        )

    return evidence


# ============================================================
# 6) ONE LIGHTWEIGHT CREWAI CALL
# ============================================================
def run_procurement_agent(
    groq_key: str,
    context: dict,
    evidence: list,
) -> dict:
    """
    IMPORTANT:
    - No CrewAI tools.
    - No native tool calling.
    - No max_iter argument.
    - One LLM call only.
    - Small input + small output.
    This avoids the errors seen with Groq free-tier TPM and
    the old max_iter/native-tools path.
    """

    os.environ["GROQ_API_KEY"] = groq_key

    llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0.1,
        max_tokens=1200,
    )

    agent = Agent(
        role="Procurement Decision Analyst",
        goal=(
            "Analyze the supplied procurement evidence and return a "
            "short factual ranking. Never invent prices, specifications, "
            "warranty details, or product facts."
        ),
        backstory=(
            "You are a procurement analyst. You distinguish verified "
            "evidence from missing information and prioritize the buyer's "
            "stated priorities."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    compact_context = {
        "company": context["company_name"],
        "product_needed": context["procurement_need"],
        "quantity": context["quantity"],
        "budget_per_unit_usd": context["budget_per_unit_usd"],
        "must_have_specs": context["must_have_specs"],
        "priorities": context["priority_order"],
    }

    prompt = (
        "Return ONLY valid JSON. No markdown. No code fences.\n\n"
        "Required JSON schema:\n"
        "{"
        '"executive_summary":"...",'
        '"recommendation":"...",'
        '"recommendation_reason":"...",'
        '"ranked_products":['
        "{"
        '"rank":1,'
        '"name":"...",'
        '"source":"...",'
        '"url":"...",'
        '"price":"...",'
        '"fit":"Good|Partial|Poor|Not verified",'
        '"key_points":["..."],'
        '"risks":["..."]'
        "}"
        "]"
        "}\n\n"
        "Buyer requirements:\n"
        f"{json.dumps(compact_context, ensure_ascii=False)}\n\n"
        "Product evidence:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "1. Rank only products present in the evidence.\n"
        "2. Never invent missing values.\n"
        "3. If price/spec/warranty is not supported, say Not verified.\n"
        "4. Respect the buyer's priority order.\n"
        "5. Keep each key_points list to at most 3 short items.\n"
        "6. Keep risks to at most 2 short items per product.\n"
        "7. Return no extra text outside the JSON object."
    )

    task = Task(
        description=prompt,
        expected_output="One valid JSON object following the supplied schema.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # A short retry is useful if the previous minute's TPM window
    # has not fully reset yet.
    last_error = None

    for attempt in range(2):
        try:
            result = crew.kickoff()
            raw = getattr(result, "raw", str(result)).strip()

            # Remove accidental code fences.
            raw = raw.replace("```json", "").replace("```", "").strip()

            # Parse the first JSON object in the response.
            decoder = json.JSONDecoder()
            start = raw.find("{")

            if start == -1:
                raise ValueError("The model did not return a JSON object.")

            parsed, _ = decoder.raw_decode(raw[start:])

            if not isinstance(parsed, dict):
                raise ValueError("The model returned an invalid JSON structure.")

            return parsed

        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()

            if (
                "rate limit" in error_text
                or "tokens per minute" in error_text
                or "429" in error_text
            ) and attempt == 0:
                time.sleep(20)
                continue

            raise last_error


# ============================================================
# 7) HTML REPORT
# ============================================================
def esc(value) -> str:
    import html
    return html.escape(str(value or "Not verified"))


def render_report(context: dict, analysis: dict) -> str:
    products = analysis.get("ranked_products", [])

    rows = []

    for p in products:
        points = "".join(
            f"<li>{esc(x)}</li>"
            for x in p.get("key_points", [])[:3]
        )

        risks = "".join(
            f"<li>{esc(x)}</li>"
            for x in p.get("risks", [])[:2]
        )

        rows.append(
            f"""
            <tr>
                <td><strong>#{esc(p.get("rank"))}</strong></td>
                <td>
                    <strong>{esc(p.get("name"))}</strong>
                    <div class="small">{esc(p.get("source"))}</div>
                </td>
                <td>{esc(p.get("price"))}</td>
                <td>{esc(p.get("fit"))}</td>
                <td>
                    <ul>{points or "<li>Not verified</li>"}</ul>
                </td>
                <td>
                    <ul>{risks or "<li>None reported</li>"}</ul>
                </td>
                <td>
                    <a href="{esc(p.get("url"))}" target="_blank">
                        Product page
                    </a>
                </td>
            </tr>
            """
        )

    rows_html = "\n".join(rows)

    specs_html = "".join(
        f"<li>{esc(x)}</li>"
        for x in context["must_have_specs"]
    )

    priorities_html = ", ".join(
        esc(x) for x in context["priority_order"]
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Procurement Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f4f6f8;
    color: #1f2937;
}}
.container {{
    max-width: 1400px;
    margin: 30px auto;
    padding: 0 20px;
}}
.card {{
    background: white;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
}}
h1 {{ margin-top: 0; }}
h2 {{ margin-top: 0; }}
.small {{ color: #6b7280; font-size: 12px; margin-top: 4px; }}
table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
}}
th, td {{
    border: 1px solid #e5e7eb;
    padding: 10px;
    vertical-align: top;
    text-align: left;
}}
th {{ background: #f3f4f6; }}
li {{ margin-bottom: 5px; }}
.badge {{
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: #eef2ff;
}}
a {{ text-decoration: none; }}
</style>
</head>
<body>
<div class="container">

<div class="card">
    <h1>🤖 Procurement Decision Report</h1>
    <p><strong>Company:</strong> {esc(context["company_name"])}</p>
    <p><strong>Product Needed:</strong> {esc(context["procurement_need"])}</p>
    <p><strong>Quantity:</strong> {esc(context["quantity"])}</p>
    <p><strong>Budget per Unit:</strong> ${esc(context["budget_per_unit_usd"])}</p>
    <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d")}</p>
</div>

<div class="card">
    <h2>Executive Summary</h2>
    <p>{esc(analysis.get("executive_summary"))}</p>
</div>

<div class="card">
    <h2>Buyer Requirements</h2>
    <ul>{specs_html}</ul>
    <p><strong>Priorities:</strong> {priorities_html}</p>
</div>

<div class="card">
    <h2>Final Recommendation</h2>
    <p><span class="badge">{esc(analysis.get("recommendation"))}</span></p>
    <p>{esc(analysis.get("recommendation_reason"))}</p>
</div>

<div class="card">
    <h2>Product Comparison</h2>
    <div style="overflow-x:auto;">
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Product</th>
                <th>Price</th>
                <th>Fit</th>
                <th>Key Points</th>
                <th>Risks / Missing Data</th>
                <th>Source</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
</div>

<div class="card">
    <h2>⚠️ Verification Note</h2>
    <p>
        Product information comes from online search results and publicly
        accessible product pages. Prices, stock, warranty terms and
        specifications should be verified with the supplier before issuing
        a purchase order.
    </p>
</div>

</div>
</body>
</html>
"""


# ============================================================
# 8) MAIN WORKFLOW
# ============================================================
if run_button:
    if not groq_key or not tavily_key:
        st.error("Please enter both GROQ_API_KEY and TAVILY_API_KEY.")
        st.stop()

    context = {
        "company_name": company_name.strip(),
        "procurement_need": procurement_need.strip(),
        "quantity": int(quantity),
        "budget_per_unit_usd": float(budget),
        "must_have_specs": [
            x.strip()
            for x in must_have.splitlines()
            if x.strip()
        ],
        "priority_order": [
            x.strip()
            for x in priority.split(",")
            if x.strip()
        ],
    }

    try:
        with st.status(
            "🤖 Procurement agents are working...",
            expanded=True,
        ) as status:

            st.write("🔍 Searching products with Tavily...")
            products = search_products(tavily_key, context)

            if not products:
                status.update(
                    label="❌ No products found",
                    state="error",
                )
                st.error(
                    "Tavily returned no product results. "
                    "Try a broader product name."
                )
                st.stop()

            st.write(f"✅ Found {len(products)} product sources.")

            st.write("📄 Collecting compact product evidence...")
            evidence = collect_evidence(products)

            st.write("🧠 CrewAI Procurement Analyst is ranking the options...")
            analysis = run_procurement_agent(
                groq_key,
                context,
                evidence,
            )

            status.update(
                label="✅ Procurement report is ready!",
                state="complete",
            )

        html_report = render_report(context, analysis)

        st.subheader("📄 Final Procurement Report")
        st.components.v1.html(
            html_report,
            height=760,
            scrolling=True,
        )

        filename = (
            f"procurement_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        )

        st.download_button(
            "⬇️ Download Report (HTML)",
            data=html_report,
            file_name=filename,
            mime="text/html",
            use_container_width=True,
        )

    except Exception as exc:
        error_text = str(exc)
        lower = error_text.lower()

        st.error("❌ The workflow could not be completed.")

        if (
            "rate limit" in lower
            or "tokens per minute" in lower
            or "429" in lower
        ):
