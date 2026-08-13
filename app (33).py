"""
Multi-Agent Procurement Assistant - Streamlit App
مصمم للنشر المجاني والعالمي على Streamlit Community Cloud.

الموديل: Groq (مجاني - Llama 3.3) بدل OpenAI المدفوع
البحث: Tavily (له باقة مجانية)
السكرابينج: أداة مجانية مبنية بـ requests + BeautifulSoup (من غير أي API مدفوع)
"""

import os
import json
import requests
from datetime import datetime

import streamlit as st
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import Type

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tavily import TavilyClient

# ----------------------------------------------------------------------
# 1) إعداد الصفحة
# ----------------------------------------------------------------------
st.set_page_config(page_title="مساعد المشتريات الذكي", page_icon="🤖", layout="wide")

st.title("🤖 مساعد المشتريات متعدد الوكلاء (Multi-Agent Procurement Assistant)")
st.caption("مبني بـ CrewAI — يبحث، يقارن، ويطلع تقرير مشتريات جاهز، وكله ببلاش (Free Tier).")

# ----------------------------------------------------------------------
# 2) الشريط الجانبي: مفاتيح API (مجانية) + بيانات الشركة
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 مفاتيح API (مجانية)")
    st.markdown(
        "- **Groq**: احصل على مفتاح مجاني من [console.groq.com/keys](https://console.groq.com/keys)\n"
        "- **Tavily**: احصل على مفتاح مجاني من [app.tavily.com](https://app.tavily.com)"
    )

    groq_key = st.text_input(
        "GROQ_API_KEY",
        value=st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "",
        type="password",
    )
    tavily_key = st.text_input(
        "TAVILY_API_KEY",
        value=st.secrets.get("TAVILY_API_KEY", "") if hasattr(st, "secrets") else "",
        type="password",
    )

    st.divider()
    st.header("🏢 بيانات الشركة والطلب")

    company_name = st.text_input("اسم الشركة", "Constant Tech Solutions")
    procurement_need = st.text_input("المنتج المطلوب", "أجهزة Laptop للمهندسين")
    quantity = st.number_input("الكمية", min_value=1, value=15)
    budget = st.number_input("الميزانية للوحدة (USD)", min_value=1, value=1500)
    must_have = st.text_area(
        "المواصفات الأساسية (سطر لكل مواصفة)",
        "RAM 16GB أو أكتر\nSSD 512GB أو أكتر\nمعالج Intel i7 / Ryzen 7\nضمان سنتين على الأقل",
    )
    priority = st.text_input(
        "أولويات المقارنة (مفصولة بفاصلة)", "Price, Specifications, Warranty, Brand Reputation"
    )

    run_button = st.button("🚀 شغّل الوكلاء", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# 3) الأدوات (Tools) - مجانية بالكامل
# ----------------------------------------------------------------------
class TavilySearchInput(BaseModel):
    query: str = Field(..., description="نص البحث عن المنتج أو السعر أو المواصفات")


class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = (
        "أداة بحث على الإنترنت لإيجاد المنتجات، الموردين، الأسعار، والمواصفات. "
        "استخدمها بإدخال جملة بحث واضحة."
    )
    args_schema: Type[BaseModel] = TavilySearchInput
    tavily_api_key: str = ""

    def _run(self, query: str) -> str:
        client = TavilyClient(api_key=self.tavily_api_key)
        results = client.search(query=query, search_depth="advanced", max_results=5)
        formatted = []
        for r in results.get("results", []):
            formatted.append(
                f"- العنوان: {r.get('title')}\n  الرابط: {r.get('url')}\n  ملخص: {r.get('content')[:400]}"
            )
        return "\n\n".join(formatted) if formatted else "لم يتم العثور على نتائج."


class SimpleScrapeInput(BaseModel):
    url: str = Field(..., description="رابط صفحة المنتج المراد استخراج بياناتها")


class SimpleScraperTool(BaseTool):
    name: str = "simple_scraper"
    description: str = (
        "أداة مجانية لاستخراج النص من صفحة ويب (بدون أي API مدفوع). "
        "استخدمها بإدخال رابط صفحة المنتج."
    )
    args_schema: Type[BaseModel] = SimpleScrapeInput

    def _run(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ProcurementBot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:3000]
        except Exception as e:
            return f"تعذر استخراج بيانات من {url}: {e}"


# ----------------------------------------------------------------------
# 4) بناء وتشغيل الـ Crew
# ----------------------------------------------------------------------
def run_crew(company_context: dict, groq_key: str, tavily_key: str):
    os.environ["GROQ_API_KEY"] = groq_key

    llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.3)

    search_tool = TavilySearchTool(tavily_api_key=tavily_key)
    scrape_tool = SimpleScraperTool()

    search_agent = Agent(
        role="Product Search Specialist",
        goal="إيجاد أفضل المنتجات والموردين المتاحين اللي يطابقوا احتياج الشركة",
        backstory="خبير بحث عن المنتجات والأسعار أونلاين، بيعرف يفلتر النتائج ويجيب أنسب الروابط.",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    scraper_agent = Agent(
        role="Data Collection Specialist",
        goal="استخراج بيانات دقيقة (سعر، مواصفات، ضمان) من صفحات المنتجات",
        backstory="متخصص في استخراج البيانات المنظمة من صفحات الويب.",
        tools=[scrape_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst_agent = Agent(
        role="Procurement Analyst",
        goal="مقارنة وترتيب المنتجات حسب السعر والمواصفات والقيمة",
        backstory="محلل مشتريات متمرّس، بيقارن العروض بموضوعية وبيدي توصية مبنية على بيانات.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    report_agent = Agent(
        role="Procurement Report Writer",
        goal="كتابة تقرير Procurement احترافي بصيغة HTML",
        backstory="كاتب تقارير تنفيذية، بيحوّل التحليلات لتقرير منظم وواضح.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    context_str = json.dumps(company_context, ensure_ascii=False, indent=2)

    search_task = Task(
        description=(
            f"بناءً على احتياج الشركة التالي:\n{context_str}\n\n"
            "ابحث عن 5 منتجات على الأقل تطابق المواصفات المطلوبة، من مصادر مختلفة. "
            "اجمع اسم المنتج، الرابط، والمصدر."
        ),
        expected_output="قائمة بـ 5 منتجات على الأقل، كل واحد فيهم عنوان + رابط + مصدر.",
        agent=search_agent,
    )

    scrape_task = Task(
        description=(
            "من روابط المنتجات اللي جابها Search Agent، استخرج لكل منتج: "
            "السعر الحالي، المواصفات، ومدة الضمان لو متاحة."
        ),
        expected_output="جدول بيانات منظم لكل منتج: الاسم، السعر، المواصفات، الضمان، الرابط.",
        agent=scraper_agent,
        context=[search_task],
    )

    analysis_task = Task(
        description=(
            f"قارن المنتجات بناءً على أولويات الشركة: {company_context['priority_order']} "
            f"والميزانية: {company_context['budget_per_unit_usd']} دولار للوحدة. "
            "رتّب المنتجات من الأفضل للأقل قيمةً مع توضيح السبب."
        ),
        expected_output="ترتيب نهائي للمنتجات مع تبرير مختصر لكل ترتيب.",
        agent=analyst_agent,
        context=[scrape_task],
    )

    report_task = Task(
        description=(
            "اكتب تقرير Procurement نهائي بصيغة HTML كامل (بما فيه <html><head><body>)، "
            "يتضمن: عنوان، ملخص تنفيذي، جدول مقارنة، توصية نهائية مع الأسباب، وتاريخ التقرير. "
            "استخدم CSS بسيط داخل <style> عشان الشكل يبقى احترافي."
        ),
        expected_output="كود HTML كامل وجاهز للعرض والتحميل مباشرة.",
        agent=report_agent,
        context=[analysis_task],
    )

    crew = Crew(
        agents=[search_agent, scraper_agent, analyst_agent, report_agent],
        tasks=[search_task, scrape_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()


# ----------------------------------------------------------------------
# 5) منطق الصفحة الرئيسية
# ----------------------------------------------------------------------
if run_button:
    if not groq_key or not tavily_key:
        st.error("من فضلك دخّل مفتاحي GROQ_API_KEY و TAVILY_API_KEY في الشريط الجانبي (مجانيين).")
    else:
        company_context = {
            "company_name": company_name,
            "procurement_need": procurement_need,
            "quantity": quantity,
            "budget_per_unit_usd": budget,
            "must_have_specs": [s.strip() for s in must_have.split("\n") if s.strip()],
            "priority_order": [p.strip() for p in priority.split(",") if p.strip()],
        }

        with st.status("🤖 الوكلاء شغّالين... ده ممكن ياخد دقيقة كام", expanded=True) as status:
            st.write("🔍 Search Agent بيدور على المنتجات...")
            result = run_crew(company_context, groq_key, tavily_key)
            status.update(label="✅ التقرير جاهز!", state="complete")

        html_output = str(result)
        if "```html" in html_output:
            html_output = html_output.split("```html")[1].split("```")[0].strip()
        elif "```" in html_output:
            html_output = html_output.split("```")[1].split("```")[0].strip()

        st.subheader("📄 التقرير النهائي")
        st.components.v1.html(html_output, height=600, scrolling=True)

        filename = f"procurement_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        st.download_button(
            "⬇️ تحميل التقرير (HTML)",
            data=html_output,
            file_name=filename,
            mime="text/html",
            use_container_width=True,
        )
else:
    st.info("املأ بيانات الشركة والمفاتيح المجانية في الشريط الجانبي، وبعدين دوس 'شغّل الوكلاء'.")
