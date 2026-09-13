"""Streamlit entry point for the AI Company Research Assistant.

This iteration includes website crawling and evidence-grounded LLM analysis.
"""

import streamlit as st

from crawler import crawl_website, normalize_url
from llm_analyzer import analyze_company


APP_TITLE = "AI Company Research Assistant"
APP_SUBTITLE = "Turn company websites into structured business intelligence."


def render_evidence_list(
    title: str,
    items: list[dict],
    source_map: dict[int, dict],
) -> None:
    """Render conclusions simply, with source details available on demand."""
    st.subheader(title)
    if items:
        for index, item in enumerate(items, start=1):
            st.write(f"• {item['text']}")
            source_ids = item["source_ids"]
            with st.expander(f"Sources for item {index} ({len(source_ids)})"):
                for source_id in source_ids:
                    source = source_map.get(source_id)
                    if source:
                        st.caption(f"SOURCE {source_id} · {source['title']}")
                        st.write(source["url"])
    else:
        st.caption("Not identified from the available website sources.")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --research-ink: #14213d;
            --research-muted: #5f6b7a;
            --research-accent: #2563eb;
            --research-border: #dbe3ee;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.07), transparent 26rem),
                #ffffff;
        }

        .block-container {
            max-width: 880px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }

        .eyebrow {
            color: var(--research-accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }

        .app-subtitle {
            color: var(--research-muted);
            font-size: 1.08rem;
            line-height: 1.7;
            margin: 0.25rem 0 2rem;
        }

        .workflow-step {
            align-items: center;
            border-bottom: 1px solid var(--research-border);
            display: flex;
            gap: 0.9rem;
            padding: 0.9rem 0;
        }

        .workflow-step:last-child {
            border-bottom: 0;
        }

        .step-number {
            align-items: center;
            background: #eaf1ff;
            border-radius: 999px;
            color: var(--research-accent);
            display: inline-flex;
            flex: 0 0 2rem;
            font-size: 0.84rem;
            font-weight: 700;
            height: 2rem;
            justify-content: center;
        }

        .step-copy strong {
            color: var(--research-ink);
            display: block;
            font-size: 0.96rem;
        }

        .step-copy span {
            color: var(--research-muted);
            font-size: 0.86rem;
        }

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--research-border);
            border-radius: 1rem;
            box-shadow: 0 16px 40px rgba(20, 33, 61, 0.07);
            padding: 0.5rem;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: var(--research-accent);
            border: 0;
            font-weight: 650;
            min-height: 2.85rem;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: #1d4ed8;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 2.5rem;
                padding-left: 1.1rem;
                padding-right: 1.1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Business research workspace</div>', unsafe_allow_html=True)
st.title(APP_TITLE)
st.markdown(f'<p class="app-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

with st.form("company_research_form"):
    website_input = st.text_input(
        "Website URL",
        placeholder="https://www.siemens.com",
        help="Enter the official website of the company you want to research.",
    )
    analyze_submitted = st.form_submit_button(
        "Analyze Company",
        type="primary",
        use_container_width=True,
    )

if analyze_submitted:
    st.session_state.pop("crawl_result", None)
    st.session_state.pop("analysis_result", None)
    try:
        normalized_url = normalize_url(website_input)
    except ValueError:
        st.error("Enter a valid company website, such as https://www.siemens.com.")
    else:
        st.session_state["company_website"] = normalized_url

        with st.status("Analyzing company website...", expanded=True) as crawl_status:

            def report_progress(stage: str, message: str) -> None:
                del stage
                crawl_status.write(message)

            crawl_result = crawl_website(
                normalized_url,
                max_pages=10,
                progress_callback=report_progress,
            )

            analysis_result = analyze_company(
                crawl_result,
                progress_callback=report_progress,
            )

            if analysis_result["status"] == "success":
                coverage_level = analysis_result.get("source_coverage", {}).get(
                    "level", "limited"
                )
                label = (
                    "Company analysis completed — manual review recommended"
                    if coverage_level != "sufficient"
                    else "Company analysis completed"
                )
                crawl_status.update(label=label, state="complete")
            elif analysis_result["status"] == "configuration_error":
                crawl_status.update(label="AI analysis unavailable", state="error")
            elif crawl_result["status"] == "failed":
                crawl_status.update(label="Website crawl failed", state="error")
            else:
                crawl_status.update(
                    label="Analysis needs manual review",
                    state="complete",
                )

        st.session_state["crawl_result"] = crawl_result
        st.session_state["analysis_result"] = analysis_result
else:
    st.caption("Enter a company website to prepare a new research task.")

st.divider()

st.subheader("Analysis workflow")
crawl_result = st.session_state.get("crawl_result")
analysis_result = st.session_state.get("analysis_result")
if crawl_result:
    homepage_status = "Completed" if crawl_result.get("homepage") else "Failed"
    if not crawl_result.get("homepage"):
        discovery_status = "Not started"
    elif crawl_result.get("errors") or crawl_result.get("warnings"):
        discovery_status = "Completed with warnings"
    else:
        discovery_status = "Completed"
else:
    homepage_status = "Waiting to start"
    discovery_status = "Waiting to start"

if not analysis_result:
    extraction_status = "Waiting to start"
    validation_status = "Waiting to start"
elif analysis_result["status"] == "success":
    extraction_status = "Completed"
    validation_status = "Completed"
elif analysis_result["status"] == "configuration_error":
    extraction_status = "Blocked — API key required"
    validation_status = "Not started"
elif analysis_result.get("reason") in {
    "no_source_content",
    "insufficient_source_content",
}:
    extraction_status = "Skipped — insufficient source content"
    validation_status = "Manual review required"
elif analysis_result.get("reason") == "ai_request_failed":
    extraction_status = "Failed"
    validation_status = "Not completed"
else:
    extraction_status = "Completed with retry"
    validation_status = "Manual review required"

st.markdown(
    f"""
    <div class="workflow-step">
        <span class="step-number">1</span>
        <div class="step-copy"><strong>Fetch homepage</strong><span>{homepage_status}</span></div>
    </div>
    <div class="workflow-step">
        <span class="step-number">2</span>
        <div class="step-copy"><strong>Discover relevant pages</strong><span>{discovery_status}</span></div>
    </div>
    <div class="workflow-step">
        <span class="step-number">3</span>
        <div class="step-copy"><strong>Extract business information</strong><span>{extraction_status}</span></div>
    </div>
    <div class="workflow-step">
        <span class="step-number">4</span>
        <div class="step-copy"><strong>Validate results</strong><span>{validation_status}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if analysis_result:
    st.divider()

    source_coverage = analysis_result.get("source_coverage", {})
    coverage_level = source_coverage.get("level", "unknown")
    if coverage_level == "limited":
        st.warning(
            "Limited source coverage: the available website evidence is incomplete, "
            "so any resulting analysis requires manual review."
        )
    elif coverage_level == "insufficient":
        st.warning(
            "Insufficient source coverage: not enough readable business evidence was "
            "available for AI analysis."
        )

    if analysis_result["status"] == "success":
        analysis_data = analysis_result["data"]
        source_map = {
            source["source_id"]: source
            for source in analysis_result.get("sources", [])
        }
        st.subheader("Company Overview")
        with st.container(border=True):
            summary_columns = st.columns([3, 1])
            with summary_columns[0]:
                st.caption("COMPANY")
                st.markdown(f"### {analysis_data['company_name']}")
                st.write(analysis_data["company_overview"])
            with summary_columns[1]:
                st.metric("Confidence", analysis_data["confidence"].title())
            st.caption("Results are generated only from the crawled website content.")
            st.caption(
                f"Crawl status: {analysis_result.get('crawl_status', 'unknown').title()} · "
                f"Source coverage: {coverage_level.title()}"
            )

        result_columns = st.columns(2)
        with result_columns[0]:
            render_evidence_list(
                "Core Products",
                analysis_data["core_products"],
                source_map,
            )
        with result_columns[1]:
            render_evidence_list(
                "Services & Solutions",
                analysis_data["services_solutions"],
                source_map,
            )

        render_evidence_list(
            "Target Industries",
            analysis_data["target_industries"],
            source_map,
        )

        st.subheader("Business Keywords")
        if analysis_data["business_keywords"]:
            st.write(" · ".join(analysis_data["business_keywords"]))
        else:
            st.caption("Not identified from the available website sources.")

        render_evidence_list(
            "Key Findings",
            analysis_data["key_findings"],
            source_map,
        )
    elif analysis_result["status"] == "configuration_error":
        st.subheader("AI Analysis")
        if analysis_result.get("reason") == "missing_api_key":
            st.warning(
                "AI analysis is not configured. Set the ZHIPU_API_KEY environment "
                "variable and run the analysis again."
            )
        else:
            st.warning("The AI client could not be initialized.")
        with st.expander("Technical details"):
            st.write(analysis_result.get("details", "Configuration is incomplete."))
    else:
        st.subheader("AI Analysis Needs Manual Review")
        if analysis_result.get("reason") == "insufficient_source_content":
            st.warning(
                "The crawler did not collect enough business content for a reliable "
                "AI analysis, so no model request was made."
            )
        else:
            st.warning(
                "A reliable structured analysis could not be produced from the "
                "available sources."
            )
        st.caption(analysis_result.get("details", "Manual review is required."))

if crawl_result:
    st.divider()

    homepage = crawl_result.get("homepage")
    pages = crawl_result.get("pages", [])
    errors = crawl_result.get("errors", [])
    warnings = crawl_result.get("warnings", [])

    if homepage:
        st.subheader("Crawling Preview")
        page_count = 1 + len(pages)
        metric_columns = st.columns(2)
        metric_columns[0].metric("Pages crawled", page_count)
        metric_columns[1].metric("Crawl errors", len(errors))

        with st.container(border=True):
            st.caption("HOMEPAGE")
            st.markdown(f"**{homepage['title']}**")
            st.code(homepage["url"], language=None)
            st.write(homepage["content"][:500])

        diagnostics = homepage.get("diagnostics", {})
        if diagnostics:
            with st.expander("Homepage crawl diagnostics"):
                st.write(f"Readable characters: {diagnostics.get('visible_text_chars', 0)}")
                st.write(f"Anchor tags: {diagnostics.get('anchor_count', 0)}")
                st.write(f"Iframes: {diagnostics.get('iframe_count', 0)}")
                st.write(f"Scripts: {diagnostics.get('script_count', 0)}")
                st.write(
                    "Script-based navigation detected: "
                    f"{'Yes' if diagnostics.get('script_navigation_detected') else 'No'}"
                )

        for index, page in enumerate(pages, start=1):
            expander_title = f"{index}. {page['type'].title()} — {page['title']}"
            with st.expander(expander_title):
                st.caption(f"Page type: {page['type']}")
                st.code(page["url"], language=None)
                if page.get("description"):
                    st.markdown("**SEO description**")
                    st.write(page["description"])
                st.markdown("**Content preview**")
                st.write(page["content"][:500])

        st.subheader("Sources Crawled")
        ai_sources = analysis_result.get("sources", []) if analysis_result else []
        if ai_sources:
            for source in ai_sources:
                st.write(
                    f"SOURCE {source['source_id']} · {source['type'].title()} · "
                    f"{source['title']}"
                )
                st.caption(source["url"])
        else:
            st.markdown(f"- [Homepage]({homepage['url']})")
            for index, page in enumerate(pages, start=1):
                st.markdown(
                    f"- [{index}. {page['type'].title()} page]({page['url']})"
                )
    else:
        st.error("The homepage could not be crawled.")

    if warnings:
        st.subheader("Crawl Warnings")
        for warning in warnings:
            st.warning(warning["message"])
            if warning.get("details"):
                st.caption(warning["details"])

    if errors:
        st.subheader("Crawl Errors")
        st.caption("A failed subpage does not stop the remaining crawl.")
        for index, error in enumerate(errors, start=1):
            label = f"{index}. {error['type']} — {error['message']}"
            with st.expander(label, expanded=homepage is None):
                st.write(error["url"])
                if error.get("status_code"):
                    st.write(f"HTTP status: {error['status_code']}")
                if error.get("technical_details"):
                    st.markdown("**Technical details**")
                    st.code(error["technical_details"], language=None)

    st.caption("The crawling preview and source list contain only directly collected website content.")
