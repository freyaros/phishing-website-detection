import html
import ipaddress
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_url_model.joblib"
SAMPLES_PATH = BASE_DIR / "demo_samples.csv"
SUSPICIOUS_PATTERNS = (
    "account",
    "auth",
    "banking",
    "billing",
    "confirm",
    "login",
    "password",
    "paypal",
    "recover",
    "reset",
    "secure",
    "sign-in",
    "signin",
    "update",
    "verify",
    "webscr",
)
UNUSUAL_PUNCTUATION = frozenset("!$%^*_={}|;\"'<>#")

st.set_page_config(
    page_title="PHISHGUARD | URL Analysis",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        color-scheme: dark;
        --pg-bg: #070B14;
        --pg-surface: #0B1220;
        --pg-card: #101B2D;
        --pg-elevated: #142238;
        --pg-border: #263A52;
        --pg-cyan: #42E8F5;
        --pg-teal: #18C6B5;
        --pg-text: #F4F8FF;
        --pg-muted: #B2C2D8;
        --pg-green: #39E58C;
        --pg-red: #FF5964;
        --pg-amber: #FFC857;
    }
    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }
    .stApp { color: var(--pg-text); background: radial-gradient(ellipse at 50% 13%, rgba(24, 198, 181, .12), transparent 35%), radial-gradient(ellipse at 12% 28%, rgba(66, 232, 245, .07), transparent 26%), var(--pg-bg); }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; color: var(--pg-text); }
    [data-testid="stHeader"] { background: rgba(7, 11, 20, .92); }
    [data-testid="stSidebar"] { background: var(--pg-surface); border-right: 1px solid var(--pg-border); }
    .block-container { max-width: 1160px; margin: 0 auto; padding: 3.4rem 2rem 3rem; }
    p, label, li, [data-testid="stMarkdownContainer"] { color: var(--pg-text); font-size: 1rem; line-height: 1.6; }
    h1, h2, h3 { color: var(--pg-text); font-family: "Space Grotesk", Inter, ui-sans-serif, sans-serif; letter-spacing: 0; font-weight: 700; }
    h2 { font-size: 1.45rem; margin-top: 1rem; }
    h3 { font-size: 1.2rem; }
    code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .pg-brand-hero { display: flex; align-items: center; justify-content: center; gap: 1.35rem; margin: .15rem auto .9rem; }
    .pg-mark { width: 86px; height: 86px; flex: 0 0 86px; display: grid; place-items: center; border: 1px solid rgba(66,232,245,.72); border-radius: 23px; background: linear-gradient(145deg, rgba(20,34,56,.96), rgba(11,18,32,.96)); box-shadow: 0 0 34px rgba(66,232,245,.16), inset 0 0 24px rgba(24,198,181,.07); }
    .pg-mark svg { width: 53px; height: 53px; }
    .pg-brand { color: var(--pg-text); font-family: "Space Grotesk", Inter, ui-sans-serif, sans-serif; font-size: 3.15rem; font-weight: 700; line-height: 1; }
    .pg-subtitle { margin-top: .55rem; color: var(--pg-cyan); font-size: .88rem; font-weight: 700; letter-spacing: .12em; }
    .pg-hero-copy { position: relative; text-align: center; padding: 1rem 1rem 1.05rem; margin: 0 auto .8rem; overflow: hidden; border-radius: 24px; background: radial-gradient(ellipse at 50% 8%, rgba(66,232,245,.11), transparent 55%); }
    .pg-kicker { display: inline-flex; align-items: center; gap: .55rem; color: var(--pg-teal); font-size: .88rem; font-weight: 700; }
    .pg-kicker-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--pg-cyan); box-shadow: 0 0 12px rgba(66,232,245,.8); }
    .pg-hero-title { color: var(--pg-text); font-family: "Space Grotesk", Inter, ui-sans-serif, sans-serif; font-size: 3rem; line-height: 1.1; font-weight: 700; margin: .5rem 0 .45rem; }
    .pg-hero-title span { color: var(--pg-cyan); }
    .pg-hero-copy p { max-width: 760px; margin: 0 auto; color: var(--pg-muted); font-size: 1.12rem; line-height: 1.65; }
    .pg-info { max-width: 900px; margin: .2rem auto .9rem; padding: .65rem 1rem; color: var(--pg-muted); background: rgba(11,18,32,.82); border: 1px solid var(--pg-border); border-radius: 8px; text-align: center; font-size: .95rem; }
    .pg-scan-heading { display: flex; align-items: center; gap: .75rem; margin: 1.2rem 0 .6rem; }
    .pg-scan-icon { width: 38px; height: 38px; display: grid; place-items: center; flex: 0 0 38px; border: 1px solid rgba(66,232,245,.35); border-radius: 10px; background: rgba(66,232,245,.08); }
    .pg-scan-icon svg { width: 22px; height: 22px; }
    .pg-scan-title { color: var(--pg-text); font-family: "Space Grotesk", Inter, ui-sans-serif, sans-serif; font-size: 1.4rem; font-weight: 700; line-height: 1.25; }
    .pg-scan-caption { color: var(--pg-muted); font-size: .94rem; margin-top: .18rem; }
    div[data-testid="stForm"] { background: linear-gradient(145deg, rgba(16,27,45,.98), rgba(11,18,32,.98)); border: 1px solid var(--pg-border); border-radius: 14px; padding: 1rem 1.2rem .75rem; box-shadow: 0 18px 48px rgba(0,0,0,.28), inset 0 1px rgba(255,255,255,.025); }
    div[data-testid="stTextInput"] input { min-height: 3.65rem; padding: .7rem 1rem; background: #080F1D; color: var(--pg-text); border: 1px solid #38506D; border-radius: 9px; font-size: 1.02rem; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    div[data-testid="stTextInput"] input::placeholder { color: #A6B7CD; opacity: 1; }
    div[data-testid="stTextInput"] input:focus { border-color: var(--pg-cyan); box-shadow: 0 0 0 3px rgba(66,232,245,.18); }
    [data-baseweb="select"] > div { min-height: 2.9rem; background: var(--pg-surface); border-color: #38506D; color: var(--pg-text); border-radius: 8px; }
    [data-baseweb="popover"] [role="listbox"] { background: var(--pg-surface); }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button { min-height: 3.2rem; border-radius: 9px; font-size: .98rem; font-weight: 700; transition: transform .16s ease, filter .16s ease, box-shadow .16s ease; }
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] { background: linear-gradient(110deg, var(--pg-cyan), var(--pg-teal)); border: 1px solid var(--pg-cyan); color: #06131D; box-shadow: 0 7px 22px rgba(24,198,181,.17); }
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover { filter: brightness(1.08); color: #06131D; transform: translateY(-1px); box-shadow: 0 10px 28px rgba(24,198,181,.25); }
    div.stButton > button:not([kind="primary"]) { background: var(--pg-surface); border: 1px solid var(--pg-border); color: var(--pg-text); }
    div.stButton > button:not([kind="primary"]):hover { border-color: var(--pg-cyan); color: var(--pg-cyan); }
    button:focus-visible { outline: 2px solid var(--pg-cyan) !important; outline-offset: 3px; }
    div[data-testid="stMetric"] { min-height: 106px; background: var(--pg-card); border: 1px solid var(--pg-border); border-radius: 10px; padding: .85rem 1rem; overflow-wrap: anywhere; }
    [data-testid="stMetricLabel"] p { color: var(--pg-muted); font-size: .92rem; }
    [data-testid="stMetricValue"] { color: var(--pg-text); font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.35rem; }
    [data-testid="stCaptionContainer"] { color: var(--pg-muted); font-size: .9rem; }
    .result-card { border-radius: 11px; padding: 1.2rem 1.3rem; border: 1px solid; min-height: 96px; }
    .result-card.phishing { background: linear-gradient(110deg, #291720, #1C1722); border-color: #743642; border-left: 4px solid var(--pg-red); color: var(--pg-text); }
    .result-card.legitimate { background: linear-gradient(110deg, #12271F, #13231F); border-color: #32634B; border-left: 4px solid var(--pg-green); color: var(--pg-text); }
    .result-card strong { display: block; font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.3rem; margin-bottom: .35rem; }
    .result-card.phishing strong { color: var(--pg-red); }
    .result-card.legitimate strong { color: var(--pg-green); }
    .result-card span { color: var(--pg-muted); font-size: .97rem; line-height: 1.5; }
    .section-note { color: var(--pg-muted); font-size: .96rem; margin: -.25rem 0 .8rem; }
    .pg-risk-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(205px, 1fr)); gap: .8rem; margin: .8rem 0 1rem; }
    .pg-risk-item { min-width: 0; background: var(--pg-card); border: 1px solid var(--pg-border); border-radius: 10px; padding: .95rem 1rem; }
    .pg-risk-item h4 { color: var(--pg-muted); font: 600 .93rem Inter, ui-sans-serif, sans-serif; margin: 0 0 .45rem; }
    .pg-risk-value { color: var(--pg-cyan); font: 700 1.12rem "Space Grotesk", Inter, sans-serif; overflow-wrap: anywhere; margin-bottom: .35rem; }
    .pg-risk-description { color: var(--pg-muted); font-size: .9rem; line-height: 1.45; }
    div[data-testid="stAlert"] { background: var(--pg-surface); border: 1px solid var(--pg-border); color: var(--pg-text); border-radius: 9px; }
    div[data-testid="stAlert"] p { color: var(--pg-text); }
    div[data-testid="stDataFrame"] { border: 1px solid var(--pg-border); border-radius: 9px; overflow: hidden; }
    [data-testid="stExpander"] { background: var(--pg-surface); border: 1px solid var(--pg-border); border-radius: 9px; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--pg-text); font-weight: 650; }
    .pg-footer { border-top: 1px solid var(--pg-border); margin-top: 1.5rem; padding-top: 1rem; color: var(--pg-muted); font-size: .93rem; }
    @media (max-width: 760px) {
        .block-container { padding: 3.4rem 1rem 2rem; }
        .pg-brand-hero { gap: .85rem; }
        .pg-mark { width: 64px; height: 64px; flex-basis: 64px; border-radius: 18px; }
        .pg-mark svg { width: 41px; height: 41px; }
        .pg-brand { font-size: 2.15rem; }
        .pg-subtitle { font-size: .7rem; letter-spacing: .07em; }
        .pg-hero-copy { padding: 1rem .2rem 1.2rem; }
        .pg-hero-title { font-size: 2.25rem; }
        .pg-hero-copy p { font-size: 1rem; }
        div[data-testid="stForm"] { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
        <header class="pg-brand-hero">
      <div class="pg-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2.2 20.2 5.5v5.8c0 5.1-3.4 8.7-8.2 10.5C7.2 20 3.8 16.4 3.8 11.3V5.5L12 2.2Z" stroke="#42E8F5" stroke-width="1.25" stroke-linejoin="round"/>
                    <path d="m8.1 11.9 2.5 2.5 5.5-5.7" stroke="#18C6B5" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div class="pg-brand">PHISHGUARD</div>
                <div class="pg-subtitle">INTELLIGENT URL THREAT ANALYSIS</div>
      </div>
    </header>
        <section class="pg-hero-copy">
            <div class="pg-kicker"><span class="pg-kicker-dot"></span> LOCAL MACHINE-LEARNING ANALYSIS</div>
            <h1 class="pg-hero-title">Don't trust the link.<br><span>Verify it.</span></h1>
            <p>AI-powered URL analysis to help identify potentially malicious links before you interact with them.</p>
        </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
        '<div class="pg-info">We analyze URL text patterns only. Submitted websites are never visited or inspected, and model predictions may be incorrect.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def validate_hostname(hostname):
    try:
        ipaddress.ip_address(hostname)
        return
    except ValueError:
        pass

    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise ValueError("The hostname contains invalid international characters.") from error

    normalized_hostname = ascii_hostname[:-1] if ascii_hostname.endswith(".") else ascii_hostname
    if not normalized_hostname or len(normalized_hostname) > 253:
        raise ValueError("The hostname length is invalid.")

    labels = normalized_hostname.split(".")
    if any(
        len(label) > 63
        or not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?", label)
        for label in labels
    ):
        raise ValueError("The hostname contains an invalid label.")


def normalize_url(value):
    if not isinstance(value, str):
        raise ValueError("Enter a website URL as text.")

    value = value.strip()
    if not value:
        raise ValueError("Enter a URL to analyze.")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("The URL cannot contain spaces or control characters.")
    if "\\" in value or re.search(r"%(?![a-fA-F0-9]{2})", value):
        raise ValueError("The URL contains malformed characters or percent encoding.")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        if value.startswith("//"):
            raise ValueError("Enter a complete website URL, not a scheme-relative URL.")
        value = "http://" + value

    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as error:
        raise ValueError("The URL authority or port is malformed.") from error

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter an HTTP or HTTPS URL with a hostname.")
    if parsed.netloc.endswith(":"):
        raise ValueError("The URL contains an empty port.")

    validate_hostname(parsed.hostname)
    return value


def get_url_indicators(url):
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        uses_ip = True
    except ValueError:
        uses_ip = False

    lowered_url = unquote(url).lower()
    patterns_found = [
        pattern for pattern in SUSPICIOUS_PATTERNS
        if re.search(rf"{re.escape(pattern)}", lowered_url)
    ]
    host_labels = [label for label in hostname.rstrip(".").split(".") if label]
    if host_labels and host_labels[0].lower() == "www":
        host_labels = host_labels[1:]
    subdomain_count = max(len(host_labels) - 2, 0) if not uses_ip else 0
    unusual_punctuation = sorted({character for character in url if character in UNUSUAL_PUNCTUATION})
    return {
        "length": len(url),
        "https": parsed.scheme.lower() == "https",
        "ip": uses_ip,
        "subdomains": subdomain_count,
        "at_symbol": "@" in url,
        "patterns": patterns_found,
        "unusual_punctuation": unusual_punctuation,
    }


try:
    model = load_model()
except Exception:
    st.error("The saved model could not be loaded. Confirm that phishing_url_model.joblib is present and compatible.")
    st.stop()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

default_url = ""
if SAMPLES_PATH.exists():
    try:
        samples = pd.read_csv(SAMPLES_PATH).dropna(subset=["url", "actual"])
        samples = samples.reset_index(drop=True)
        options = ["Enter a URL yourself"] + [
            f"Demo {index + 1}: {'Phishing' if row['actual'] == 'Phishing' else 'Legitimate'}"
            for index, row in samples.iterrows()
        ]
        choice = st.selectbox("Demonstration example", options)
        if choice != options[0]:
            default_url = str(samples.iloc[options.index(choice) - 1]["url"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        st.caption("Demo examples are unavailable; you can still enter a URL below.")

st.markdown(
    """
    <div class="pg-scan-heading">
      <div class="pg-scan-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="10.8" cy="10.8" r="6.3" stroke="#42E8F5" stroke-width="1.6"/>
          <path d="m15.5 15.5 4.2 4.2M8.2 10.8h5.2M10.8 8.2v5.2" stroke="#18C6B5" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
      </div>
      <div><div class="pg-scan-title">Analyze a suspicious URL</div><div class="pg-scan-caption">Check a link before you click. Your URL stays in this local app.</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.form("url_check"):
    input_column, action_column = st.columns([5, 1.45], vertical_alignment="bottom")
    with input_column:
        url_input = st.text_input(
            "URL to analyze",
            value=default_url,
            placeholder="https://example.com",
        )
    with action_column:
        submitted = st.form_submit_button(
            "Analyze URL  →",
            type="primary",
            width="stretch",
        )

if submitted:
    st.session_state.last_result = None
    if not url_input.strip():
        st.warning("Enter a URL to start the analysis.")
    else:
        try:
            normalized_url = normalize_url(url_input)
            prediction = model.predict([normalized_url])[0]
            probabilities = model.predict_proba([normalized_url])[0]
            classes = list(model.classes_)
            score = float(probabilities[classes.index(prediction)])
            result = {
                "url": normalized_url,
                "prediction": str(prediction),
                "score": score,
                "indicators": get_url_indicators(normalized_url),
            }
            st.session_state.last_result = result
            st.session_state.scan_history.insert(
                0,
                {
                    "URL": normalized_url,
                    "Prediction": str(prediction),
                    "Predicted-class model score": f"{score * 100:.1f}%",
                },
            )
            st.session_state.scan_history = st.session_state.scan_history[:8]
        except ValueError as error:
            st.warning(str(error))
        except Exception:
            st.error("The model could not analyze this input. Please try another URL.")

result = st.session_state.last_result
if result:
    st.divider()
    st.subheader("Analysis result")
    result_column, score_column = st.columns([1.5, 1])
    is_phishing = result["prediction"] == "Phishing"
    with result_column:
        if is_phishing:
            st.markdown(
                '<div class="result-card phishing"><strong>⚠ &nbsp; POTENTIAL PHISHING</strong>'
                '<span>The model classified this URL as phishing. Treat it with caution.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card legitimate"><strong>✓ &nbsp; LIKELY LEGITIMATE</strong>'
                '<span>The model classified this URL as legitimate. This is not a safety guarantee.</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Predicted class: {result['prediction']}")
    with score_column:
        st.metric("Predicted-class model score", f"{result['score'] * 100:.1f}%")
        st.caption("Model probability for the predicted class; not a guarantee of safety.")

    st.subheader("Explainable URL risk indicators")
    st.markdown(
        '<p class="section-note">These structural observations are separate from the trained model and are not proof that a URL is malicious. Subdomain count is approximate because public-suffix rules vary.</p>',
        unsafe_allow_html=True,
    )
    indicators = result["indicators"]
    indicator_cards = [
        (
            "URL length",
            f"{indicators['length']} characters",
            "Long addresses can make the destination harder to spot; length alone does not indicate intent.",
        ),
        (
            "Subdomains",
            f"{indicators['subdomains']} (approx.)",
            "Counts labels before the final two hostname labels, excluding www. Country-code suffixes can affect this estimate.",
        ),
        (
            "IP address hostname",
            "Present" if indicators["ip"] else "Not present",
            "An IP address in place of a domain may deserve extra scrutiny, but can have legitimate uses.",
        ),
        (
            "@ symbol",
            "Present" if indicators["at_symbol"] else "Not present",
            "An @ can make the preceding text look like the destination; it is not conclusive by itself.",
        ),
        (
            "Login-related words",
            ", ".join(indicators["patterns"]) if indicators["patterns"] else "None detected",
            "Words like login, verify, or password can appear in legitimate URLs too.",
        ),
        (
            "Unusual punctuation",
            ", ".join(indicators["unusual_punctuation"]) if indicators["unusual_punctuation"] else "None detected",
            "These characters can occur in valid paths or queries; this is only a structural observation.",
        ),
        (
            "HTTPS scheme",
            "Present" if indicators["https"] else "Not present",
            "HTTPS protects the connection but does not prove that a website is legitimate.",
        ),
    ]
    card_markup = []
    for label, value, explanation in indicator_cards:
        card_markup.append(
            '<article class="pg-risk-item">'
            f"<h4>{html.escape(label)}</h4>"
            f'<div class="pg-risk-value">{html.escape(value)}</div>'
            f'<div class="pg-risk-description">{html.escape(explanation)}</div>'
            "</article>"
        )
    st.markdown(
        '<div class="pg-risk-grid">' + "".join(card_markup) + "</div>",
        unsafe_allow_html=True,
    )

    if is_phishing:
        st.warning(
            "Safety advice: Do not enter passwords, payment details, or verification codes. "
            "Reach the organization through a known official address or contact method instead. "
            "Predictions may be incorrect."
        )
    else:
        st.info("Still verify the domain carefully before sharing sensitive information.")

st.divider()
history_heading, history_action = st.columns([4, 1])
with history_heading:
    st.subheader("Recent scans")
with history_action:
    if st.session_state.scan_history and st.button("Clear history", width="stretch"):
        st.session_state.scan_history = []
        st.rerun()

if st.session_state.scan_history:
    st.dataframe(
        pd.DataFrame(st.session_state.scan_history),
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("Scans from this session will appear here. History is not saved after the session ends.")

st.markdown(
    '<footer class="pg-footer">PHISHGUARD provides URL-pattern analysis only. It does not visit websites, inspect live content, or certify a URL as safe.</footer>',
    unsafe_allow_html=True,
)

with st.expander("Model Performance"):
    st.caption(
        "Saved notebook evaluation on a stratified 80/20 holdout split "
        "(random_state=42; 10,000 test URLs). These are dataset results, not a guarantee for new URLs."
    )
    performance_columns = st.columns(3)
    performance_columns[0].metric("Accuracy", "99.38%")
    performance_columns[1].metric("Test URLs", "10,000")
    performance_columns[2].metric("Evaluation", "Held-out test split")

    metric_table = pd.DataFrame(
        {
            "Precision": [1.00, 0.99],
            "Recall": [0.99, 1.00],
            "F1-score": [0.99, 0.99],
            "Support": [5000, 5000],
        },
        index=["Phishing", "Legitimate"],
    )
    matrix_column, report_column = st.columns(2)
    with matrix_column:
        st.markdown("**Confusion matrix**")
        st.caption("Rows: actual class; columns: predicted class")
        confusion_matrix = pd.DataFrame(
            [[4938, 62], [0, 5000]],
            index=["Actual: Phishing", "Actual: Legitimate"],
            columns=["Predicted: Phishing", "Predicted: Legitimate"],
        )
        st.dataframe(confusion_matrix, width="stretch")
    with report_column:
        st.markdown("**Classification report**")
        st.dataframe(metric_table, width="stretch")
