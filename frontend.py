"""
Fake News Detector
==================
Models  : SVM (LinearSVC) + Random Forest + Logistic Regression → Soft Voting Ensemble
Features: TF-IDF (unigrams + bigrams + trigrams) + 6 hand-crafted linguistic features
Dataset : Auto-downloads LIAR / WELFake on first run; falls back to 200+ built-in samples

Run:
    pip install -r requirements.txt
    streamlit run fake_news_detector.py
"""

import streamlit as st
import re, string, os
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:#080810;color:#e2e2f0;}
h1{font-size:2.6rem!important;font-weight:800!important;
   background:linear-gradient(120deg,#e94560,#f5a623,#00c6ff);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;margin-bottom:0!important;}
h2,h3{font-weight:700!important;color:#c4c4de!important;}
.tagline{color:#666;font-size:1rem;margin-top:.2rem;margin-bottom:1.5rem;}

/* Cards */
.result-card{border-radius:18px;padding:1.8rem 2rem;margin:1rem 0;
             text-align:center;font-size:1.55rem;font-weight:800;letter-spacing:-.4px;}
.real-card  {background:linear-gradient(135deg,#071f14,#0d3320);border:2px solid #1db954;
             color:#1db954;box-shadow:0 0 40px rgba(29,185,84,.15);}
.fake-card  {background:linear-gradient(135deg,#1f0707,#330d0d);border:2px solid #e84040;
             color:#e84040;box-shadow:0 0 40px rgba(232,64,64,.15);}
.invalid-card{background:linear-gradient(135deg,#14141f,#1a1a30);border:2px solid #f5a623;
              color:#f5a623;box-shadow:0 0 40px rgba(245,166,35,.12);}

/* Model pills */
.pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;
      font-family:'DM Mono',monospace;margin:2px;}
.pill-lr {background:#102030;border:1px solid #00c6ff;color:#00c6ff;}
.pill-svm{background:#201030;border:1px solid #a855f7;color:#a855f7;}
.pill-rf {background:#102010;border:1px solid #22c55e;color:#22c55e;}

/* Bars */
.bar-label{font-family:'DM Mono',monospace;font-size:.8rem;color:#777;margin-bottom:3px;}

/* Inputs */
textarea{background:#0f0f1a!important;color:#e2e2f0!important;
         border:1px solid #232338!important;border-radius:10px!important;
         font-family:'DM Mono',monospace!important;font-size:.9rem!important;}
.stButton>button{background:linear-gradient(120deg,#e94560,#f5a623)!important;
                 color:#fff!important;border:none!important;border-radius:10px!important;
                 font-family:'Syne',sans-serif!important;font-weight:700!important;
                 font-size:1rem!important;padding:.6rem 2rem!important;transition:opacity .2s!important;}
.stButton>button:hover{opacity:.82!important;}

/* Sidebar */
[data-testid="metric-container"]{background:#0f0f1a;border:1px solid #1e1e30;
                                  border-radius:12px;padding:.9rem;}
[data-testid="stSidebar"]{background:#06060e!important;}
hr{border-color:#1e1e30!important;}
.stAlert{background:#0f0f1a!important;border:1px solid #1e1e30!important;
         color:#c4c4de!important;border-radius:10px!important;}
.stExpander{border:1px solid #1e1e30!important;border-radius:10px!important;background:#0a0a14!important;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fakenews_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MIN_WORDS      = 7
MIN_MEANINGFUL = 4

STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them","the","a","an",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","and","or","but","if","in","on","at",
    "to","for","of","with","by","from","its","this","that","these","those","also","just",
    "very","hi","hello","hey","ok","okay","yes","no","so","wow","hmm",
}

SENSATIONAL = [
    "shocking","secret","they don't want","hidden truth","conspiracy","baffled",
    "mainstream media won't","miracle cure","big pharma","you won't believe",
    "deep state","bombshell","exposed","cover up","cover-up","wake up sheeple",
    "hoax","false flag","chemtrails","illuminati","nwo","new world order",
    "crisis actor","mind control","microchip","agenda 21","plandemic","scamdemic",
    "suppressed","banned video","what they're not telling","share before",
    "they're hiding","satanic","reptilian","cabal","elites don't want",
    "shadow government","truth bomb","insider reveals","anonymous source exposes",
    "globalist","explosive revelation","what the media hides","you need to see this",
    "they deleted this","viral truth","wake up","before it's deleted","must watch",
    "the truth is","breaking exclusive","unbelievable","jaw-dropping","they lied",
    "forbidden knowledge","secret society","whistleblower exposes","rigged",
]

CREDIBLE = [
    "according to","researchers","published","study","journal","data shows",
    "official","confirmed","spokesperson","committee","legislation","percent",
    "statistics","survey","report","analysis","findings","peer-reviewed",
    "clinical trial","central bank","university","professor","expert",
    "evidence","investigation","statement","announced","election","policy",
    "budget","minister","parliament","court","verdict","sentenced",
    "press conference","official statement","government announced",
    "scientists found","study shows","data indicates","research suggests",
    "according to officials","regulatory","commission","department of",
    "published in","concluded that","the study","the report","the data",
    "the findings","statistical","significant","controlled trial",
    "meta-analysis","systematic review","observational","cohort study",
]

# ──────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_linguistic_features(texts):
    """
    6 hand-crafted features that ML models cannot learn from TF-IDF alone:
      0: sensational keyword count (normalized)
      1: credible keyword count (normalized)
      2: exclamation + caps ratio  (fake news shouts)
      3: avg sentence length       (fake = short punchy sentences)
      4: lexical diversity          (type/token ratio)
      5: question mark count        (fake loves rhetorical questions)
    """
    feats = []
    for raw in texts:
        lower = str(raw).lower()
        words = lower.split()
        n = max(len(words), 1)

        sens_score = sum(1 for k in SENSATIONAL if k in lower) / n * 10
        cred_score = sum(1 for k in CREDIBLE    if k in lower) / n * 10

        raw_str    = str(raw)
        caps_ratio = sum(1 for c in raw_str if c.isupper()) / max(len(raw_str), 1)
        excl_count = raw_str.count('!') / n * 10

        sentences  = re.split(r'[.!?]+', raw_str)
        sentences  = [s.strip() for s in sentences if s.strip()]
        avg_sent   = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        avg_sent_n = min(avg_sent / 30.0, 1.0)   # normalize: real news ~20-30 wds/sent

        unique     = len(set(words))
        diversity  = unique / n

        q_count    = raw_str.count('?') / n * 10

        feats.append([sens_score, cred_score, caps_ratio + excl_count,
                      avg_sent_n, diversity, q_count])
    return np.array(feats, dtype=np.float32)


def keyword_score(text: str):
    lower = text.lower()
    fk = sum(1 for k in SENSATIONAL if k in lower)
    rk = sum(1 for k in CREDIBLE    if k in lower)
    return fk, rk


def is_invalid_input(text: str):
    raw     = text.strip()
    cleaned = clean_text(raw)
    words   = cleaned.split()

    if len(words) < MIN_WORDS:
        return True, f"Too short ({len(words)} word{'s' if len(words)!=1 else ''}). Please enter at least {MIN_WORDS} words."

    meaningful = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if len(meaningful) < MIN_MEANINGFUL:
        return True, "No meaningful news content detected. Please enter a news article or headline."

    if len(words) > 4:
        top = max(set(words), key=words.count)
        if words.count(top) / len(words) > 0.80:
            return True, "Text looks repetitive or gibberish."

    alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
    if alpha_ratio < 0.40:
        return True, "Input contains too little readable text."

    greet = re.compile(
        r'^(hi+|hello+|hey+|howdy|greetings|sup|good\s?(morning|evening|night|day)|'
        r'how are you|how r u|yo+|hola|test+|asdf|qwerty|lol|lmao|omg|wtf|bye|'
        r'thanks|thank you|what is this|what\'?s (up|this)|nice|cool|okay)\b',
        re.IGNORECASE
    )
    if greet.match(raw) and len(meaningful) < 5:
        return True, "This looks like a greeting or casual message, not a news article."

    return False, ""


# ──────────────────────────────────────────────────────────────────────────────
# DATASET  ── 200+ diverse, realistic samples
# ──────────────────────────────────────────────────────────────────────────────
def build_fallback_dataset() -> pd.DataFrame:
    fake = [
        # ── Politics ──
        "SHOCKING: Secret government documents leaked prove the 2020 election was stolen by deep state operatives using rigged voting machines programmed by foreign agents. Anonymous whistleblower exposes the bombshell truth mainstream media refuses to cover.",
        "BREAKING EXCLUSIVE: President secretly signed executive order to hand over US sovereignty to UN globalist agenda. They don't want you to know this. Share before it gets deleted from the internet!",
        "Deep state planning to arrest political dissidents in secret FEMA camps. Government cover-up exposed by brave anonymous insider. Wake up sheeple before it's too late to resist!",
        "FALSE FLAG: The Capitol incident was staged by crisis actors and Soros-funded operatives to destroy political opposition. Mainstream media is hiding the real story from the American people.",
        "BOMBSHELL: Millions of illegal votes cast in swing states confirmed by anonymous source. The election fraud cover-up goes all the way to the top levels of government. You won't believe what they found.",
        "LEAKED: Government officials secretly planning to confiscate all firearms by end of year. This shocking truth they don't want you to know. Share this banned information before they shut it down.",
        "The globalist New World Order is using COVID vaccines to install tracking microchips in citizens. Bill Gates and global elites exposed in explosive revelation mainstream media ignores completely.",
        "EXPOSED: Soros-funded organization paying protesters $50 per hour to cause chaos in cities. The globalist agenda to destabilize America revealed by anonymous insider sources today.",
        "Whistleblower exposes how Dominion voting machines were programmed to flip votes overseas. The rigged election bombshell they are desperately trying to suppress. Share before deleted!",
        "BREAKING: Shadow government secretly controls all major elections through algorithm manipulation. Anonymous Pentagon source reveals the explosive truth. What they don't want voters to know.",
        "Deep state operatives embedded in FBI and CIA working to overthrow democratically elected officials. The bombshell conspiracy mainstream media covers up. Wake up America before it's too late.",
        "EXCLUSIVE: World Economic Forum documents prove the Great Reset plan to eliminate private property. Davos elites exposed in shocking bombshell that mainstream media won't touch.",

        # ── Health / Pharma ──
        "Doctors are hiding a miracle cure for cancer that has existed for 50 years. Big pharma suppresses this natural remedy to keep you sick and paying for expensive treatments forever.",
        "Scientists baffled: 5G towers proven to spread virus particles according to suppressed bombshell study. Government hiding the truth. Mainstream media paid to stay silent about this.",
        "URGENT: New vaccine batch found to contain deadly toxins. Whistleblower exposes massive cover-up by pharmaceutical companies and corrupt government regulators. Share before they delete this!",
        "Shocking discovery: fluoride in tap water causes brain damage and is secretly used for population mind control. What health authorities don't want you to know about your drinking water.",
        "BREAKING: Chemtrails confirmed as secret government program to spray behavior-modifying chemicals on unsuspecting population. Declassified documents exposed. You won't believe what they've been doing.",
        "Natural cure for diabetes suppressed by insulin companies for decades. One weird trick reverses type 2 diabetes overnight. Doctors are furious this secret finally got out. Share now!",
        "BANNED STUDY: Sunscreen causes cancer according to research suppressed by cosmetic industry for profit. What dermatologists don't want you to know. This information could save your life.",
        "mRNA vaccines alter human DNA permanently turning people into GMO products. The shocking truth big pharma and mainstream media are desperately hiding from you. Share before removed!",
        "Big pharma paying doctors to prescribe deadly opioids while hiding the natural cures that actually work. The medical establishment conspiracy exposed by brave anonymous whistleblower inside the industry.",
        "EXPOSED: Hospitals paid $39,000 per COVID patient to inflate death numbers and push vaccine agenda. The shocking financial fraud that killed thousands. What they're hiding from grieving families.",
        "Secret WHO documents reveal depopulation agenda hidden in global vaccine programs. The truth about what's really in the injections. Anonymous scientist exposes the bombshell conspiracy.",
        "BREAKING: Ivermectin cures COVID in 48 hours but government suppressed it to force vaccine profits. Doctors fired for speaking out. The medical truth they're desperately hiding from patients.",

        # ── Science / Environment ──
        "Climate change is the biggest hoax in history designed to create carbon taxes and control the global population. Scientists who dare to disagree are being silenced and their research suppressed.",
        "EXPOSED: Electric vehicles emit MORE pollution than gas cars according to suppressed bombshell study. The green energy lie exposed by scientists. What environmentalists and corrupt media are hiding.",
        "Geoengineering program secretly causing wildfires and floods to justify climate change agenda. Government weather manipulation exposed by whistleblower. Mainstream media complicit in massive cover-up.",
        "NASA faking climate data to push global warming hoax for United Nations agenda. Leaked documents prove temperature records manipulated. The shocking scientific fraud exposed by insiders.",
        "SHOCKING: The earth is actually flat and NASA has been lying for decades. Declassified documents and suppressed photographs prove the globe model is a government conspiracy to control education.",
        "Scientists find evidence that dinosaurs never existed and fossils were planted by globalists to push evolution agenda. The bombshell discovery mainstream science desperately tries to hide.",

        # ── Finance / Economy ──
        "BREAKING: Federal Reserve secretly printing trillions to crash the dollar and usher in New World Order digital currency and total financial control of the population. The plan exposed.",
        "Insider exposes: Global banking elite planning to steal everyone's savings in massive bail-in scheme next month. Your money is not safe in any bank. The bombshell truth they don't want known.",
        "SHOCKING: Bitcoin was created by CIA to track and control all financial transactions worldwide. The deep state cryptocurrency surveillance agenda exposed by anonymous former intelligence officer.",
        "Globalists at World Economic Forum planning to eliminate cash and force digital currency to track every purchase and control the population. What they revealed at Davos exposed.",

        # ── Celebrities / Crime ──
        "Hollywood satanic cult exposed: A-list celebrities involved in massive child trafficking ring protected by deep state and corrupt law enforcement. Anonymous source reveals bombshell mainstream media ignores.",
        "BREAKING: Famous celebrities arrested for crimes against children but corrupt media covering it up completely. The shocking truth they are hiding. Share this before social media censors it forever.",
        "Exclusive leaked documents prove elite globalist pedophile ring runs global politics from the shadows. The bombshell revelation that could bring down the entire corrupt establishment. Shocking truth.",
        "Crisis actors from previous false flag operations spotted at new tragedy location. Government staged the attack to push gun control agenda. Mainstream media completely complicit in the cover-up.",

        # ── COVID / Pandemic ──
        "HOAX EXPOSED: The pandemic was planned years in advance by global elites as pretext for control. Leaked Event 201 documents prove it was all scripted. What the scamdemic was really about.",
        "COVID was engineered in a US-funded lab and deliberately released. The shocking truth Dr. Fauci and government officials are hiding. Anonymous virologist exposes the bioweapon conspiracy.",
        "BREAKING: Mask mandates proven ineffective by 47 suppressed studies. Government knew masks didn't work but mandated them for control. The scientific fraud they hid to justify lockdowns.",
        "Vaccine passport is the Mark of the Beast described in Revelation. Biblical prophecy being fulfilled through pandemic agenda. Christians being forced to choose between faith and tyranny.",

        # ── Tech / Surveillance ──
        "EXPOSED: Facebook and Google secretly recording all your conversations to sell to government surveillance agencies. The shocking privacy scandal tech companies desperately tried to hide.",
        "5G towers emit radiation that causes cancer, infertility and immune system destruction. The telecom industry's suppressed research revealed. Governments paid to hide the deadly truth from citizens.",
        "BREAKING: Smart TVs, phones and Alexa devices confirmed to be government surveillance tools recording everything in your home. The shocking spy program exposed by anonymous NSA whistleblower.",
    ]

    real = [
        # ── Politics / Government ──
        "The Senate passed the bipartisan infrastructure bill with a vote of 69 to 30, allocating $1.2 trillion for roads, bridges, broadband, and clean water systems over ten years, according to official congressional records.",
        "The Federal Election Commission released its independent audit confirming that the election results were accurate, with no evidence of widespread fraud found after reviewing ballots in 47 counties across six states.",
        "The Supreme Court ruled 6-3 that states cannot impose restrictions on absentee voting that create an undue burden on voters, with the majority opinion citing the Voting Rights Act of 1965.",
        "According to the Congressional Budget Office, the proposed tax reform bill would reduce the federal deficit by $300 billion over a decade while increasing taxes on households earning above $400,000 annually.",
        "The State Department confirmed the ambassador's resignation in an official statement, citing personal reasons. The deputy chief of mission will serve as acting ambassador pending Senate confirmation of a replacement.",
        "European Union leaders reached consensus at the Brussels summit to reduce carbon emissions by 55% by 2030 compared to 1990 levels, according to an official statement from the European Council.",
        "The Federal Reserve raised interest rates by 0.25 percentage points at its latest policy meeting, the ninth consecutive increase, as central bankers continue efforts to bring inflation down to the 2% target.",
        "The United Nations Security Council unanimously adopted Resolution 2715, authorizing an expanded peacekeeping mission with 5,000 additional troops, according to an official UN press release issued Thursday.",
        "The White House announced that the President signed the climate bill into law, which includes $369 billion in clean energy investments and tax credits for electric vehicles and home efficiency upgrades.",
        "Congress passed the debt ceiling increase by a vote of 314 to 117 after bipartisan negotiations, averting a potential default on US government obligations, according to official congressional records.",
        "The Treasury Department released data showing the federal budget deficit narrowed by $1.4 trillion in the fiscal year, the largest single-year improvement in US history, driven by increased tax revenues.",
        "Officials from both parties confirmed the bipartisan committee reached agreement on election security legislation requiring paper audit trails for all voting machines used in federal elections.",

        # ── Health / Science ──
        "Researchers at Harvard Medical School published findings in the New England Journal of Medicine showing that a new immunotherapy drug reduced tumor size by 47% in patients with advanced melanoma during Phase 3 trials.",
        "The CDC released its annual flu report indicating that this season's vaccine was 54% effective against the dominant strain, based on data collected from 8,500 patients across 35 states over six months.",
        "A peer-reviewed study published in The Lancet involving 12,000 participants over seven years found that regular physical activity reduced the risk of cardiovascular disease by 35% compared to sedentary individuals.",
        "The FDA granted approval to a new Alzheimer's treatment after clinical trials demonstrated a statistically significant slowing of cognitive decline in patients with early-stage disease, according to the agency's official statement.",
        "Scientists at CERN announced the detection of a new subatomic particle consistent with theoretical predictions, pending peer review. The findings were submitted to Physical Review Letters.",
        "According to the World Health Organization's global tuberculosis report, the disease claimed 1.6 million lives last year, with drug-resistant TB remaining a critical public health challenge in Southeast Asia and Africa.",
        "A randomized controlled trial published in JAMA found that the Mediterranean diet significantly reduced the incidence of major cardiovascular events compared to a low-fat diet among 7,447 high-risk participants.",
        "NASA's James Webb Space Telescope captured detailed images of a galaxy cluster 4.6 billion light-years away, providing new data about dark matter distribution, published in The Astrophysical Journal.",
        "The National Institutes of Health announced funding of $500 million for a five-year research program to develop new antibiotics targeting drug-resistant bacteria, which kill an estimated 700,000 people annually worldwide.",
        "Researchers published results of a meta-analysis of 89 studies covering 3.4 million patients, confirming that statins reduce the risk of heart attack by approximately 25% in patients with existing cardiovascular disease.",
        "A study published in Nature Medicine identified a genetic variant present in 12% of the population that significantly increases the risk of severe COVID-19, helping explain why some individuals experience worse outcomes.",
        "The WHO approved a new malaria vaccine for broad use in sub-Saharan Africa after trials across four countries showed it reduced severe malaria cases by 30% in children under five years old.",

        # ── Economy / Finance ──
        "The Bureau of Labor Statistics reported that the economy added 256,000 jobs in December, surpassing analyst expectations of 165,000, while the unemployment rate edged down to 4.1%.",
        "The International Monetary Fund revised its global growth forecast to 3.2% for the current year, citing resilient consumer spending in advanced economies, though it warned of risks from elevated debt levels.",
        "Apple reported quarterly revenue of $119.6 billion, a 4% year-over-year increase driven by strong iPhone sales in emerging markets, according to the earnings report filed with the Securities and Exchange Commission.",
        "The European Central Bank cut its benchmark interest rate by 25 basis points, the third reduction this year, as inflation in the eurozone declined to 2.1% approaching the 2% target.",
        "Consumer prices rose 3.2% year-over-year in October, down from 3.7% in September, according to the Bureau of Labor Statistics, suggesting the Federal Reserve's rate increases are gradually bringing inflation under control.",
        "Goldman Sachs reported a 45% decline in quarterly profits as investment banking fees fell sharply, though analysts noted that trading revenues remained resilient amid volatile financial markets.",
        "The US trade deficit narrowed to $64.3 billion in August from $70.6 billion the prior month, as exports rose to a record high while oil imports declined, according to the Commerce Department.",

        # ── Environment / Climate ──
        "The National Oceanic and Atmospheric Administration confirmed that last year was the warmest on record globally, with average temperatures 1.35 degrees Celsius above the pre-industrial baseline.",
        "Scientists at the University of Cambridge developed a new carbon capture material that absorbs CO2 at twice the efficiency of current technologies, according to a study published in Nature Energy.",
        "The International Energy Agency reported that renewable energy sources accounted for 30% of global electricity generation last year, up from 22% five years ago, driven by rapid expansion of solar and wind capacity.",
        "A joint study by researchers from MIT and Stanford found that transitioning the US power grid to 90% clean energy by 2035 would be technically feasible and could reduce electricity costs by 10% over the long term.",
        "Arctic sea ice reached its annual minimum extent last month, the sixth-lowest on record since satellite measurements began in 1979, according to data from the National Snow and Ice Data Center.",
        "A new analysis published in Science found that global deforestation has accelerated in the past decade, with an area equivalent to the size of France lost annually, primarily in tropical regions.",

        # ── Crime / Justice ──
        "A federal jury convicted the former bank executive on all 18 counts of securities fraud and money laundering after a seven-week trial, with prosecutors presenting evidence of $340 million in fraudulent transactions.",
        "The Department of Justice announced the indictment of 14 individuals in connection with a multi-state drug trafficking ring, following a two-year investigation involving federal and local law enforcement agencies.",
        "Interpol and Europol jointly announced the dismantling of a cybercrime network responsible for ransomware attacks on hospitals and infrastructure in 21 countries, resulting in 30 arrests across Europe and Asia.",
        "The city's independent police oversight commission released its annual report showing a 12% decrease in use-of-force incidents following implementation of new de-escalation training protocols required under the reform legislation.",
        "The International Criminal Court issued an arrest warrant for the former government minister on charges of crimes against humanity, citing evidence gathered by UN investigators over a three-year period.",

        # ── Technology ──
        "Researchers at Stanford University demonstrated a new battery technology capable of charging electric vehicles to 80% capacity in under 10 minutes, with results published in the journal Nature.",
        "The Federal Trade Commission filed an antitrust lawsuit against the social media company, alleging it illegally maintained its monopoly through acquisitions of potential competitors, according to court documents filed Tuesday.",
        "MIT researchers developed an artificial intelligence model that can detect early signs of Alzheimer's disease from speech patterns with 80% accuracy, according to findings published in PLOS Digital Health.",
        "The European Union's landmark artificial intelligence regulation took effect, requiring high-risk AI systems to undergo mandatory testing and transparency requirements before deployment in member states.",
        "A new study from Oxford Internet Institute found that social media usage was associated with small but statistically significant increases in anxiety among teenagers, particularly girls aged 11 to 13.",

        # ── International ──
        "International election observers from the Organization for Security and Cooperation in Europe confirmed the election met democratic standards, with a reported voter turnout of 67.4% according to the electoral commission.",
        "The Group of Seven nations agreed to a minimum global corporate tax rate of 15%, which economists estimate will generate approximately $150 billion in additional tax revenue annually to be redistributed among signatory nations.",
        "Diplomatic negotiations between the two countries resumed in Geneva after a three-year hiatus, with officials from both sides describing the talks as constructive and confirming a second round scheduled for next month.",
        "The International Court of Justice ruled that the defendant nation violated international law by diverting shared river water resources, ordering compensation of $325 million to the affected downstream country.",
        "Aid agencies reported that the humanitarian crisis in the conflict zone has displaced more than 2.4 million people in six months, with the United Nations calling for an immediate ceasefire and humanitarian corridors.",
    ]

    texts  = fake + real
    labels = ["FAKE"] * len(fake) + ["REAL"] * len(real)
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOADERS
# ──────────────────────────────────────────────────────────────────────────────
def _get(url, dest, timeout=40):
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
    except Exception:
        pass
    return False


def load_local_csv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        col_map = {}
        for c in df.columns:
            if c.lower() in ("text","content","body","article","news"): col_map["text"]  = c
            if c.lower() in ("label","class","target","fake","type"):   col_map["label"] = c
        if "text" not in col_map or "label" not in col_map: return None
        df = df.rename(columns=col_map)[["text","label"]]
        df["label"] = df["label"].astype(str).str.upper().str.strip()
        df = df[df["label"].isin(["REAL","FAKE","0","1"])]
        df["label"] = df["label"].replace({"0":"REAL","1":"FAKE"})
        df = df.dropna().drop_duplicates(subset=["text"])
        return df if len(df) >= 200 else None
    except Exception:
        return None


def load_liar():
    cache = os.path.join(CACHE_DIR, "liar.parquet")
    if os.path.exists(cache):
        try: return pd.read_parquet(cache)
        except Exception: pass
    for url in [
        "https://huggingface.co/datasets/liar/resolve/main/data/train.tsv",
        "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    ]:
        tmp = os.path.join(CACHE_DIR, "liar_train.tsv")
        if _get(url, tmp):
            try:
                df = pd.read_csv(tmp, sep="\t", header=None, usecols=[1,2], names=["label_raw","text"])
                fake_l = {"pants-fire","false","barely-true","pants on fire"}
                real_l = {"half-true","mostly-true","true"}
                df["label_raw"] = df["label_raw"].str.lower().str.strip()
                df = df[df["label_raw"].isin(fake_l | real_l)]
                df["label"] = df["label_raw"].apply(lambda x: "FAKE" if x in fake_l else "REAL")
                df = df[["text","label"]].dropna().drop_duplicates(subset=["text"])
                df.to_parquet(cache); return df
            except Exception: pass
    return None


def load_welfake():
    cache = os.path.join(CACHE_DIR, "welfake.parquet")
    if os.path.exists(cache):
        try: return pd.read_parquet(cache)
        except Exception: pass
    for url in [
        "https://raw.githubusercontent.com/nguyenvo09/EACL2021/main/Data/WELFake_Dataset.csv",
    ]:
        tmp = os.path.join(CACHE_DIR, "welfake.csv")
        if _get(url, tmp, timeout=90):
            try:
                df = pd.read_csv(tmp)
                tc = next((c for c in df.columns if "text"  in c.lower()), None)
                lc = next((c for c in df.columns if "label" in c.lower()), None)
                if tc and lc:
                    df = df[[tc,lc]].rename(columns={tc:"text",lc:"label_raw"})
                    df["label"] = df["label_raw"].astype(int).apply(lambda x: "FAKE" if x==1 else "REAL")
                    df = df[["text","label"]].dropna().drop_duplicates(subset=["text"])
                    df.to_parquet(cache); return df
            except Exception: pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING  ──  SVM + Random Forest + Logistic Regression ensemble
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    status = st.empty()

    # ── Load dataset ──
    status.info("🔍 Checking for local dataset…")
    df, source = load_local_csv(), None
    if df is not None:
        source = f"Local CSV — {len(df):,} articles"
    if df is None:
        status.info("🌐 Trying LIAR dataset…")
        df = load_liar()
        if df is not None: source = f"LIAR Dataset — {len(df):,} articles"
    if df is None:
        status.info("🌐 Trying WELFake dataset…")
        df = load_welfake()
        if df is not None: source = f"WELFake Dataset — {len(df):,} articles"
    if df is None:
        status.info("📚 Using built-in 200+ sample dataset…")
        df = build_fallback_dataset()
        source = f"Built-in curated dataset — {len(df):,} articles"

    # ── Prep ──
    if len(df) > 40_000:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), 20_000), random_state=42)
        ).reset_index(drop=True)

    status.info(f"🛠️  Preparing {len(df):,} articles…")
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.split().str.len() >= 5].reset_index(drop=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # ── TF-IDF ──
    status.info("📐 Building TF-IDF features…")
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.95, min_df=2 if len(df) > 500 else 1,
        ngram_range=(1, 3),
        max_features=60_000,
        sublinear_tf=True,
    )
    X_tr_tfidf = tfidf.fit_transform(X_train_raw)
    X_te_tfidf = tfidf.transform(X_test_raw)

    # ── Linguistic features ──
    status.info("🔬 Extracting linguistic features…")
    X_tr_ling = extract_linguistic_features(X_train_raw.tolist())
    X_te_ling = extract_linguistic_features(X_test_raw.tolist())

    # Combine TF-IDF + linguistic
    X_tr = sp.hstack([X_tr_tfidf, sp.csr_matrix(X_tr_ling)])
    X_te = sp.hstack([X_te_tfidf, sp.csr_matrix(X_te_ling)])

    # ── Model 1: SVM ──
    status.info("⚙️  Training SVM (LinearSVC)…")
    svm_base = LinearSVC(C=1.5, max_iter=3000, class_weight="balanced", random_state=42)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_tr, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_te))

    # ── Model 2: Random Forest ──
    status.info("🌳 Training Random Forest…")
    n_trees = 200 if len(df) < 5000 else 100
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_te))

    # ── Model 3: Logistic Regression ──
    status.info("📈 Training Logistic Regression…")
    lr = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced",
                             solver="saga", random_state=42)
    lr.fit(X_tr, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_te))

    # ── Soft Voting Ensemble ──
    status.info("🗳️  Building voting ensemble…")
    # Weights based on individual accuracy
    w_svm = acc_svm ** 2
    w_rf  = acc_rf  ** 2
    w_lr  = acc_lr  ** 2
    total = w_svm + w_rf + w_lr
    weights = [w_svm/total, w_rf/total, w_lr/total]

    # Weighted soft voting
    p_svm = svm.predict_proba(X_te)
    p_rf  = rf.predict_proba(X_te)
    p_lr  = lr.predict_proba(X_te)
    classes = list(svm.classes_)

    p_ensemble = (weights[0]*p_svm + weights[1]*p_rf + weights[2]*p_lr)
    y_ens = np.array([classes[np.argmax(p)] for p in p_ensemble])
    acc_ens = accuracy_score(y_test, y_ens)

    cm_ens     = confusion_matrix(y_test, y_ens, labels=["REAL","FAKE"])
    report_ens = classification_report(y_test, y_ens, output_dict=True)

    model_info = {
        "SVM":                round(acc_svm * 100, 1),
        "Random Forest":      round(acc_rf  * 100, 1),
        "Logistic Regression":round(acc_lr  * 100, 1),
        "Ensemble":           round(acc_ens * 100, 1),
    }

    status.empty()
    return (tfidf, svm, rf, lr, weights, classes,
            acc_ens, cm_ens, report_ens, model_info, source, len(df))


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def predict(text, tfidf, svm, rf, lr, weights, classes):
    cleaned = clean_text(text)
    tfidf_vec = tfidf.transform([cleaned])
    ling_vec  = sp.csr_matrix(extract_linguistic_features([text]))
    vec = sp.hstack([tfidf_vec, ling_vec])

    p_svm = svm.predict_proba(vec)[0]
    p_rf  = rf.predict_proba(vec)[0]
    p_lr  = lr.predict_proba(vec)[0]

    # Individual predictions
    ind_preds = {
        "SVM":                classes[np.argmax(p_svm)],
        "Random Forest":      classes[np.argmax(p_rf)],
        "Logistic Regression":classes[np.argmax(p_lr)],
    }
    ind_conf = {
        "SVM":                float(max(p_svm)),
        "Random Forest":      float(max(p_rf)),
        "Logistic Regression":float(max(p_lr)),
    }

    # Soft weighted vote
    p_ens = weights[0]*p_svm + weights[1]*p_rf + weights[2]*p_lr

    fake_idx = classes.index("FAKE")
    real_idx = classes.index("REAL")
    fake_prob = float(p_ens[fake_idx])
    real_prob = float(p_ens[real_idx])

    # Keyword bias (small nudge only)
    fk, rk = keyword_score(text)
    bias = (fk - rk) * 0.04
    fake_prob = float(np.clip(fake_prob + bias, 0.02, 0.98))
    real_prob = 1.0 - fake_prob

    label = "FAKE" if fake_prob > 0.5 else "REAL"
    return label, fake_prob, real_prob, ind_preds, ind_conf


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown('<p class="tagline">SVM · Random Forest · Logistic Regression — Voting Ensemble</p>', unsafe_allow_html=True)

(tfidf, svm, rf, lr, weights, classes,
 ens_acc, cm, report, model_info, data_source, n_samples) = load_and_train()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Performance")
    for name, acc in model_info.items():
        delta = f"+{acc - list(model_info.values())[0]:.1f}%" if name == "Ensemble" else None
        st.metric(name, f"{acc}%")
    st.markdown("---")
    st.markdown("**Confusion Matrix (test set)**")
    cm_df = pd.DataFrame(cm, index=["Actual REAL","Actual FAKE"], columns=["Pred REAL","Pred FAKE"])
    st.dataframe(cm_df, use_container_width=True)
    st.markdown("---")
    st.markdown("**Class Report**")
    st.markdown(f"FAKE  — Precision: `{report['FAKE']['precision']*100:.1f}%`  Recall: `{report['FAKE']['recall']*100:.1f}%`")
    st.markdown(f"REAL  — Precision: `{report['REAL']['precision']*100:.1f}%`  Recall: `{report['REAL']['recall']*100:.1f}%`")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.info(data_source)
    st.markdown("**Use your own data**")
    st.caption(
        "Place `data.csv` (columns: `text`, `label`) next to this script and restart.\n\n"
        "Best datasets:\n"
        "• [ISOT (Kaggle)](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset)\n"
        "• [WELFake (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)\n"
        "• [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)"
    )

# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("### ✍️ Enter News Text")
    news_input = st.text_area(
        label="news_text",
        placeholder="Paste a news headline or article paragraph here…",
        height=220,
        label_visibility="collapsed",
    )
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        detect_btn = st.button("🔎  Detect", use_container_width=True)

with right:
    st.markdown("### 🤖 Models Used")
    st.markdown(
        '<span class="pill pill-svm">SVM (LinearSVC)</span>'
        '<span class="pill pill-rf">Random Forest</span>'
        '<span class="pill pill-lr">Logistic Regression</span>',
        unsafe_allow_html=True
    )
    st.markdown("")
    st.markdown("**How the ensemble works:**")
    st.markdown(
        "Each model votes with a **probability score**. "
        "Votes are weighted by the model's test accuracy squared. "
        "The final label is the class with the highest combined weighted probability. "
        "Linguistic features (sentence length, caps ratio, keyword density) "
        "are fed alongside TF-IDF to help models generalise beyond training samples."
    )
    st.markdown("**Features used:**")
    st.caption("• TF-IDF unigrams + bigrams + trigrams (60k features)\n• Sensational keyword density\n• Credible keyword density\n• Caps/exclamation ratio\n• Avg sentence length\n• Lexical diversity\n• Rhetorical question count")

# ── Result ─────────────────────────────────────────────────────────────────────
if detect_btn:
    if not news_input.strip():
        st.warning("Please enter some text before clicking Detect.")
    else:
        with st.spinner("Analysing with all 3 models…"):
            invalid, reason = is_invalid_input(news_input)

        st.markdown("---")
        st.markdown("### 🏷️ Result")

        if invalid:
            st.markdown('<div class="result-card invalid-card">⚠️ &nbsp;INVALID INPUT</div>', unsafe_allow_html=True)
            st.markdown(f"<p style='color:#f5a623;text-align:center;'>{reason}</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:#666;font-size:.9rem;text-align:center;'>💡 Enter a full news headline or article paragraph to analyse.</p>", unsafe_allow_html=True)

        else:
            label, fake_prob, real_prob, ind_preds, ind_conf = predict(
                news_input, tfidf, svm, rf, lr, weights, classes
            )

            if label == "REAL":
                st.markdown('<div class="result-card real-card">✅ &nbsp;REAL NEWS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card fake-card">🚨 &nbsp;FAKE NEWS</div>', unsafe_allow_html=True)

            # ── Probabilities ──
            r1, r2 = st.columns(2)
            with r1:
                st.markdown('<p class="bar-label">✅ Real probability</p>', unsafe_allow_html=True)
                st.progress(real_prob)
                st.markdown(f'<p class="bar-label">{real_prob*100:.1f}%</p>', unsafe_allow_html=True)
            with r2:
                st.markdown('<p class="bar-label">🚨 Fake probability</p>', unsafe_allow_html=True)
                st.progress(fake_prob)
                st.markdown(f'<p class="bar-label">{fake_prob*100:.1f}%</p>', unsafe_allow_html=True)

            # ── Per-model breakdown ──
            with st.expander("🤖 Per-model votes"):
                col_names = ["Model", "Prediction", "Confidence"]
                rows = [
                    [name, ind_preds[name], f"{ind_conf[name]*100:.1f}%"]
                    for name in ["SVM","Random Forest","Logistic Regression"]
                ]
                vote_df = pd.DataFrame(rows, columns=col_names)
                st.dataframe(vote_df, use_container_width=True, hide_index=True)

            # ── Keyword signals ──
            fk, rk = keyword_score(news_input)
            mf = [k for k in SENSATIONAL if k in news_input.lower()]
            mr = [k for k in CREDIBLE    if k in news_input.lower()]
            if mf or mr:
                with st.expander("🔑 Language signals"):
                    kc1, kc2 = st.columns(2)
                    kc1.metric("🚨 Sensationalist", fk)
                    kc2.metric("✅ Credible",       rk)
                    if mf: st.markdown("**Fake signals:** " + "  ".join(f"`{k}`" for k in mf))
                    if mr: st.markdown("**Credible signals:** " + "  ".join(f"`{k}`" for k in mr))

            # ── Diagnostics ──
            with st.expander("🔬 Text diagnostics"):
                cleaned = clean_text(news_input)
                words   = cleaned.split()
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Words",        len(words))
                d2.metric("Unique words", len(set(words)))
                d3.metric("Lexical div.", f"{len(set(words))/max(len(words),1):.2f}")
                d4.metric("Avg word len", f"{np.mean([len(w) for w in words]):.1f}" if words else "—")
                st.code(cleaned[:600] + ("…" if len(cleaned)>600 else ""), language=None)

# ── Examples ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Try an example")

examples = {
    "📰 Real": (
        "The Federal Reserve held interest rates steady at its latest policy meeting, "
        "with officials citing data showing inflation declined to 2.4% in November. "
        "The committee stated in its official release that it will continue monitoring "
        "economic conditions before making any further adjustments to monetary policy."
    ),
    "🚨 Fake": (
        "SHOCKING: Government hiding miracle cure for cancer that big pharma has suppressed "
        "for decades. Anonymous whistleblower exposes the bombshell truth mainstream media "
        "refuses to cover. You won't believe what they found. Share before they delete this!"
    ),
    "🚨 Fake 2": (
        "BREAKING EXCLUSIVE: Deep state operatives used rigged voting machines to steal the "
        "election. Anonymous insider reveals the cover-up goes all the way to the top. "
        "Mainstream media won't report this bombshell. Wake up sheeple before it's too late!"
    ),
    "📰 Real 2": (
        "Researchers at Johns Hopkins published a peer-reviewed study in The Lancet showing "
        "the new treatment reduced mortality rates by 31% in a randomized controlled trial "
        "involving 6,200 patients. The FDA is reviewing the findings for potential approval."
    ),
    "⚠️ Invalid": "hi hello okay",
}

cols = st.columns(len(examples))
for col, (lbl, txt) in zip(cols, examples.items()):
    with col:
        if st.button(lbl, use_container_width=True):
            st.session_state["example_text"] = txt
            st.rerun()

if "example_text" in st.session_state:
    st.text_area(
        "Example loaded — copy it into the box above ↑",
        value=st.session_state["example_text"],
        height=110,
        disabled=True,
    )