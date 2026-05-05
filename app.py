"""
AI-Based Intelligent Enquiry Chatbot for Himachal Pradesh Colleges
Final Year Project - Complete Implementation
"""

import streamlit as st
import random
import string
import time
import json
import re
from collections import Counter
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  NLTK SETUP (with graceful fallback)
# ─────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    for pkg in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    NLTK_AVAILABLE = True
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    NLTK_AVAILABLE = False
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
        "it", "its", "they", "what", "which", "who", "this", "that", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "a", "an", "the", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "about", "as", "into",
        "through", "during", "before", "after", "can", "am",
    }

# ─────────────────────────────────────────────
#  ML IMPORTS
# ─────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HP College Enquiry Chatbot",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 60%, #1a3a5c 100%);
    color: white;
    padding: 1.75rem 2rem 1.5rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(29,78,137,0.22);
}
.main-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.4px;
}
.main-header .subtitle {
    font-size: 0.92rem;
    opacity: 0.78;
    margin: 0 0 4px;
}
.main-header .hint {
    font-size: 0.82rem;
    opacity: 0.55;
    margin: 0;
    font-weight: 400;
}

/* ── Chat window ── */
.chat-container {
    background: #f0f4f9;
    border-radius: 16px;
    padding: 1.25rem;
    min-height: 420px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 1rem;
    border: 1px solid #dde4ef;
}
.user-bubble {
    background: #1d4e89;
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 22%;
    font-size: 0.93rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(29,78,137,0.22);
}
.bot-bubble {
    background: white;
    color: #1a2740;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 22% 8px 0;
    font-size: 0.93rem;
    line-height: 1.55;
    border: 1px solid #dde4ef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sender-label {
    font-size: 0.68rem;
    font-weight: 600;
    opacity: 0.5;
    margin-bottom: 3px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    border: 1px solid #dde4ef;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.metric-card h3 { font-size: 2rem; margin: 0 0 4px; color: #1d4e89; font-weight: 700; }
.metric-card p  { margin: 0; font-size: 0.78rem; color: #6a7a90; font-weight: 500; text-transform: uppercase; letter-spacing: 0.4px; }

/* ── Inputs & buttons ── */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #dde4ef !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1d4e89 !important;
    box-shadow: 0 0 0 3px rgba(29,78,137,0.08) !important;
}
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    background: #1d4e89 !important;
    color: white !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    font-size: 0.88rem !important;
}
.stButton > button:hover {
    background: #163d6e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(29,78,137,0.28) !important;
}
.stSelectbox label { font-weight: 600; color: #1a2740; font-size: 0.88rem; }

/* ── Sidebar ── */
.sidebar-college-box {
    background: #eef3fa;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 1.2rem;
    border: 1px solid #d4dff0;
}
.quick-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: #6a7a90;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
thead tr th { background: #f0f4f9 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE – 6 HP Colleges
# ═══════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = {

    "NIT Hamirpur": {
        "greetings": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "howdy", "greetings", "what's up", "namaste", "hi there",
            ],
            "responses": [
                "Hello! Welcome to NIT Hamirpur enquiry. How can I assist you today?",
                "Hi there! I'm the NIT Hamirpur virtual assistant. Ask me anything!",
                "Namaste! How can I help you with NIT Hamirpur information?",
            ],
        },
        "fees": {
            "patterns": [
                "what is the fee", "fees structure", "how much fees", "annual fees",
                "semester fees", "tuition fee", "total fees", "fee details",
                "how much does it cost", "fee per year",
            ],
            "responses": [
                "NIT Hamirpur B.Tech fee is approximately ₹1,48,700/year for general category. SC/ST students pay ₹48,700/year. The fee includes tuition, hostel, and mess charges.",
                "The tuition fee at NIT Hamirpur is ₹1,00,000/semester. Total B.Tech (4 year) expenditure is approximately ₹5.5–6 lakhs.",
            ],
        },
        "courses": {
            "patterns": [
                "courses available", "what courses", "which programmes", "branches offered",
                "engineering courses", "what can i study", "list of courses",
                "available programmes", "departments", "what do you offer",
            ],
            "responses": [
                "NIT Hamirpur offers B.Tech in: CSE, ECE, EE, ME, CE, Chemical Engineering, and Material Science. Also offers M.Tech and PhD programmes.",
                "Programmes at NIT Hamirpur: B.Tech (7 branches), M.Tech (multiple specializations), MBA, MSc, and PhD across all major engineering disciplines.",
            ],
        },
        "hostel": {
            "patterns": [
                "hostel facility", "accommodation", "hostel available", "hostel charges",
                "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
                "hostel fee", "is hostel available",
            ],
            "responses": [
                "NIT Hamirpur has separate hostels for boys and girls. 8 boys' hostels and 2 girls' hostels are available. Charges approx. ₹40,000/year including mess.",
                "Hostel accommodation is available for all students at NIT Hamirpur. Modern facilities with Wi-Fi, gym, and mess services are provided.",
            ],
        },
        "placements": {
            "patterns": [
                "placement record", "how are placements", "highest package", "companies visit",
                "placement percentage", "average salary", "job opportunities",
                "campus recruitment", "placement stats", "average package",
            ],
            "responses": [
                "NIT Hamirpur placement 2023: Highest package ₹44 LPA (Microsoft), Average package ₹8.5 LPA. 85%+ students placed. Top recruiters: TCS, Infosys, Wipro, Amazon, Zomato.",
                "Excellent placement record at NIT Hamirpur! 150+ companies visit campus. CSE branch average package is ₹12 LPA. Core engineering branches average ₹6–8 LPA.",
            ],
        },
        "contact": {
            "patterns": [
                "contact number", "phone number", "email address", "how to contact",
                "admission office", "address", "website", "helpline",
                "get in touch", "reach them",
            ],
            "responses": [
                "NIT Hamirpur Contact: Phone: 01972-254001 | Email: info@nith.ac.in | Website: www.nith.ac.in | Address: NIT Hamirpur, HP - 177005",
            ],
        },
        "exit": {
            "patterns": [
                "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
                "that's all", "no more questions",
            ],
            "responses": [
                "Thank you for enquiring about NIT Hamirpur! Best of luck with your admission. Goodbye! 🏔️",
                "Glad I could help! Feel free to return anytime. All the best! 👋",
            ],
        },
    },

    "IIT Mandi": {
        "greetings": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "howdy", "greetings", "namaste", "hi there", "what's up",
            ],
            "responses": [
                "Hello! Welcome to IIT Mandi enquiry chatbot. How may I assist you?",
                "Hi! I'm the IIT Mandi virtual assistant. What would you like to know?",
            ],
        },
        "fees": {
            "patterns": [
                "what is the fee", "fees structure", "how much fees", "annual fees",
                "semester fees", "tuition fee", "total fees", "fee details",
                "how much does it cost", "fee per year",
            ],
            "responses": [
                "IIT Mandi B.Tech fee: ₹2,00,000/year for general category. SC/ST/PH students pay ₹12,000/year. Hostel + Mess adds ₹50,000–70,000/year.",
                "Total IIT Mandi B.Tech cost (4 years) is approximately ₹10–12 lakhs for general category including all expenses.",
            ],
        },
        "courses": {
            "patterns": [
                "courses available", "what courses", "which programmes", "branches offered",
                "engineering courses", "what can i study", "list of courses",
                "available programmes", "departments", "what do you offer",
            ],
            "responses": [
                "IIT Mandi offers B.Tech in: CSE, ECE, EE, ME, CE, Data Science & AI, and Bioscience. Also M.Tech, MS (Research), and PhD programmes.",
                "Unique programmes at IIT Mandi include B.Tech in Data Science & AI, Smart Manufacturing, and Mechanical Engineering with focus on Innovation & Design.",
            ],
        },
        "hostel": {
            "patterns": [
                "hostel facility", "accommodation", "hostel available", "hostel charges",
                "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
                "hostel fee", "is hostel available",
            ],
            "responses": [
                "IIT Mandi has a beautiful campus in Kamand valley. All students are required to stay in hostels. Monthly mess charges: ₹3,500–4,500. Modern amenities with mountain views!",
            ],
        },
        "placements": {
            "patterns": [
                "placement record", "how are placements", "highest package", "companies visit",
                "placement percentage", "average salary", "job opportunities",
                "campus recruitment", "placement stats", "average package",
            ],
            "responses": [
                "IIT Mandi placement 2023: Highest package ₹62 LPA (International), Average ₹14.5 LPA. 90%+ placement rate. Top recruiters: Google, Microsoft, Samsung R&D, Qualcomm, DE Shaw.",
            ],
        },
        "contact": {
            "patterns": [
                "contact number", "phone number", "email address", "how to contact",
                "admission office", "address", "website", "helpline",
                "get in touch", "reach them",
            ],
            "responses": [
                "IIT Mandi Contact: Phone: 01905-267001 | Email: registrar@iitmandi.ac.in | Website: www.iitmandi.ac.in | Address: IIT Mandi, Kamand, Mandi, HP - 175075",
            ],
        },
        "exit": {
            "patterns": [
                "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
                "that's all", "no more questions",
            ],
            "responses": [
                "Thank you for exploring IIT Mandi! Wishing you great success in your academic journey! 🌟",
                "Goodbye! Feel free to come back with more queries. Best wishes! 👋",
            ],
        },
    },

    "HPU Shimla": {
        "greetings": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "howdy", "greetings", "namaste", "hi there", "what's up",
            ],
            "responses": [
                "Hello! Welcome to Himachal Pradesh University enquiry chatbot!",
                "Namaste! How can I assist you with HPU Shimla information today?",
            ],
        },
        "fees": {
            "patterns": [
                "what is the fee", "fees structure", "how much fees", "annual fees",
                "semester fees", "tuition fee", "total fees", "fee details",
                "how much does it cost", "fee per year",
            ],
            "responses": [
                "HPU Shimla is very affordable! BA/BSc/BCom: ₹3,000–8,000/year. PG courses: ₹10,000–25,000/year. Professional courses (BCA, MBA) range ₹30,000–80,000/year.",
                "HPU tuition fee varies by programme: UG arts ₹4,500/year | UG science ₹7,200/year | MBA ₹55,000/year | MCA ₹40,000/year.",
            ],
        },
        "courses": {
            "patterns": [
                "courses available", "what courses", "which programmes", "branches offered",
                "arts courses", "science courses", "what can i study", "list of courses",
                "available programmes", "departments", "what do you offer",
            ],
            "responses": [
                "HPU Shimla offers: BA (20+ subjects), BSc (Physics, Chemistry, Maths, Bio), BCom, BCA, MBA, MCA, MA, MSc, LLB, PhD across 32 teaching departments.",
                "HPU has 32 teaching departments with 100+ courses. Notable: Mass Communication, Tourism, Sanskrit, Law, Fine Arts, and all mainstream science/arts subjects.",
            ],
        },
        "hostel": {
            "patterns": [
                "hostel facility", "accommodation", "hostel available", "hostel charges",
                "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
                "hostel fee", "is hostel available",
            ],
            "responses": [
                "HPU Shimla has hostel facilities for boys and girls separately. Located on Summer Hill, Shimla. Hostel fee is very nominal at ₹15,000–20,000/year including mess.",
            ],
        },
        "placements": {
            "patterns": [
                "placement record", "how are placements", "highest package", "companies visit",
                "placement percentage", "average salary", "job opportunities",
                "campus recruitment", "placement stats", "average package",
            ],
            "responses": [
                "HPU Shimla has a dedicated placement cell. MBA and MCA students get good placements with average packages ₹3–5 LPA. Government jobs are also popular among HPU graduates.",
                "HPU placement focuses mainly on MBA, MCA, and professional courses. Companies like TCS, Wipro, HCL, and local government firms recruit from campus.",
            ],
        },
        "contact": {
            "patterns": [
                "contact number", "phone number", "email address", "how to contact",
                "admission office", "address", "website", "helpline",
                "get in touch", "reach them",
            ],
            "responses": [
                "HPU Shimla Contact: Phone: 0177-2830595 | Email: registrar@hpuniv.ac.in | Website: www.hpuniv.ac.in | Address: Summer Hill, Shimla, HP - 171005",
            ],
        },
        "exit": {
            "patterns": [
                "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
                "that's all", "no more questions",
            ],
            "responses": [
                "Thank you for enquiring about HPU Shimla! Best of luck for your future! 🌄",
                "Goodbye! Visit HPU's beautiful Shimla campus sometime! 👋",
            ],
        },
    },

    "Jaypee University Waknaghat": {
        "greetings": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "howdy", "greetings", "namaste", "hi there", "what's up",
            ],
            "responses": [
                "Hello! Welcome to Jaypee University Waknaghat enquiry! How can I help you?",
                "Hi! I'm the Jaypee University virtual assistant. Ask me anything!",
            ],
        },
        "fees": {
            "patterns": [
                "what is the fee", "fees structure", "how much fees", "annual fees",
                "semester fees", "tuition fee", "total fees", "fee details",
                "how much does it cost", "fee per year",
            ],
            "responses": [
                "Jaypee University B.Tech fee: ₹1,70,000/year. Total 4-year cost approx ₹8–9 lakhs. MBA: ₹2,20,000/year. Scholarships available for meritorious students.",
                "Jaypee B.Tech all-inclusive fee is ₹1,70,000 per annum (tuition + hostel + mess). Merit scholarships up to 50% fee waiver are available.",
            ],
        },
        "courses": {
            "patterns": [
                "courses available", "what courses", "which programmes", "branches offered",
                "engineering courses", "what can i study", "list of courses",
                "available programmes", "departments", "what do you offer",
            ],
            "responses": [
                "Jaypee University offers B.Tech in CSE, ECE, EE, ME, CE, Biotechnology, Chemical Eng. Also MBA, MCA, MSc Biotechnology, and PhD programmes.",
                "Popular courses at Jaypee: B.Tech CSE (AI/ML specialization), Biotechnology (strong research focus), MBA, and M.Tech across all engineering disciplines.",
            ],
        },
        "hostel": {
            "patterns": [
                "hostel facility", "accommodation", "hostel available", "hostel charges",
                "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
                "hostel fee", "is hostel available",
            ],
            "responses": [
                "Jaypee University is a fully residential campus at Waknaghat near Solan. Excellent hostel facilities with 24/7 Wi-Fi, sports complex, and cafeteria. All students live on campus.",
            ],
        },
        "placements": {
            "patterns": [
                "placement record", "how are placements", "highest package", "companies visit",
                "placement percentage", "average salary", "job opportunities",
                "campus recruitment", "placement stats", "average package",
            ],
            "responses": [
                "Jaypee University placement 2023: Highest package ₹40 LPA, Average CSE: ₹7.2 LPA. 80%+ placement rate. Top recruiters: TCS, Infosys, Accenture, Capgemini, and Samsung.",
            ],
        },
        "contact": {
            "patterns": [
                "contact number", "phone number", "email address", "how to contact",
                "admission office", "address", "website", "helpline",
                "get in touch", "reach them",
            ],
            "responses": [
                "Jaypee University Contact: Phone: 01792-257999 | Email: admissions@juitsolan.in | Website: www.juitsolan.in | Address: Waknaghat, Solan, HP - 173234",
            ],
        },
        "exit": {
            "patterns": [
                "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
                "that's all", "no more questions",
            ],
            "responses": [
                "Thanks for your interest in Jaypee University! Wishing you all the best! 🎓",
                "Goodbye! Feel free to return anytime. Good luck! 👋",
            ],
        },
    },

    "CSKHPKV Palampur": {
        "greetings": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "howdy", "greetings", "namaste", "hi there", "what's up",
            ],
            "responses": [
                "Hello! Welcome to CSK HPKV Palampur enquiry! How may I help you?",
                "Namaste! I'm the CSK HPKV virtual assistant. Ask me about agriculture studies in Himachal Pradesh!",
            ],
        },
        "fees": {
            "patterns": [
                "what is the fee", "fees structure", "how much fees", "annual fees",
                "semester fees", "tuition fee", "total fees", "fee details",
                "how much does it cost", "fee per year",
            ],
            "responses": [
                "CSKHPKV BSc Agriculture fee: ₹25,000–40,000/year. Fees are subsidized heavily for HP domicile students. MVSc and PhD programmes: ₹30,000–60,000/year.",
                "CSKHPKV is very affordable! HP domicile students pay less than ₹30,000/year for BSc Agriculture. Hostel charges are approximately ₹12,000/year.",
            ],
        },
        "courses": {
            "patterns": [
                "courses available", "what courses", "which programmes", "agriculture courses",
                "what can i study", "list of courses", "available programmes",
                "departments", "what do you offer", "bsc agriculture",
            ],
            "responses": [
                "CSKHPKV Palampur offers: BSc Agriculture (4 years), BSc Horticulture, BFSc (Fisheries), BSc Forestry, MSc Agriculture (20+ specializations), MVSc, and PhD.",
                "Key programmes: BSc Agriculture, BSc Horticulture, BSc Forestry, BTech Agricultural Engineering, plus PG and doctoral programmes across 35+ departments.",
            ],
        },
        "hostel": {
            "patterns": [
                "hostel facility", "accommodation", "hostel available", "hostel charges",
                "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
                "hostel fee", "is hostel available",
            ],
            "responses": [
                "CSKHPKV has a beautiful residential campus in Palampur, one of the most scenic university campuses in India. Hostel is mandatory for outstation students. Charges: ₹10,000–15,000/year.",
            ],
        },
        "placements": {
            "patterns": [
                "placement record", "how are placements", "highest package", "companies visit",
                "placement percentage", "average salary", "job opportunities",
                "campus recruitment", "placement stats", "average package",
            ],
            "responses": [
                "CSKHPKV graduates get government jobs in HP Agriculture Dept, ICAR institutes, seed companies, agro-industries. Average package ₹4–7 LPA. Many graduates go into research and civil services.",
                "Career options from CSKHPKV: Govt agriculture officer, bank (agriculture officer), agri-business, UPSC (IFS), and research positions. Strong alumni network in HP government.",
            ],
        },
        "contact": {
            "patterns": [
                "contact number", "phone number", "email address", "how to contact",
                "admission office", "address", "website", "helpline",
                "get in touch", "reach them",
            ],
            "responses": [
                "CSKHPKV Contact: Phone: 01894-230513 | Email: registrar@hillagric.ac.in | Website: www.hillagric.ac.in | Address: Palampur, HP - 176062",
            ],
        },
        "exit": {
            "patterns": [
                "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
                "that's all", "no more questions",
            ],
            "responses": [
                "Thank you for your interest in CSKHPKV Palampur! Agriculture is the backbone of Himachal. Best wishes! 🌿",
                "Goodbye! Hope to see you at our green, scenic campus in Palampur! 👋",
            ],
        },
    },

    "JNVU Mandi (JNGEC)": {
        "greetings": {
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "howdy", "greetings", "namaste", "hi there", "what's up",
            ],
            "responses": [
                "Hello! Welcome to JNGEC Mandi enquiry chatbot! How can I assist you?",
                "Hi there! I'm the JNGEC virtual assistant. What would you like to know about us?",
            ],
        },
        "fees": {
            "patterns": [
                "what is the fee", "fees structure", "how much fees", "annual fees",
                "semester fees", "tuition fee", "total fees", "fee details",
                "how much does it cost", "fee per year",
            ],
            "responses": [
                "JNGEC Mandi B.Tech fee: ₹75,000–85,000/year. Being a government college, fees are very reasonable. Hostel charges approx ₹25,000/year.",
                "Total JNGEC B.Tech (4 year) cost is around ₹3.5–4 lakhs, making it one of the most affordable engineering colleges in HP.",
            ],
        },
        "courses": {
            "patterns": [
                "courses available", "what courses", "which programmes", "branches offered",
                "engineering courses", "what can i study", "list of courses",
                "available programmes", "departments", "what do you offer",
            ],
            "responses": [
                "JNGEC Mandi offers B.Tech in: CSE, ECE, EE, ME, and CE. Also M.Tech in CSE and ECE. Intake is limited ensuring good faculty-student ratio.",
            ],
        },
        "hostel": {
            "patterns": [
                "hostel facility", "accommodation", "hostel available", "hostel charges",
                "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
                "hostel fee", "is hostel available",
            ],
            "responses": [
                "JNGEC Mandi provides hostel facilities for boys and girls separately. Located in scenic Mandi district. Hostel charges are very nominal at ₹20,000–25,000/year including mess.",
            ],
        },
        "placements": {
            "patterns": [
                "placement record", "how are placements", "highest package", "companies visit",
                "placement percentage", "average salary", "job opportunities",
                "campus recruitment", "placement stats", "average package",
            ],
            "responses": [
                "JNGEC Mandi placement 2023: Average package ₹4–6 LPA. Highest ₹18 LPA (CSE). Companies: TCS, Wipro, Infosys, L&T, and HP government departments.",
                "JNGEC placement is improving steadily. 60–70% placement rate. Government jobs and HP Public Works Department also recruit many JNGEC graduates.",
            ],
        },
        "contact": {
            "patterns": [
                "contact number", "phone number", "email address", "how to contact",
                "admission office", "address", "website", "helpline",
                "get in touch", "reach them",
            ],
            "responses": [
                "JNGEC Mandi Contact: Phone: 01905-220103 | Email: principal@jngecmandi.ac.in | Website: www.jngecmandi.ac.in | Address: Sundernagar Road, Mandi, HP - 175001",
            ],
        },
        "exit": {
            "patterns": [
                "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
                "that's all", "no more questions",
            ],
            "responses": [
                "Thank you for enquiring about JNGEC Mandi! Best of luck! 🏔️",
                "Goodbye! Feel free to ask more questions anytime. Good luck with admissions! 👋",
            ],
        },
    },
}

COLLEGE_NAMES = list(KNOWLEDGE_BASE.keys())

# ═══════════════════════════════════════════════════════════════════
#  COMPARISON DATA
# ═══════════════════════════════════════════════════════════════════
COMPARISON_DATA = {
    "NIT Hamirpur":            {"fees": "₹1,48,700/yr", "avg_package": "₹8.5 LPA",  "highest_pkg": "₹44 LPA",  "courses": "7 branches", "type": "Government NIT"},
    "IIT Mandi":               {"fees": "₹2,00,000/yr", "avg_package": "₹14.5 LPA", "highest_pkg": "₹62 LPA",  "courses": "8 branches", "type": "IIT (Central)"},
    "HPU Shimla":              {"fees": "₹4,500–55,000/yr","avg_package":"₹3.5 LPA", "highest_pkg": "₹12 LPA",  "courses": "100+ courses","type": "State University"},
    "Jaypee University Waknaghat": {"fees": "₹1,70,000/yr", "avg_package": "₹7.2 LPA",  "highest_pkg": "₹40 LPA",  "courses": "10 branches", "type": "Private Deemed"},
    "CSKHPKV Palampur":        {"fees": "₹30,000/yr",    "avg_package": "₹4.5 LPA",  "highest_pkg": "₹15 LPA",  "courses": "Agriculture-focused","type": "Agriculture Univ"},
    "JNVU Mandi (JNGEC)":      {"fees": "₹80,000/yr",    "avg_package": "₹5 LPA",    "highest_pkg": "₹18 LPA",  "courses": "5 branches",  "type": "Government"},
}

# ═══════════════════════════════════════════════════════════════════
#  NLP PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def preprocess(text: str) -> str:
    """
    Pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. Rejoin
    """
    text = text.lower()
    text = re.sub(r"[" + string.punctuation + r"]", " ", text)
    if NLTK_AVAILABLE:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# ═══════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_dataset(college: str):
    X, y = [], []
    for intent, data in KNOWLEDGE_BASE[college].items():
        for pattern in data["patterns"]:
            X.append(preprocess(pattern))
            y.append(intent)
    return X, y

# ═══════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_models():
    """Train both NB and LR models on combined dataset of all colleges."""
    all_X, all_y = [], []
    for college in COLLEGE_NAMES:
        X, y = build_dataset(college)
        all_X.extend(X)
        all_y.extend(y)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(all_X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, all_y, test_size=0.25, random_state=42, stratify=all_y
    )

    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

    lr_model = LogisticRegression(max_iter=500, C=1.5, solver="lbfgs", multi_class="auto")
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    best_model = lr_model if lr_acc >= nb_acc else nb_model
    best_name  = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"

    return vectorizer, nb_model, lr_model, nb_acc, lr_acc, best_model, best_name

# ═══════════════════════════════════════════════════════════════════
#  INTENT PREDICTION
# ═══════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.30

def predict_intent(text: str, vectorizer, model):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    intent = model.predict(vec)[0]
    proba  = model.predict_proba(vec).max()
    return intent, float(proba)

# ═══════════════════════════════════════════════════════════════════
#  RESPONSE GENERATOR
# ═══════════════════════════════════════════════════════════════════
def get_response(college: str, intent: str, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return "❓ Sorry, I didn't quite understand that. Could you please rephrase your query? Try asking about fees, courses, hostel, placements, or contact details."
    responses = KNOWLEDGE_BASE[college][intent]["responses"]
    return random.choice(responses)

# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "chat_history": [],   # list of (role, message)
        "query_count":  0,
        "intent_log":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ═══════════════════════════════════════════════════════════════════
#  TRAIN MODELS (CACHED)
# ═══════════════════════════════════════════════════════════════════
with st.spinner("🔄 Setting up the assistant..."):
    vectorizer, nb_model, lr_model, nb_acc, lr_acc, best_model, best_name = train_models()

# Always use the best model — hidden from user
active_model = best_model

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#1a2740;margin-bottom:4px;'>🏔️ HP College Assistant</div>"
        "<div style='font-size:0.78rem;color:#8a9ab5;margin-bottom:1.2rem;'>Your smart college guide</div>",
        unsafe_allow_html=True,
    )

    # College selector
    st.markdown('<div class="sidebar-college-box">', unsafe_allow_html=True)
    selected_college = st.selectbox(
        "🎓 Select a College",
        COLLEGE_NAMES,
        help="Choose the college you want to enquire about",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick Ask buttons
    st.markdown('<div class="quick-label">Quick Questions</div>', unsafe_allow_html=True)
    quick_map = {
        "💰 Fees":       "Tell me about fees",
        "📚 Courses":    "Tell me about courses",
        "🏠 Hostel":     "Tell me about hostel",
        "💼 Placements": "Tell me about placements",
        "📞 Contact":    "Tell me about contact",
    }
    for label, query in quick_map.items():
        if st.button(label, key=f"quick_{label}", use_container_width=True):
            intent, confidence = predict_intent(query, vectorizer, active_model)
            response = get_response(selected_college, intent, confidence)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", response))
            st.session_state.query_count += 1
            st.session_state.intent_log.append(intent)
            st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f2;margin:1rem 0;'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<div style='font-size:0.72rem;color:#b0bec5;text-align:center;margin-top:1.5rem;line-height:1.6;'>"
        "HP College Enquiry Chatbot<br>Final Year Project"
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🏔️ HP College Enquiry Chatbot</h1>
    <p class="subtitle">Instant answers about Himachal Pradesh colleges — admissions, fees, courses & more</p>
    <p class="hint">Select a college and ask your query below</p>
</div>
""", unsafe_allow_html=True)

# Tabs — no Model Info tab
tab_chat, tab_compare, tab_analytics = st.tabs(
    ["💬 Chat", "⚖️ Compare Colleges", "📊 Usage Stats"]
)

# ─────────────────────────────────────────────
#  TAB 1 – CHAT
# ─────────────────────────────────────────────
with tab_chat:
    # College indicator
    st.markdown(
        f"<div style='font-size:0.82rem;color:#6a7a90;margin-bottom:0.6rem;'>"
        f"🎓 Chatting with <strong style='color:#1d4e89;'>{selected_college}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Chat display
    chat_html = '<div class="chat-container">'
    if not st.session_state.chat_history:
        chat_html += (
            '<div style="text-align:center;color:#8a9ab5;padding:3.5rem 1rem 2rem;">'
            '<div style="font-size:2rem;margin-bottom:0.75rem;">👋</div>'
            '<div style="font-size:0.97rem;font-weight:600;color:#6a7a90;margin-bottom:0.4rem;">Welcome! How can I help you?</div>'
            '<div style="font-size:0.82rem;color:#a0b0c5;">Try asking: <em>"What are the fees?"</em> or use the Quick Questions in the sidebar.</div>'
            '</div>'
        )
    else:
        for role, msg in st.session_state.chat_history:
            if role == "user":
                chat_html += f'<div class="user-bubble"><div class="sender-label">You</div>{msg}</div>'
            else:
                chat_html += f'<div class="bot-bubble"><div class="sender-label">Assistant</div>{msg}</div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input row
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input(
            "Your query",
            placeholder="Type your question here — e.g. What are the hostel charges?",
            label_visibility="collapsed",
            key="user_query",
        )
    with col_btn:
        send_btn = st.button("Send ➤", use_container_width=True)

    if send_btn and user_input.strip():
        query = user_input.strip()
        intent, confidence = predict_intent(query, vectorizer, active_model)
        response = get_response(selected_college, intent, confidence)
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("bot", response))
        st.session_state.query_count += 1
        st.session_state.intent_log.append(intent)
        st.rerun()
    elif send_btn:
        st.warning("⚠️ Please type a question before sending.")

# ─────────────────────────────────────────────
#  TAB 2 – COMPARE COLLEGES
# ─────────────────────────────────────────────
with tab_compare:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#1a2740;margin-bottom:4px;'>⚖️ Compare Colleges Side by Side</div>"
        "<div style='font-size:0.83rem;color:#8a9ab5;margin-bottom:1.2rem;'>Select any two colleges to see a detailed comparison</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        college_a = st.selectbox("First College", COLLEGE_NAMES, key="ca")
    with c2:
        college_b = st.selectbox("Second College", COLLEGE_NAMES, index=1, key="cb")

    if college_a == college_b:
        st.warning("Please select two different colleges to compare.")
    else:
        da, db = COMPARISON_DATA[college_a], COMPARISON_DATA[college_b]
        fields = {
            "College Type":       ("type",        "type"),
            "Annual Fees":        ("fees",        "fees"),
            "Average Package":    ("avg_package", "avg_package"),
            "Highest Package":    ("highest_pkg", "highest_pkg"),
            "Programmes Offered": ("courses",     "courses"),
        }

        st.markdown("<hr style='border:none;border-top:1px solid #e8eef5;margin:0.8rem 0 1rem;'>", unsafe_allow_html=True)

        rows = []
        for label, (ka, kb) in fields.items():
            rows.append({"Parameter": label, college_a: da[ka], college_b: db[kb]})
        df = pd.DataFrame(rows).set_index("Parameter")
        st.table(df)

        # Average package bar chart
        st.markdown(
            "<div style='font-size:0.88rem;font-weight:600;color:#1a2740;margin:1rem 0 0.5rem;'>Average Placement Package Comparison</div>",
            unsafe_allow_html=True,
        )
        pkgs = {
            college_a: float(da["avg_package"].split()[0].replace("₹", "").replace(",", "")),
            college_b: float(db["avg_package"].split()[0].replace("₹", "").replace(",", "")),
        }
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#1d4e89", "#5ba3d9"]
        bars = ax.bar(pkgs.keys(), pkgs.values(), color=colors, width=0.42, edgecolor="none", zorder=3)
        ax.set_ylabel("Avg. Package (LPA)", fontsize=10)
        ax.set_ylim(0, max(pkgs.values()) * 1.35)
        ax.set_facecolor("#f7f9fc")
        fig.patch.set_facecolor("#f7f9fc")
        ax.grid(axis="y", color="#e2e8f2", linewidth=0.8, zorder=0)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0, labelsize=9)
        for bar, val in zip(bars, pkgs.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pkgs.values()) * 0.03,
                f"₹{val} LPA",
                ha="center", va="bottom", fontsize=10, fontweight="700", color="#1a2740",
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────
#  TAB 3 – ANALYTICS
# ─────────────────────────────────────────────
with tab_analytics:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#1a2740;margin-bottom:4px;'>📊 Chatbot Usage Stats</div>"
        "<div style='font-size:0.83rem;color:#8a9ab5;margin-bottom:1.2rem;'>See how students are using this assistant</div>",
        unsafe_allow_html=True,
    )

    # Seed demo data so analytics always look meaningful on first load
    DEMO_INTENTS = [
        "fees", "fees", "fees", "courses", "courses", "hostel",
        "placements", "placements", "contact", "greetings", "courses",
        "hostel", "fees", "placements", "courses",
    ]
    display_log   = st.session_state.intent_log if st.session_state.intent_log else DEMO_INTENTS
    display_count = st.session_state.query_count if st.session_state.query_count else len(DEMO_INTENTS)

    intent_counts = Counter(display_log)
    top_intent    = intent_counts.most_common(1)[0][0] if intent_counts else "—"
    unique_count  = len(set(display_log))

    # Friendly label map
    intent_labels = {
        "fees":       "Fees & Charges",
        "courses":    "Courses & Programmes",
        "hostel":     "Hostel & Accommodation",
        "placements": "Placements & Jobs",
        "contact":    "Contact & Address",
        "greetings":  "Greetings",
        "exit":       "Exit / Bye",
    }
    top_label = intent_labels.get(top_intent, top_intent.capitalize())

    # Metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><h3>{display_count}</h3><p>Queries Asked</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h3 style="font-size:1.3rem;padding-top:0.3rem;">{top_label}</h3><p>Most Popular Topic</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h3>{unique_count}</h3><p>Topics Explored</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none;border-top:1px solid #e8eef5;margin:1.2rem 0;'>", unsafe_allow_html=True)

    # Bar chart — friendly labels, sorted by count
    st.markdown(
        "<div style='font-size:0.9rem;font-weight:600;color:#1a2740;margin-bottom:0.6rem;'>Questions by Topic</div>",
        unsafe_allow_html=True,
    )
    sorted_counts  = intent_counts.most_common()
    chart_labels   = [intent_labels.get(k, k.capitalize()) for k, _ in sorted_counts]
    chart_values   = [v for _, v in sorted_counts]
    chart_colors   = ["#1d4e89", "#2d6a9f", "#4a86b8", "#6ba0cc", "#8ebde0", "#b4d4ed", "#d6eaf8"]

    fig2, ax2 = plt.subplots(figsize=(8, max(3, len(chart_labels) * 0.55)))
    bars2 = ax2.barh(
        chart_labels[::-1], chart_values[::-1],
        color=chart_colors[:len(chart_labels)][::-1],
        edgecolor="none", height=0.52, zorder=3,
    )
    ax2.set_xlabel("Number of Questions", fontsize=9, color="#6a7a90")
    ax2.set_facecolor("#f7f9fc")
    fig2.patch.set_facecolor("#f7f9fc")
    ax2.grid(axis="x", color="#e2e8f2", linewidth=0.8, zorder=0)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax2.tick_params(axis="both", which="both", length=0, labelsize=9)
    for bar, val in zip(bars2, chart_values[::-1]):
        ax2.text(
            val + 0.08, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9, fontweight="600", color="#1a2740",
        )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    if not st.session_state.intent_log:
        st.caption("Showing sample data. Start chatting to see your own usage stats!")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<hr style='border:none;border-top:1px solid #edf0f5;margin:2rem 0 0.8rem;'>"
    "<div style='text-align:center;color:#b0bec5;font-size:0.78rem;'>"
    "HP College Enquiry Chatbot &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; Himachal Pradesh"
    "</div>",
    unsafe_allow_html=True,
)