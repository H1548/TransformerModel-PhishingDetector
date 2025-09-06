#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined Email Dataset Generator — CSV: email,label
- Balanced across 9 refined labels (no spear class)
- Heavily expanded scenario/topic banks for variety
- English-only ASCII output
- Subject/body coherence with stable "Subject:" header and paragraphs
- Rephrasing for lexical diversity
- Near-duplicate filtering (trigram Jaccard)
- Label content guards to reduce cross-label leakage
- Phishing emails are SANITIZED with training disclaimers and neutered links:
  e.g., hxxps://secure-nimbusdrive[.]invalid/account/verify?rid=123456

Refined label set:
  phishing-credential
  phishing-payment        (merges financial, invoice, tax)
  phishing-delivery
  phishing-techsupport
  phishing-job
  safe-work               (replaces safe-business)
  safe-personal
  safe-marketing
  safe-transactional
"""

from __future__ import annotations
import argparse, csv, hashlib, random, re, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

# ------------------------------ Labels ------------------------------------
LABEL_MAP = {
    'phishing-credential': 0,   # merges phishing-login
    'phishing-payment': 1,      # merges phishing-financial, phishing-invoice, phishing-tax
    'phishing-delivery': 2,
    'phishing-techsupport': 3,
    'phishing-job': 4,
    'safe-work': 5,             # replaces safe-business
    'safe-personal': 6,
    'safe-marketing': 7,
    'safe-transactional': 8,
}
LABELS = list(LABEL_MAP.keys())

# ------------------------------ Vocab -------------------------------------
FAKE_BRANDS = {
    'bank': [
        'Contoso Bank','Northbridge Savings','Pioneer Credit','BlueOak Bank','Apex Federal',
        'Union & Crown','Harborline Trust','Equinox Financial','First Meridian','SilverGate Capital'
    ],
    'wallet': ['NovaPay','QuantaWallet','SwiftPur','AlphaWallet','PaySphere','OrbitWallet'],
    'parcel': [
        'ParcelPro','DoorDashr','QuickPost','FlyerGo','ArrowShip','MetroParcel','RapidBox','CityCourier'
    ],
    'tax': [
        'Revenue Office','CivicTax','National Revenue Service','Tax Bureau','Public Revenue Agency','Local Taxation Office'
    ],
    'cloud': [
        'NimbusDrive','SkyDocs','CloudCask','ArkSync','Vertex365','DocNest','CoreVault'
    ],
    'isp': ['MetroNet','UrbanFiber','Skyline ISP','VelocityNet','Cityline Broadband'],
    'hr': ['TalentNest','JobForge','HireWorks','WorkHive','StaffSpring','CareerArcade'],
    'it': ['HelpDeskNow','TechAid','SupportHub','ShieldOps','BlueLight IT','FixPoint'],
    'ecom': ['Shopado','MegaMart','ClickCart','Outletly','CartHub','DailyDealers'],
    'soft': ['RapidSuite','SyncWave','PageForge','BugTrackr','PlanGrid','PulseMail']
}
NAMES = [
    'Alex','Jordan','Taylor','Casey','Riley','Morgan','Sam','Jamie','Avery','Quinn','Chris','Drew','Cameron','Lee','Kendall','Robin','Dana','Skyler','Devin','Shawn',
    'Parker','Elliot','Rowan','Hayden','Sage','Emerson','Finley','Arden','Blair','Reese'
]
COMPANY_ROLES = ['Finance','Operations','IT','HR','Sales','Legal','Marketing','Support','Procurement','Engineering','Data']
COMPANY_NAMES = [
    'Ardenica Ltd','Hawthorne Group','Veltrex Solutions','Orchid Labs','Greyfield PLC','LuminaWorks','Zentra Systems','Riverton Analytics',
    'Stonebridge Media','Oakline Trading','Brightshore Partners','Helios Robotics','NovaVista Biotech','Pinecrest Logistics'
]
CURRENCIES = ['USD','GBP','EUR']
CURRENCY_SIGNS = {'USD':'$','GBP':'£','EUR':'€'}
AMOUNTS = ['9.99','12.49','18.30','27.50','38.90','49.00','74.99','129.00','219.85','480.00','1,249.00','2,480.50']
DATE_SNIPPETS = ['today','in 24 hours','by end of day','within 48 hours','immediately','by Friday','before midnight','by tomorrow','this afternoon','this morning']
HEDGES = ['it appears','we believe','it seems','likely','possibly','as a precaution','to be safe','based on logs']
URGENCY = ['urgent','immediate','final notice','action required','time-sensitive','priority','high importance']
GREETINGS = ['Hi','Hello','Good morning','Good afternoon','Dear','Hey']
SIGNOFFS = ['Regards','Best','Sincerely','Thanks','Warm regards','Kind regards','Many thanks']

# ASCII-only guard
ENGLISH_ONLY_PATTERN = re.compile(r'^[\x00-\x7F]+$')

# ------------------------------ Scenario banks (EXPANDED) -----------------
SCENARIO = {
    # phishing-credential (ex-login)
    'phishing-credential': {
        'device': [
            'Windows 11 laptop','MacBook Pro','iPhone 15','Android tablet','Linux workstation','ChromeOS device',
            'public library PC','Internet café terminal','unknown smart TV','new browser profile','unrecognized desktop'
        ],
        'geo': [
            'London, UK','Paris, FR','Berlin, DE','Dublin, IE','Madrid, ES','New York, US','Toronto, CA','Sydney, AU',
            'Singapore, SG','Tokyo, JP','Cape Town, ZA','Warsaw, PL','Dubai, AE','Sao Paulo, BR'
        ],
        'trigger': [
            'unusual sign-in attempt','multiple failed login attempts','password reset request','new device registration',
            'unrecognized location access','MFA prompt bypass','account lock due to suspicious activity','API token misuse',
            'stale session detected','suspicious OAuth grant','disabled 2FA method'
        ],
        'asset': [
            'Google Drive','Microsoft 365','Dropbox','Box','Company VPN','Corporate Intranet','Email account',
            'HR system','Finance portal','Git repository','KMS','Ticketing system','CI pipeline'
        ],
        'time': DATE_SNIPPETS,
    },
    # phishing-payment (merged financial+invoice+tax)
    'phishing-payment': {
        'transaction': [
            'wire transfer','ACH debit','direct deposit','crypto payment','PayPal transaction','bank draft','SWIFT transfer',
            'credit card charge','Apple Pay transaction','Google Pay authorization','standing order'
        ],
        'reason': [
            'suspected fraud','limit exceeded','risk score too high','manual verification required','AML flag',
            'temporary suspension','compliance review','chargeback pending','KYC mismatch','pending sanction screen'
        ],
        'origin': [
            'mobile app','ATM machine','online portal','contactless terminal','unverified merchant','overseas IP','foreign ATM','POS terminal'
        ],
        'risk': [
            'account freeze','loss of funds','regulatory audit','temporary hold','chargeback fees','blocked withdrawals','late fees'
        ],
        'doc': [
            'invoice','receipt','purchase order','credit memo','rebate notice','tax form','year-end summary',
            'proforma invoice','debit notice','statement','assessment','refund','adjustment'
        ],
        'vendor': COMPANY_NAMES + ['Northwind Supplies','Fabrikam Metals','Tailspin Logistics','ProseWare Consulting','Adventure Works Europe'],
        'method': [
            'ACH transfer','SWIFT payment','Visa card','MasterCard','American Express','bank deposit','PayPal','direct debit','Apple Pay','company card'
        ],
        'late': [
            'overdue','payment overdue','final warning','pending collection','escalated notice','urgent settlement required','aging item','outstanding balance'
        ],
        'field': [
            'full name','address','bank account number','SWIFT code','tax identification number','national insurance number','employment ID',
            'VAT registration','company registration number','contact phone','date of birth'
        ],
        'office': FAKE_BRANDS['tax'] + ['State Revenue Service','Provincial Tax Office'],
        'channel': [
            'secure portal','encrypted link','tax filing website','online claim center','dedicated refund page','verification page','web form'
        ],
        'time': DATE_SNIPPETS,
    },
    # phishing-delivery
    'phishing-delivery': {
        'status': [
            'missed delivery','customs hold','address issue','fee due','damaged parcel','incorrect postcode','locker full','recipient unavailable'
        ],
        'carrier': FAKE_BRANDS['parcel'],
        'item': [
            'electronics','documents','clothing','equipment','gift','books','replacement parts','office supplies','accessories','medication'
        ],
        'action': [
            'schedule redelivery','confirm address','pay customs','choose pickup','upload ID','select timeslot','verify postcode','authorize release'
        ],
        'time': DATE_SNIPPETS,
    },
    # phishing-techsupport
    'phishing-techsupport': {
        'finding': [
            'malware','ransomware risk','outdated agent','suspicious traffic','phishing site visited','unsupported OS version',
            'unpatched browser','keylogger indicator','blocked attachment','credential stuffing alert'
        ],
        'scope': [
            'workstation','VPN','email account','server session','browser profile','VDI session','laptop','mobile device'
        ],
        'authority': FAKE_BRANDS['it'] + ['Security Operations Center','Endpoint Defense','Corporate IT'],
        'action': [
            'run scan','reset token','isolate device','re-enroll','approve patch','restart agent','rotate password','start remediation'
        ],
        'time': DATE_SNIPPETS,
    },
    # phishing-job
    'phishing-job': {
        'role': [
            'data entry','virtual assistant','tester','brand ambassador','local courier','social media associate','invoice clerk','operations aide'
        ],
        'perk': [
            'remote','flexible hours','weekly pay','no interview','equipment provided','start today','training included','sign-on bonus'
        ],
        'next': [
            'onboarding','background check','training','account setup','tax form','contract signing','portal registration','bank setup'
        ],
        'time': DATE_SNIPPETS,
    },
    # safe-work
    'safe-work': {
        'meeting': ['standup','planning','retro','sync','offsite','all-hands','1:1','workshop','design review','budget review'],
        'artifact': ['deck','notes','brief','roadmap','budget','spec','prototype','minutes','OKRs','risk register'],
        'theme': ['timeline','risks','scope','hiring','vendors','reporting','KPIs','migration','audit prep','Q3 goals'],
        'followup': ['review','sign-off','next steps','Q&A','action items','feedback','revisions','handoff'],
        'cadence': ['weekly','biweekly','monthly','ad-hoc','quarterly'],
    },
    # safe-personal
    'safe-personal': {
        'topic': ['dinner','trip','photos','plants','game night','birthday','movie','hike','brunch','concert','museum','beach day'],
        'tone': ['cheerful','casual','brief','warm','apologetic','excited','low-key'],
        'ask': ['plans?','help?','share?','join?','call?','host?','ideas?','drive?'],
        'time': ['Friday','weekend','tonight','tomorrow','next week','this evening','Saturday','Sunday'],
        'plan': ['at my place','at yours','somewhere central','open to ideas','new cafe','park nearby','along the river','cinema in town'],
    },
    # safe-marketing
    'safe-marketing': {
        'offer': ['20% off','BOGO','free shipping','limited drop','member price','early access','bundle & save','student discount','clearance'],
        'urgency': ['ends soon','this week','today only','while supplies last','final hours','last chance','limited run'],
        'category': ['apparel','home','tech','outdoor','gifts','fitness','beauty','stationery','kitchen'],
        'cta': ['shop now','view collection','claim offer','browse','get early access','unlock deal','see what is new'],
        'brand': FAKE_BRANDS['ecom'] + FAKE_BRANDS['soft'],
    },
    # safe-transactional
    'safe-transactional': {
        'event': [
            'order confirmed','shipped','ready for pickup','password changed','payment received','subscription updated','invoice available',
            'two-step verification enabled','email updated','return processed','delivery scheduled'
        ],
        'channel': ['email','app','SMS','account','dashboard'],
        'detail': ['tracking soon','receipt attached','support available','update preferences','manage order','view history','download PDF'],
        'brand': FAKE_BRANDS['ecom'] + FAKE_BRANDS['soft'],
    },
}

# ------------------------------ Label extras ------------------------------
def id_placeholder() -> str:
    return f"[ID-{random.randint(100000,999999)}]"

LABEL_EXTRAS: Dict[str, List[str]] = {
    # Phishing families (explicit safety)
    'phishing-credential': [
        "SIMULATION: This message is for security training only.",
        "Do not share verification codes with anyone.",
        "Reference: " + id_placeholder(),
    ],
    'phishing-payment': [
        "SIMULATION: Training content. No payment is requested.",
        "We will never ask for your full PIN.",
        "Case: " + id_placeholder(),
    ],
    'phishing-delivery': [
        "SIMULATION: This is a training scenario.",
        "Photo ID may be required at pickup (scenario).",
        "Reference: " + id_placeholder(),
    ],
    'phishing-techsupport': [
        "SIMULATION: Internal phishing drill.",
        "Do not install unapproved software.",
        "Case: " + id_placeholder(),
    ],
    'phishing-job': [
        "SIMULATION: Recruiting lure example.",
        "Never pay fees to apply.",
        "Candidate: " + id_placeholder(),
    ],

    # Safe families
    'safe-work': [
        "I will file the notes in the shared workspace.",
        "Please reply with corrections or additions.",
        "Tracker: " + id_placeholder(),
    ],
    'safe-personal': [
        "No stress if timing is tricky.",
        "Totally fine to suggest another plan.",
    ],
    'safe-marketing': [
        "Offer applies automatically at checkout.",
        "You are receiving this because you opted in. Unsubscribe anytime.",
        "Promo: " + id_placeholder(),
    ],
    'safe-transactional': [
        "You will receive tracking when it ships.",
        "If anything looks off, reply to this email.",
        "Support can help with changes.",
    ],
}

# ------------------------------ Link builder (neutered) -------------------
LINK_PURPOSE_PATH = {
    'portal': ['account','security','update','access','portal'],
    'verify': ['verify','confirm','validate','recheck','auth'],
    'scan':   ['scan','remediate','quarantine','secure','isolate'],
    'form':   ['form','submit','details','update','reclaim'],
    'invoice':['invoice','statement','billing','pay','settle'],
    'priority':['priority','urgent','exec','vip','direct'],
    'apply':  ['apply','onboard','start','enroll','begin'],
    'track':  ['track','parcel','delivery','status','notice'],
}

def brand_slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def neutered_link(label: str, purpose: str) -> str:
    # pick a brand family based on refined label
    if 'credential' in label:
        brand = random.choice(FAKE_BRANDS['cloud'])
    elif 'payment' in label:
        brand = random.choice(FAKE_BRANDS.get('bank', []) + FAKE_BRANDS.get('tax', []))
    elif 'delivery' in label:
        brand = random.choice(FAKE_BRANDS['parcel'])
    elif 'techsupport' in label:
        brand = random.choice(FAKE_BRANDS['it'])
    elif 'job' in label:
        brand = random.choice(FAKE_BRANDS['hr'])
    else:
        brand = random.choice(sum(FAKE_BRANDS.values(), []))
    sub = random.choice(['secure','account','update','verify','notice','auth'])
    host = f"{sub}-{brand_slug(brand)}[.]invalid"
    path = random.choice(LINK_PURPOSE_PATH.get(purpose, ['go']))
    rid = random.randint(100000, 999999)
    return f"hxxps://{host}/{path}?rid={rid}"

# ------------------------------ Utilities --------------------------------
random_choice = random.choice
def pick_name() -> str: return random_choice(NAMES)
def pick_company() -> str: return random_choice(COMPANY_NAMES)
def pick_brand(kind: str) -> str: return random_choice(FAKE_BRANDS.get(kind, ['ExampleCo']))

def money_snippet() -> str:
    cur = random_choice(CURRENCIES)
    sign = CURRENCY_SIGNS[cur]
    amt = random_choice(AMOUNTS)
    return f"{sign}{amt} {cur}"

def subjectify(text: str, max_len: int = 98) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    base = lines[0] if lines else text[:max_len]
    base = re.sub(r"\s+", " ", base)
    if len(base) > max_len: base = base[: max_len - 3] + "..."
    if base: base = base[0:1].upper() + base[1:]
    return base

def assemble_email(subject: str, body: str) -> str:
    return f"Subject: {subject}\n\n{body}"

def normalize_for_hash(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+|hxxps?://\S+", "", t)
    t = re.sub(r"\[[^\]]+\]", " ", t)  # strip placeholders/brackets
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    return len(a & b) / float(len(a | b))

def token_ngrams(text: str, n: int = 3) -> Set[str]:
    toks = normalize_for_hash(text).split()
    return {" ".join(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1))}

# ------------------------------ Content Guards ---------------------------
CONTENT_GUARDS: Dict[str, List[str]] = {
    'safe-personal': [r'\bunsubscribe\b', r'\baccount\b', r'\bverify\b', r'\bportal\b', r'\breset\b',
                      r'\bpassword\b', r'\bpayment\b', r'\binvoice\b', r'\border\b', r'\btracking\b',
                      r'\btax\b', r'\bcustoms\b', r'\bdelivery\b', r'\brefund\b'],
    'safe-work': [r'\bunsubscribe\b', r'\bverify\b', r'\bpassword\b', r'\breset\b', r'\binvoice\b',
                  r'\btracking\b', r'\bcustoms\b', r'\brefund\b'],
    'safe-marketing': [r'\bpassword\b', r'\b2fa\b', r'\bmalware\b', r'\bquarantine\b'],
    'safe-transactional': [r'\bgift cards?\b'],
    # Phishing classes: avoid marketing footers
    'phishing-credential': [r'\bunsubscribe\b'],
    'phishing-payment': [r'\bunsubscribe\b'],
    'phishing-delivery': [r'\bunsubscribe\b'],
    'phishing-techsupport': [r'\bunsubscribe\b'],
    'phishing-job': [r'\bunsubscribe\b'],
}
def passes_content_guard(label: str, subject: str, body: str) -> bool:
    text = f"{subject}\n{body}".lower()
    for pat in CONTENT_GUARDS.get(label, []):
        if re.search(pat, text): return False
    return True

# ------------------------------ Rephrasing --------------------------------
REWRITE_PATTERNS = [
    (re.compile(r'\bplease\b', re.I), lambda: random_choice(['kindly','please','if you could'])),
    (re.compile(r'\bverify\b', re.I), lambda: random_choice(['verify','confirm','check','revalidate'])),
    (re.compile(r'\brefund\b', re.I), lambda: random_choice(['refund','rebate','credit'])),
    (re.compile(r'\border\b', re.I), lambda: random_choice(['order','purchase','transaction'])),
    (re.compile(r'\bissue\b', re.I), lambda: random_choice(['issue','problem','incident','matter'])),
    (re.compile(r'\bnow\b', re.I), lambda: random_choice(['now','right away','as soon as possible'])),
]
def rephrase(text: str) -> str:
    t = text
    for rx, repl in REWRITE_PATTERNS:
        t = rx.sub(lambda _: repl(), t)
    t = t.replace(' — ', ' - ').replace('…', '...')
    return t

# ------------------------------ Template Engine ---------------------------
@dataclass
class Sample:
    subject: str
    body: str
    def render(self) -> str:
        return assemble_email(self.subject, self.body)

class Generator:
    def __init__(self, label: str): self.label = label
    def generate_one(self) -> Sample: raise NotImplementedError
    def greet(self) -> str: return f"{random_choice(GREETINGS)} {pick_name()},"
    def signoff(self) -> str: return f"{random_choice(SIGNOFFS)},\n{pick_name()}"
    def maybe_hedge(self) -> str: return random_choice(["", random_choice(HEDGES) + " "]).strip()
    def urgency_phrase(self) -> str: return random_choice(["", random_choice(URGENCY)])
    def lengthen(self, paragraphs: List[str]) -> List[str]:
        raw = LABEL_EXTRAS.get(self.label, [])
        expanded = []
        for item in raw:
            # refresh IDs each time
            expanded.append(re.sub(r'\[ID-\d+\]', id_placeholder(), item))
        k = random.randint(0, 2)
        return paragraphs + random.sample(expanded, k) if k and len(expanded) >= k else paragraphs

# ------------------------------ Phishing Generators (sanitized) -----------
class PhishingCredential(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['phishing-credential'].items()}
        brand = pick_brand('cloud')
        act = random_choice(['reset your password','confirm your login','verify unusual sign-in','re-enable access','confirm your identity'])
        s1 = f"{brand} {sc['trigger']} on {sc['device']} in {sc['geo']} ({self.urgency_phrase() or 'review'})"
        b = [
            self.greet(),
            f"We {self.maybe_hedge()}detected {sc['trigger']} for your {sc['asset']} from {sc['device']} in {sc['geo']}.",
            f"To keep your account safe, kindly {act} {random_choice(DATE_SNIPPETS)} via {neutered_link('phishing-credential','portal')}.",
            f"Open security page: {neutered_link('phishing-credential','verify')}",
            "SIMULATION: This content is for training only.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class PhishingPayment(Generator):
    def generate_one(self) -> Sample:
        sc = SCENARIO['phishing-payment']
        subtype = random_choice(['bank','invoice','tax'])
        if subtype == 'bank':
            brand = pick_brand('bank'); amt = money_snippet()
            reason = random_choice(sc['reason']); trans = random_choice(sc['transaction'])
            origin = random_choice(sc['origin']); risk = random_choice(sc['risk'])
            s1 = f"{brand}: {trans} {reason} - {amt} ({self.urgency_phrase() or 'review required'})"
            b = [
                self.greet(),
                f"A {trans} of {amt} from {origin} is on hold due to {reason}.",
                f"Confirm to avoid {risk}: {neutered_link('phishing-payment','verify')}",
                "SIMULATION: Do not provide real payment info.",
                self.signoff(),
            ]
        elif subtype == 'invoice':
            vendor = random_choice(sc['vendor']); amt = money_snippet()
            doc = random_choice(sc['doc']); late = random_choice(sc['late']); method = random_choice(sc['method'])
            s1 = f"{doc.capitalize()} {random.randint(1000,9999)} {late} - {amt} payable to {vendor}"
            b = [
                self.greet(),
                f"Kindly review the online {doc} for {amt} payable via {method}.",
                f"Pay securely via: {neutered_link('phishing-payment','invoice')}",
                "SIMULATION: No payment is due; training example only.",
                self.signoff(),
            ]
        else:  # tax
            brand = random_choice(sc['office']); amt = money_snippet()
            doc = random_choice([d for d in sc['doc'] if d in ['rebate notice','tax form','statement','refund','assessment','adjustment']])
            field = random_choice(sc['field']); channel = random_choice(sc['channel'])
            s1 = f"{brand}: {doc} available - verify {field} to receive {amt}"
            b = [
                self.greet(),
                f"A {doc} of {amt} is ready; {self.maybe_hedge()}your {field} needs reconfirmation.",
                f"Submit via {channel}: {neutered_link('phishing-payment','form')}",
                "SIMULATION: Never enter personal data on unknown sites.",
                self.signoff(),
            ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class PhishingDelivery(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['phishing-delivery'].items()}
        fee = money_snippet()
        s1 = f"{sc['status'].capitalize()} - {sc['carrier']}: {sc['action']} required"
        b = [
            self.greet(),
            f"Your {sc['carrier']} parcel containing {sc['item']} is on hold due to {sc['status']} and a charge of {fee}.",
            f"{sc['action'].capitalize()} here: {neutered_link('phishing-delivery','track')}",
            "SIMULATION: Delivery scenario for security awareness.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class PhishingTechSupport(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['phishing-techsupport'].items()}
        brand = random_choice(SCENARIO['phishing-techsupport']['authority'])
        s1 = f"{brand}: {sc['finding']} on {sc['scope']} ({self.urgency_phrase() or 'security review'})"
        b = [
            self.greet(),
            f"Your {sc['scope']} triggered an automatic quarantine {self.maybe_hedge()}after detecting {sc['finding']}.",
            f"{sc['action'].capitalize()} now: {neutered_link('phishing-techsupport','scan')}",
            "SIMULATION: IT support lure for training only.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class PhishingJob(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['phishing-job'].items()}
        brand = pick_brand('hr')
        s1 = f"{brand}: shortlisted for {sc['role']} - {sc['perk']}"
        b = [
            self.greet(),
            f"Your profile impressed our team for a {sc['perk']} {sc['role']} position.",
            f"To proceed with {sc['next']} {random_choice(DATE_SNIPPETS)}, complete the portal setup: {neutered_link('phishing-job','apply')}",
            "SIMULATION: Recruiting lure; do not share personal data.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

# ------------------------------ Safe Generators ---------------------------
class SafeWork(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['safe-work'].items()}
        company = pick_company()
        s1 = f"{company} {sc['meeting']} {sc['cadence']} - {sc['theme']} / {sc['followup']}"
        b = [
            self.greet(),
            f"Sharing the {sc['artifact']} from our {sc['meeting']} ({sc['cadence']}).",
            f"Main points: {sc['theme']} and {sc['followup']}.",
            "Please review before our next session and add comments in the doc.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class SafePersonal(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['safe-personal'].items()}
        s1 = f"{sc['topic'].capitalize()} {sc['time']}?"
        b = [
            self.greet(),
            f"Quick note ({sc['tone']}): thinking {sc['topic']} {sc['time']} {sc['plan']}.",
            f"Let me know if you can {sc['ask']}",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class SafeMarketing(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['safe-marketing'].items()}
        brand = random_choice(SCENARIO['safe-marketing']['brand'])
        s1 = f"{brand}: {sc['offer']} on {random_choice(SCENARIO['safe-marketing']['category'])} ({sc['urgency']})"
        b = [
            self.greet(),
            f"{sc['offer']} {sc['urgency']} — {sc['cta'].capitalize()}",
            "Offer applies at checkout. Terms may apply.",
            "You are receiving this because you opted in. Unsubscribe anytime.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

class SafeTransactional(Generator):
    def generate_one(self) -> Sample:
        sc = {k: random_choice(v) for k, v in SCENARIO['safe-transactional'].items()}
        brand = random_choice(SCENARIO['safe-transactional']['brand'])
        s1 = f"{brand}: {sc['event'].capitalize()}"
        b = [
            self.greet(),
            f"This is a confirmation via {sc['channel']}: {sc['event']}.",
            f"For details, {random_choice(sc['detail'])}.",
            "If anything looks off, reply to this email.",
            self.signoff(),
        ]
        body = "\n\n".join(self.lengthen(b))
        return Sample(subject=subjectify(s1), body=rephrase(body))

# ------------------------------ Registry ----------------------------------
GENERATOR_REGISTRY: Dict[str, Generator] = {
    'phishing-credential': PhishingCredential('phishing-credential'),
    'phishing-payment':    PhishingPayment('phishing-payment'),
    'phishing-delivery':   PhishingDelivery('phishing-delivery'),
    'phishing-techsupport':PhishingTechSupport('phishing-techsupport'),
    'phishing-job':        PhishingJob('phishing-job'),
    'safe-work':           SafeWork('safe-work'),
    'safe-personal':       SafePersonal('safe-personal'),
    'safe-marketing':      SafeMarketing('safe-marketing'),
    'safe-transactional':  SafeTransactional('safe-transactional'),
}

# ------------------------------ Generation Loop ---------------------------
def generate_samples(per_label: int, seed: int, jaccard_threshold: float = 0.45, max_tries_factor: int = 40):
    random.seed(seed)
    out: List[Tuple[str,str]] = []
    seen_ngrams: List[Set[str]] = []

    # Generate per label to ensure strict balance
    for label in LABELS:
        generator = GENERATOR_REGISTRY[label]
        need = per_label
        tries = 0
        max_tries = per_label * max_tries_factor
        while need > 0 and tries < max_tries:
            tries += 1
            sample = generator.generate_one()

            # Guards
            if not ENGLISH_ONLY_PATTERN.match(sample.render()):
                continue
            if not passes_content_guard(label, sample.subject, sample.body):
                continue

            # Dedup by trigram Jaccard (global)
            grams = token_ngrams(sample.render())
            if any(jaccard(grams, g) >= jaccard_threshold for g in seen_ngrams):
                continue

            seen_ngrams.append(grams)
            out.append((sample.render(), label))
            need -= 1

        if need > 0:
            # If we couldn't fill due to uniqueness constraints, relax threshold slightly for this label
            tries_relax = 0
            while need > 0 and tries_relax < per_label * 10:
                tries_relax += 1
                sample = generator.generate_one()
                if not ENGLISH_ONLY_PATTERN.match(sample.render()):
                    continue
                if not passes_content_guard(label, sample.subject, sample.body):
                    continue
                grams = token_ngrams(sample.render())
                if any(jaccard(grams, g) >= (jaccard_threshold + 0.1) for g in seen_ngrams):
                    continue
                seen_ngrams.append(grams)
                out.append((sample.render(), label))
                need -= 1

    return out

def finalize_balance(rows: List[Tuple[str,str]], per_label: int) -> List[Tuple[str,str]]:
    # Keep up to per_label for each label in label order
    buckets: Dict[str, List[Tuple[str,str]]] = {l: [] for l in LABELS}
    for email, lab in rows:
        if len(buckets.get(lab, [])) < per_label:
            buckets[lab].append((email, lab))
    merged: List[Tuple[str,str]] = []
    for l in LABELS:
        merged.extend(buckets[l])
    return merged

# ------------------------------ CLI ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate refined synthetic emails (balanced across labels).")
    ap.add_argument("--out", type=str, default="emails_refined.csv", help="Output CSV path")
    ap.add_argument("--per-label", type=int, default=1000, help="Samples per label")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--jaccard-threshold", type=float, default=0.45, help="Near-duplicate threshold (higher = stricter)")
    args = ap.parse_args()

    rows = generate_samples(per_label=args.per_label, seed=args.seed, jaccard_threshold=args.jaccard_threshold)
    rows = finalize_balance(rows, per_label=args.per_label)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["email","label"])
        for email, lab in rows:
            w.writerow([email, lab])

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
