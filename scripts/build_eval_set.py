"""Author the 150-question eval set, drafted from the corpus for adjudication.

SPEC's M4.1 and M4.2. Every question here was written against text actually
present in `corpus/chunks.sqlite`, and every citation names a page that exists in
`corpus/manifest.jsonl`. The build refuses to emit a set that breaks either rule,
so a question written from memory rather than from the corpus fails here rather
than at judging time.

These are drafts. The brief's own warning is the reason this file exists as a
script rather than as a hand-edited JSONL: text-to-SQL benchmarks measured label
error rates of 52.8% and 62.8%, and a model-authored gold answer is exactly the
kind of label that looks right and is not. M4.3 and M4.4 are the audit that
catches what is wrong here, and the disagreement rate they produce is a published
number rather than a formality.

    uv run python scripts/build_eval_set.py            # write eval/questions.jsonl
    uv run python scripts/build_eval_set.py --check    # validate without writing

Columns in each tuple: id suffix, type, language, difficulty, question,
gold answer, citations. Unanswerable rows carry a note instead of citations,
because the note is the evidence that the absence was checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dastavez.evalset import (  # noqa: E402
    Question,
    content_hash,
    coverage_errors,
    dump,
    structural_errors,
)

CHUNKS = Path("corpus/chunks.sqlite")
RUN_ID = "pypdf.strip.fixed.none.d5c4ff3b99cd"

# Shorthand for the documents cited most, so a citation reads as a claim about a
# page rather than as a string nobody checks.
FAQ = "pmkisan-faq"
FAQR = "pmkisan-faq-revised"
FAQA = "pmkisan-faq-additional"
CLAR = "pmkisan-clauses-clarification"
KMY = "pmkmy-faq"
UJJ = "ujjwala-deprivation-en"

# Recorded on every drafted row, so the M4.3 audit is a comparison between two
# named annotators rather than a review of anonymous text.
DRAFT_ANNOTATOR = "draft-claude-opus-5-2026-08-31"

E, H, HG = "en", "hi", "hinglish"
EASY, MED, HARD = "easy", "medium", "hard"

# Language rebalance, applied at build time so the citations above stay the single
# place a page is named. SPEC reports language_tag separately because an aggregate
# over three languages hides which one the system is bad at, and a slice of two
# questions cannot carry a number. These are translations of questions already in
# the set rather than additions, because SPEC sizes the set at 150 and says not to
# pad it further.
#
# The question's language is not the document's language. A Hindi question against
# an English page is what this corpus's users actually ask, and it is the harder
# retrieval case, which is the point of tagging it.
TRANSLATED = {
    # --- to Hindi ---
    "dz-003": (H, "पीएम-किसान योजना किस तारीख से प्रभावी है?", "01.12.2018 से।"),
    "dz-019": (H, "पीएम-किसान में गलत घोषणा देने वाले लाभार्थी का क्या होता है?",
     "वह हस्तांतरित वित्तीय लाभ की वसूली और कानून के अनुसार अन्य दंडात्मक कार्रवाई के लिए उत्तरदायी होगा।"),
    "dz-035": (H, "क्या पिछले साल आयकर भरने वाला किसान पीएम-किसान का लाभ ले सकता है?",
     "नहीं। पिछले निर्धारण वर्ष में आयकर देने वाले सभी व्यक्ति अपात्र हैं।"),
    "dz-038": (H, "क्या संस्थागत भूमिधारक पीएम-किसान के पात्र हैं?",
     "नहीं। सभी संस्थागत भूमिधारक अपात्र हैं।"),
    "dz-050": (H, "क्या जिस घर में पहले से एलपीजी कनेक्शन है, वह उज्ज्वला के लिए पात्र है?",
     "नहीं। पहले से मौजूद एलपीजी कनेक्शन एक बहिष्करण मानदंड है।"),
    "dz-056": (H, "पीएम-किसान मान-धन योजना में 18 वर्ष की आयु पर प्रवेश करने पर मासिक अंशदान कितना है?",
     "55 रुपये प्रति माह।"),
    "dz-057": (H, "पीएम-किसान मान-धन योजना में 40 वर्ष की आयु पर प्रवेश करने पर मासिक अंशदान कितना है?",
     "200 रुपये प्रति माह।"),
    "dz-077": (H,
     "एक भूमिधारक की मार्च 2019 में मृत्यु हो गई और उसके पुत्र को भूमि विरासत में मिली। "
     "क्या कट-ऑफ तिथि के बावजूद पुत्र पीएम-किसान के लिए पात्र है?",
     "हाँ। भूमिधारक की मृत्यु के कारण उत्तराधिकार से स्वामित्व का हस्तांतरण 01.02.2019 की "
     "कट-ऑफ तिथि का घोषित अपवाद है।"),
    "dz-080": (H,
     "पति, पत्नी और दो अवयस्क बच्चों के परिवार में प्रत्येक के नाम 0.8 हेक्टेयर भूमि है। "
     "परिवार को पीएम-किसान से कितना लाभ मिलेगा?",
     "कुल मिलाकर 6000 रुपये प्रति वर्ष। परिवार के अलग-अलग सदस्यों के नाम की भूमि जोड़ी जाती है, "
     "और परिवार के लिए अधिकतम लाभ 6000 रुपये ही रहता है।"),
    "dz-099": (H, "यदि किसी किसान का नाम पीएम-किसान लाभार्थी सूची में नहीं है तो उसे क्या करना चाहिए?",
     "अपने जिले की जिला स्तरीय शिकायत निवारण निगरानी समिति से नाम शामिल कराने के लिए संपर्क करना चाहिए।"),
    "dz-105": (H, "क्या पीएम-किसान के लिए बैंक खाते का विवरण देना अनिवार्य है?",
     "हाँ। लाभार्थियों को आधार संख्या सहित बैंक खाता विवरण देना होगा, और बैंक खाता विवरण के "
     "बिना कोई लाभ नहीं दिया जा सकता।"),
    "dz-111": (H, "उज्ज्वला कनेक्शन लेने से पहले आवेदक को क्या घोषित करना होता है?",
     "कि आवेदक और उसके परिवार के सदस्य, परिवार संरचना के अनुसार, सूचीबद्ध बहिष्करण मानदंडों में "
     "से किसी को पूरा नहीं करते।"),
    "dz-121": (H, "पीएम-किसान का लाभ लाभार्थी तक कैसे पहुँचाया जाता है?",
     "आय सहायता का लाभ सीधे लाभार्थियों के बैंक खातों में जमा किया जाता है।"),
    "dz-124": (H, "वित्त वर्ष 2027-28 के लिए पीएम-किसान की किस्त की राशि कितनी है?",
     "कोष में इसका उल्लेख नहीं है।"),
    "dz-126": (H, "पीएम-किसान का टोल फ्री हेल्पलाइन नंबर क्या है?", "कोष में इसका उल्लेख नहीं है।"),
    "dz-128": (H, "पीएम-किसान मान-धन योजना का वार्षिक बजट आवंटन कितना है?",
     "कोष में इसका उल्लेख नहीं है।"),
    "dz-131": (H, "2025-26 में उज्ज्वला के तहत कितने एलपीजी कनेक्शन जारी किए गए?",
     "कोष में इसका उल्लेख नहीं है।"),
    "dz-135": (H, "पीएम-किसान पंजीकरण को स्वीकृत होने में कितना समय लगता है?",
     "कोष में इसका उल्लेख नहीं है।"),
    "dz-142": (H, "क्या पीएम-किसान लाभार्थी लाभ प्राप्त करने के लिए किसी को नामांकित कर सकता है?",
     "कोष में इसका उल्लेख नहीं है।"),
    "dz-149": (H, "पीएम-किसान मान-धन योजना में सबसे अधिक नामांकन किन राज्यों में है?",
     "कोष में इसका उल्लेख नहीं है।"),
    # --- to Hinglish ---
    "dz-002": (HG, "PM-KISAN ka paisa kitni installments mein aur kitna kitna milta hai?",
     "Teen barabar installments mein, har ek Rs. 2000 ki, har chaar mahine par."),
    "dz-005": (HG, "PM-KISAN ki eligibility tay karne ki cut-off date kya hai?", "01.02.2019."),
    "dz-008": (HG, "PM-KMY join karne ki umar kya honi chahiye?", "18 se 40 saal."),
    "dz-031": (HG, "Kya serving Central Government employee jiske paas kheti ki zameen hai, "
     "PM-KISAN le sakta hai?",
     "Nahi. Central aur State Government ke serving aur retired officers aur "
     "employees excluded hain."),
    "dz-033": (HG, "Ek retired officer ki monthly pension Rs. 12,000 hai. Kya uska parivar "
     "PM-KISAN ke liye eligible hai?",
     "Nahi. Rs. 10,000 ya usse zyada monthly pension wale retired pensioners excluded hain, "
     "sivaay Multi Tasking Staff, Class IV aur Group D ke."),
    "dz-039": (HG, "Kya baithe hue MLA PM-KISAN ke liye eligible hain?",
     "Nahi. Lok Sabha, Rajya Sabha, State Legislative Assemblies aur Councils ke poorv aur "
     "vartmaan sadasya excluded hain."),
    "dz-058": (HG, "PM-KMY mein 29 saal ki umar par entry lene par member ka monthly contribution "
     "kitna hai?",
     "Rs. 100 per month."),
    "dz-060": (HG, "PM-KMY mein 30 saal ki entry age par member aur Government milakar total "
     "monthly contribution kitna hai?",
     "Rs. 210, jisme Rs. 105 member ka aur Rs. 105 Government ka hai."),
    "dz-074": (HG, "Ek kisan ke paas 5 hectare zameen hai aur woh income tax nahi bharta. Kya "
     "woh aaj PM-KISAN ke liye eligible hai, aur launch ke waqt tha?",
     "Aaj eligible hai, launch ke waqt nahi tha. Shuru mein sirf 2 hectare tak ke small aur "
     "marginal farmers covered the, aur 1.6.2019 se scheme sabhi farmer families ke liye "
     "extend kar di gayi."),
    "dz-082": (HG, "Ek 42 saal ka chhota kisan PM-KMY se pension lena chahta hai. Kya woh join "
     "kar sakta hai?",
     "Nahi. Entry age group 18 se 40 saal hai, to 42 par woh join nahi kar sakta, chahe baaki "
     "criteria poore hon."),
    "dz-084": (HG, "Ek ghar ke paas 6 acre irrigated zameen hai jo do crop seasons mein use hoti "
     "hai, aur koi LPG connection nahi hai. Kya woh Ujjwala ke liye eligible hai?",
     "Nahi. Do ya zyada crop seasons ke liye 5 acre ya usse zyada irrigated zameen ek "
     "exclusion criterion hai, aur LPG connection na hona isse override nahi karta."),
    "dz-101": (HG, "PM-KISAN portal ke Farmers Corner mein kaun si teen suvidhayein hain?",
     "New Farmer Registration, Edit Aadhaar details, aur Beneficiary Status."),
    "dz-104": (HG, "Jahan Aadhaar available nahi hai, wahan PM-KISAN ke liye kaun se alternate "
     "documents chalte hain?",
     "Aadhaar enrolment number, driving licence, voter ID card, NREGA job card, ya Central, "
     "State ya UT Government dwara jaari koi anya pehchaan patra."),
    "dz-108": (HG, "PM-KISAN mein benefit sanction hone ki jaankari laabharthi ko kaise milti hai?",
     "State ya UT dwara system generated SMS se, aur Panchayat par lagayi gayi beneficiary "
     "list se bhi."),
    "dz-112": (HG, "Agar Ujjwala ki declaration jhooti nikle to Oil Marketing Company kya kar "
     "sakti hai?",
     "Gas supply band kar sakti hai, connection terminate kar sakti hai, equipment seize kar "
     "sakti hai, bank account mein bheji gayi subsidy vasool kar sakti hai, aur legal action "
     "le sakti hai."),
    "dz-125": (HG, "Bihar mein PM-KISAN ke kitne laabharthi hain?",
     "Corpus mein iska ullekh nahi hai."),
    "dz-132": (HG, "Ujjwala ke tehat LPG cylinder ka abhi kya daam hai?",
     "Corpus mein iska ullekh nahi hai."),
    "dz-137": (HG, "PM-KMY mein 60 saal se pehle ke contributions par kitna interest milta hai?",
     "Corpus mein iska ullekh nahi hai."),
    "dz-141": (HG, "Agar PM-KISAN laabharthi hamesha ke liye videsh chala jaye to kya hota hai?",
     "Corpus mein iska ullekh nahi hai."),
    "dz-147": (HG, "Eligibility dobara jaanchne ke baad PM-KISAN se kitne kisan hataye gaye?",
     "Corpus mein iska ullekh nahi hai."),
}


# (suffix, type, lang, difficulty, question, gold_answer, [(doc, page), ...])
FACTUAL = [
    ("001", E, EASY, "How much money does a farmer family get per year under PM-KISAN?",
     "Rs. 6000 per annum per family.", [(FAQ, 1)]),
    ("002", E, EASY, "In how many installments is the PM-KISAN benefit paid, and of what size?",
     "Three equal installments of Rs. 2000 each, every four months.", [(FAQ, 2)]),
    ("003", E, EASY, "From what date does the PM-KISAN scheme take effect?",
     "01.12.2018.", [(FAQ, 1)]),
    ("004", E, MED, "On what date was PM-KISAN launched by the Prime Minister?",
     "24th February 2019.", [(FAQR, 1)]),
    ("005", E, MED, "What is the cut-off date for determining eligibility under PM-KISAN?",
     "01.02.2019.", [(FAQ, 3)]),
    ("006", E, MED, "How is a landholder farmer family defined under PM-KISAN?",
     "A family comprising husband, wife and minor children who own cultivable land as per "
     "the land records of the concerned State or UT.", [(FAQR, 4)]),
    ("007", E, EASY, "What monthly pension does PM-KMY pay on attaining 60?",
     "Rs. 3000 per month.", [(KMY, 1)]),
    ("008", E, EASY, "What is the entry age group for PM-KMY?",
     "18 to 40 years.", [(KMY, 1)]),
    ("009", E, MED, "What is the cut-off date for eligibility under PM-KMY?",
     "01.08.2019.", [(KMY, 2)]),
    ("010", E, MED, "Under PM-KMY, what does a subscriber's spouse receive if the subscriber "
     "dies while drawing pension?",
     "50 percent of the pension the beneficiary was receiving, as family pension, provided "
     "the spouse is not already a beneficiary of the scheme.", [(KMY, 1)]),
    ("011", E, MED, "How does PM-KMY define a small and marginal landholder farmer?",
     "A farmer who owns cultivable land up to 2 hectare as per the land record of the "
     "concerned State or UT.", [(KMY, 1)]),
    ("012", E, HARD, "Within a PM-KISAN farmer family holding land in several members' names, "
     "who receives the payment?",
     "The person in the family holding the highest quantum of land; if two or more members "
     "hold the same quantum, the eldest among them.", [(FAQ, 7)]),
    ("013", E, MED, "Is a mobile number mandatory for PM-KISAN enrollment?",
     "No. It is advised where available so that transfer information can be communicated, "
     "but it is not mandatory.", [(FAQR, 5)]),
    ("014", E, MED, "Where are PM-KISAN beneficiary lists displayed?",
     "At Panchayats, and States or UTs also notify sanction by system generated SMS.",
     [(FAQ, 6)]),
    ("015", E, HARD, "Which three States or UTs were exempted from the mandatory Aadhaar "
     "requirement for PM-KISAN, and until when?",
     "Assam, Meghalaya and Jammu and Kashmir, until 31st March 2020.", [(FAQR, 4)]),
    ("016", E, MED, "Under PM-KISAN, is land in urban areas covered?",
     "Yes. There is no distinction between urban and rural cultivable land, provided land "
     "in urban areas is under actual cultivation.", [(FAQA, 1)]),
    ("017", E, MED, "What is the cut-off date for a minor child becoming major under PM-KISAN?",
     "1.2.2019.", [(FAQA, 1)]),
    ("018", E, HARD, "Under PM-KMY, what happens to contributions if a subscriber is found to "
     "have given an incorrect declaration?",
     "The beneficiary gets back their contributions without any interest, and the Central "
     "Government's matching contribution is stopped.", [(KMY, 3)]),
    ("019", E, MED, "What happens to a PM-KISAN beneficiary who gives an incorrect declaration?",
     "They are liable for recovery of the transferred financial benefit and other penal "
     "actions as per law.", [(FAQ, 3)]),
    ("020", E, HARD, "Does the Government contribute to PM-KMY alongside the farmer?",
     "Yes. The Government contributes an amount matching the member's contribution, so the "
     "total is twice the member's share.", [(KMY, 2)]),
    ("021", HG, EASY, "PM-KISAN mein ek saal mein kitne paise milte hain?",
     "Rs. 6000 per annum per family.", [(FAQ, 1)]),
    ("022", HG, MED, "PM-KISAN ki eligibility ke liye cut-off date kya hai?",
     "01.02.2019.", [(FAQ, 3)]),
    ("023", HG, MED, "PM-KMY mein 60 saal ke baad kitni monthly pension milti hai?",
     "Rs. 3000 per month.", [(KMY, 1)]),
    ("024", HG, HARD, "Agar PM-KISAN family ke paas alag alag members ke naam par zameen hai, "
     "to paisa kise milega?",
     "Jis member ke paas sabse zyada zameen hai; barabar hone par sabse bade member ko.",
     [(FAQ, 7)]),
    ("025", H, MED, "पीएम-किसान मान-धन योजना में न्यूनतम सुनिश्चित पेंशन कितनी है?",
     "60 वर्ष की आयु प्राप्त करने के बाद 3000 रुपये प्रति माह।", [(KMY, 11)]),
    ("026", H, HARD, "पीएम-किसान मान-धन योजना में प्रवेश की आयु सीमा क्या है?",
     "18 से 40 वर्ष।", [(KMY, 11)]),
    ("027", E, MED, "How many exclusion criteria does the Ujjwala deprivation declaration list?",
     "Thirteen, grouped into income and employment, asset and wealth, and existing LPG "
     "connection.", [(UJJ, 1), (UJJ, 2)]),
    ("028", E, HARD, "Under PM-KISAN, what is done with land held by a family across different "
     "revenue records in different villages?",
     "The entire land held by the family is pooled together, and the maximum benefit remains "
     "Rs. 6000 per annum.", [(FAQ, 6)]),
    ("029", E, MED, "What was the original landholding ceiling when PM-KISAN was launched?",
     "Two hectare of combined landholding, since benefits were initially admissible only to "
     "small and marginal farmer families.", [(FAQR, 1)]),
    ("030", E, HARD, "From what date was PM-KISAN extended beyond small and marginal farmers?",
     "1.6.2019, when the scheme was revised to cover all farmer families irrespective of the "
     "size of their landholding.", [(FAQR, 1)]),
]

EXCLUSION = [
    ("031", E, EASY, "Is a serving Central Government employee who owns farmland eligible for "
     "PM-KISAN?",
     "No. Serving and retired officers and employees of Central and State Government "
     "Ministries, Offices, Departments and their field units are excluded.", [(FAQ, 3)]),
    ("032", E, MED, "Is a retired Class IV government employee excluded from PM-KISAN?",
     "No. Multi Tasking Staff, Class IV and Group D employees are explicitly excluded from "
     "the exclusion, so they remain eligible if otherwise qualified.", [(FAQ, 3)]),
    ("033", E, MED, "A retired official draws a monthly pension of Rs. 12,000. Is the family "
     "eligible for PM-KISAN?",
     "No. Superannuated or retired pensioners with a monthly pension of Rs. 10,000 or more "
     "are excluded, other than Multi Tasking Staff, Class IV and Group D.", [(FAQ, 2)]),
    ("034", E, MED, "A retired official draws a monthly pension of Rs. 8,000. Does the pension "
     "exclusion apply?",
     "No. The exclusion applies at Rs. 10,000 per month or more, so a pension of Rs. 8,000 "
     "does not trigger it.", [(FAQ, 2)]),
    ("035", E, EASY, "Is a farmer who paid income tax last year eligible for PM-KISAN?",
     "No. All persons who paid income tax in the last assessment year are excluded.",
     [(FAQ, 2)]),
    ("036", E, MED, "A farmer's wife paid income tax last year. Is the family eligible for "
     "PM-KISAN?",
     "No. If any member of the family is an income tax payee in the last assessment year, "
     "the family is not eligible.", [(FAQ, 4)]),
    ("037", E, MED, "Is a practising doctor who owns farmland eligible for PM-KISAN?",
     "No. Professionals such as doctors, engineers, lawyers, chartered accountants and "
     "architects registered with professional bodies and carrying out practice are excluded.",
     [(FAQ, 2)]),
    ("038", E, EASY, "Are institutional land holders eligible for PM-KISAN?",
     "No. All institutional land holders are excluded.", [(FAQ, 1)]),
    ("039", E, MED, "Is a sitting Member of a State Legislative Assembly eligible for PM-KISAN?",
     "No. Former and present Members of Lok Sabha, Rajya Sabha, State Legislative Assemblies "
     "and State Legislative Councils are excluded.", [(FAQ, 2)]),
    ("040", E, MED, "Is a former Mayor of a Municipal Corporation eligible for PM-KISAN?",
     "No. Former and present Mayors of Municipal Corporations are excluded.", [(FAQ, 2)]),
    ("041", E, HARD, "Is a tenant farmer who cultivates land owned by someone else eligible "
     "for PM-KISAN?",
     "No. Land holding in the farmer's own name is the sole criteria, so a tenant farmer "
     "cultivating another person's land is not eligible.", [(FAQR, 4)]),
    ("042", E, HARD, "A man cultivates land still recorded in his father's name. Is he eligible "
     "for PM-KISAN?",
     "No. The land must be in his own name. He becomes eligible only if ownership is "
     "transferred to him on account of succession.", [(FAQR, 4)]),
    ("043", E, MED, "Is agricultural land being used for a non-agricultural purpose covered by "
     "PM-KISAN?",
     "No. Agricultural land being used for non-agricultural purpose is not covered.",
     [(FAQA, 1)]),
    ("044", E, MED, "Are micro land holdings that are not cultivable covered by PM-KISAN?",
     "No. Micro land holdings which are not cultivable are excluded from the benefit.",
     [(FAQA, 1)]),
    ("045", E, HARD, "A farmer sells his land through a sale deed after 01.02.2019. Is the "
     "buyer eligible for PM-KISAN?",
     "No. The transferee is not eligible because he was not the land owner as on 01.02.2019.",
     [(CLAR, 1)]),
    ("046", E, HARD, "After selling all his cultivable land through a gift deed, does the "
     "seller remain eligible for PM-KISAN?",
     "No. The transferor becomes ineligible if the family has no cultivable land left after "
     "the transfer.", [(CLAR, 1)]),
    ("047", E, MED, "Is a farmer already enrolled in the National Pension Scheme eligible for "
     "PM-KMY?",
     "No. Small and marginal farmers covered under other statutory social security schemes "
     "such as NPS, ESIC and EPFO are ineligible.", [(KMY, 1)]),
    ("048", E, MED, "Is a farmer who has opted for Pradhan Mantri Shram Yogi Maan Dhan Yojana "
     "eligible for PM-KMY?",
     "No. Farmers who have opted for PM-SYM are ineligible for PM-KMY.", [(KMY, 1)]),
    ("049", E, MED, "Is a farmer owning 3 hectares of cultivable land eligible for PM-KMY?",
     "No. Any individual farmer owning more than 2 hectare of cultivable land is not "
     "eligible.", [(KMY, 3)]),
    ("050", E, EASY, "Does a household with an existing LPG connection qualify under Ujjwala?",
     "No. An existing LPG connection is an exclusion criterion.", [(UJJ, 2)]),
    ("051", E, MED, "A household member earns Rs. 12,000 per month. Does the household qualify "
     "under Ujjwala?",
     "No. Any member earning more than Rs. 10,000 per month is an exclusion criterion.",
     [(UJJ, 1)]),
    ("052", E, MED, "Does owning a Kisan credit card with a credit limit of Rs. 60,000 exclude "
     "a household from Ujjwala?",
     "Yes. Possessing a Kisan credit card with a credit limit over Rs. 50,000 is an "
     "exclusion criterion.", [(UJJ, 1)]),
    ("053", E, HARD, "Does owning a house of 40 square metre carpet area exclude a household "
     "from Ujjwala?",
     "Yes, unless the house was received through a government scheme. Owning a house of more "
     "than 30 square metre carpet area other than through a government scheme is an "
     "exclusion criterion.", [(UJJ, 2)]),
    ("054", HG, MED, "Kya income tax bharne wala kisan PM-KISAN ke liye eligible hai?",
     "Nahi. Pichhle assessment year mein income tax bharne wale sabhi log excluded hain.",
     [(FAQ, 2)]),
    ("055", HG, HARD, "Kya tenant farmer, jo doosre ki zameen par kheti karta hai, PM-KISAN le "
     "sakta hai?",
     "Nahi. Zameen apne naam par honi chahiye; land holding hi sole criteria hai.",
     [(FAQR, 4)]),
]

TABLE_LOOKUP = [
    ("056", E, EASY, "Under PM-KMY, what is the monthly contribution for a subscriber entering "
     "at age 18?",
     "Rs. 55 per month.", [(KMY, 2)]),
    ("057", E, EASY, "Under PM-KMY, what is the monthly contribution at entry age 40?",
     "Rs. 200 per month.", [(KMY, 3)]),
    ("058", E, MED, "Under PM-KMY, what is the member's monthly contribution at entry age 29?",
     "Rs. 100 per month.", [(KMY, 2)]),
    ("059", E, MED, "Under PM-KMY, what is the member's monthly contribution at entry age 35?",
     "Rs. 150 per month.", [(KMY, 3)]),
    ("060", E, MED, "Under PM-KMY, what is the total monthly contribution at entry age 30, "
     "counting both member and Government?",
     "Rs. 210, made up of Rs. 105 from the member and Rs. 105 from the Government.",
     [(KMY, 2)]),
    ("061", E, HARD, "Under PM-KMY, what is the range of monthly contribution across all entry "
     "ages?",
     "Rs. 55 to Rs. 200 per month, depending on the age of entry.", [(KMY, 2)]),
    ("062", E, MED, "Under PM-KMY, what is the superannuation age shown for every entry age in "
     "the contribution table?",
     "60 years.", [(KMY, 2)]),
    ("063", E, HARD, "Under PM-KMY, what is the member's contribution at entry age 32?",
     "Rs. 120 per month.", [(KMY, 3)]),
    ("064", E, HARD, "Under PM-KMY, how much more per month does a subscriber entering at 40 "
     "pay than one entering at 18?",
     "Rs. 145 more per month: Rs. 200 against Rs. 55.", [(KMY, 2), (KMY, 3)]),
    ("065", E, MED, "In the Ujjwala deprivation checklist, which section covers asset and "
     "wealth related exclusions?",
     "Section B.", [(UJJ, 1)]),
    ("066", E, MED, "In the Ujjwala checklist, what irrigated land holding with one irrigation "
     "equipment triggers exclusion?",
     "More than 2.5 acres of irrigated land with one irrigation equipment.", [(UJJ, 1)]),
    ("067", E, HARD, "In the Ujjwala checklist, what land holding excludes a household when "
     "held for two or more crop seasons?",
     "5 acres or more of irrigated land for two or more crop seasons.", [(UJJ, 1)]),
    ("068", E, HARD, "In the Ujjwala checklist, what is the land threshold when a household "
     "has at least one irrigation equipment?",
     "At least 7.5 acres of land or more with at least one irrigation equipment.",
     [(UJJ, 1)]),
    ("069", E, MED, "Which Ujjwala checklist section contains the professional tax criterion?",
     "Section A, income and employment related.", [(UJJ, 1)]),
    ("070", E, MED, "Under PM-KMY, what is the Government's monthly contribution at entry "
     "age 25?",
     "Rs. 80 per month.", [(KMY, 2)]),
    ("071", HG, MED, "PM-KMY mein 20 saal ki umar par entry lene par kitna monthly contribution "
     "dena padta hai?",
     "Rs. 61 per month.", [(KMY, 2)]),
    ("072", E, HARD, "Under PM-KMY, at which entry age does the member's monthly contribution "
     "first reach Rs. 100?",
     "Entry age 29.", [(KMY, 2)]),
    ("073", E, HARD, "In the Ujjwala checklist, which two vehicle related criteria appear in "
     "the asset section?",
     "Owning a motorized 3 or 4 wheeler or fishing boat, and owning mechanized 3 or 4 wheeler "
     "agricultural equipment.", [(UJJ, 2)]),
]

# The type the agentic decomposition exists for: the criterion and the exclusion
# sit in different documents, so a single retrieval that finds one and stops
# produces a confident wrong answer rather than a miss.
MULTIHOP = [
    ("074", E, HARD, "A farmer owns 5 hectares and pays no income tax. Is he eligible for "
     "PM-KISAN today, and was he at launch?",
     "He is eligible now but was not at launch. The scheme initially covered only small and "
     "marginal farmers with up to 2 hectare, and was revised with effect from 1.6.2019 to "
     "cover all farmer families irrespective of landholding size.", [(FAQ, 1), (FAQR, 1)]),
    ("075", E, HARD, "A farmer owns 4 hectares. Is he eligible for PM-KISAN, and for PM-KMY?",
     "Eligible for PM-KISAN, since the landholding ceiling was removed with effect from "
     "1.6.2019, but not for PM-KMY, which still requires cultivable land up to 2 hectare.",
     [(FAQR, 1), (KMY, 3)]),
    ("076", E, HARD, "A retired Group D employee owns 1 hectare and draws a pension of "
     "Rs. 11,000. Is the family eligible for PM-KISAN?",
     "Yes. The pension exclusion at Rs. 10,000 or more and the government employee exclusion "
     "both carve out Multi Tasking Staff, Class IV and Group D employees.",
     [(FAQ, 2), (FAQ, 3)]),
    ("077", E, HARD, "A landowner died in March 2019 and his son inherited the land. Is the "
     "son eligible for PM-KISAN despite the cut-off date?",
     "Yes. Transfer of ownership on account of succession due to death of the landowner is "
     "the stated exception to the 01.02.2019 cut-off.", [(FAQ, 3), (FAQ, 4)]),
    ("078", E, HARD, "A farmer inherited land in 2019 but is a practising chartered accountant. "
     "Is he eligible for PM-KISAN?",
     "No. Succession preserves eligibility against the cut-off date, but the professional "
     "exclusion applies independently and disqualifies him.", [(FAQ, 2), (FAQ, 4)]),
    ("079", E, HARD, "Land was transferred by sale on 15.01.2019. Does the buyer get a full "
     "first installment?",
     "No. For transfers between 01.12.2018 and 31.01.2019 the first installment is a "
     "proportionate amount from the date of transfer to 31.03.2019, provided the family is "
     "otherwise eligible.", [(FAQ, 4), (FAQR, 3)]),
    ("080", E, HARD, "A family of husband, wife and two minor children holds 0.8 hectare each "
     "in four names. How much PM-KISAN benefit does the family receive?",
     "Rs. 6000 per annum in total. Land in the names of different family members is pooled, "
     "and the maximum benefit for the family remains Rs. 6000.", [(FAQ, 6)]),
    ("081", E, HARD, "Two unrelated farmer families are recorded on a single 3 hectare holding, "
     "each holding under 2 hectare. What does each family receive under PM-KISAN?",
     "Each family is eligible for benefit up to Rs. 6000, provided each is otherwise eligible "
     "under the scheme guidelines.", [(FAQ, 7)]),
    ("082", E, HARD, "A 42 year old small farmer wants an old age pension under PM-KMY. Can he "
     "join, and what else would he need to check?",
     "No. The entry age group is 18 to 40 years, so at 42 he cannot join regardless of the "
     "other criteria such as the 2 hectare ceiling and the exclusion list.",
     [(KMY, 1), (KMY, 2)]),
    ("083", E, HARD, "A farmer is enrolled in PM-KISAN and wants to join PM-KMY. Does PM-KISAN "
     "enrollment itself disqualify him?",
     "No. The PM-KMY exclusions cover other statutory social security schemes such as NPS, "
     "ESIC and EPFO, PM-SYM, and the higher economic status categories. PM-KISAN is not among "
     "them.", [(KMY, 1), (KMY, 2)]),
    ("084", E, HARD, "A household owns 6 acres of irrigated land used for two crop seasons and "
     "has no LPG connection. Does it qualify under Ujjwala?",
     "No. Owning 5 acres or more of irrigated land for two or more crop seasons is an "
     "exclusion criterion, and the absence of an LPG connection does not override it.",
     [(UJJ, 1), (UJJ, 2)]),
    ("085", E, HARD, "A farmer land record was never updated after his father died in 2017. "
     "Can he still become a PM-KISAN beneficiary?",
     "Yes. Where land records were not updated for rights accruing by succession before "
     "01.12.2018, States may update them in a time bound manner, and the successors are "
     "eligible subject to the other conditions and the exclusion clauses.", [(CLAR, 1)]),
    ("086", E, HARD, "After a landowner death, the family is found no longer eligible for "
     "PM-KISAN. What must happen?",
     "The PM KISAN portal has to be updated so that benefits are discontinued subsequently.",
     [(CLAR, 2)]),
    ("087", E, HARD, "A government employee family owns land in a village and the employee is "
     "Group D. Which of the two exclusions decides the outcome?",
     "Neither excludes them. The government employee exclusion explicitly excludes Multi "
     "Tasking Staff, Class IV and Group D employees from its scope, so the family remains "
     "eligible if not covered by other exclusion criteria.", [(FAQ, 3), (FAQR, 3)]),
    ("088", E, HARD, "A farmer owns 1 hectare of urban land under actual cultivation and paid "
     "income tax last year. Is he eligible for PM-KISAN?",
     "No. Urban cultivable land under actual cultivation is covered, but the income tax "
     "exclusion applies independently and disqualifies him.", [(FAQA, 1), (FAQ, 2)]),
    ("089", E, HARD, "Under PM-KMY, a subscriber dies at 55 having contributed regularly. What "
     "are the spouse options?",
     "The spouse may join and continue the scheme by paying regular contributions, or exit "
     "the scheme as per the provisions of exit and withdrawal.", [(KMY, 1)]),
    ("090", E, HARD, "A 30 year old farmer owning 1.5 hectare joins PM-KMY. What does he pay "
     "monthly, and what does he receive at 60?",
     "He pays Rs. 105 per month, matched by Rs. 105 from the Government, and receives a "
     "minimum assured pension of Rs. 3000 per month on attaining 60.", [(KMY, 1), (KMY, 2)]),
    ("091", E, HARD, "A farmer holds land in two different districts. Is he eligible for "
     "PM-KISAN twice?",
     "No. The entire land held by the family is pooled together and the maximum benefit is "
     "Rs. 6000 per annum.", [(FAQ, 6)]),
    ("092", E, HARD, "Do the PM-KISAN and PM-KMY landholding ceilings agree, and what changed?",
     "They no longer agree. The PM-KISAN 2 hectare ceiling was removed with effect from "
     "1.6.2019 when the scheme was extended to all farmer families, while PM-KMY still "
     "requires cultivable land up to 2 hectare.", [(FAQR, 1), (KMY, 1)]),
    ("093", HG, HARD, "Ek farmer ke paas 3 hectare zameen hai. Kya woh PM-KISAN aur PM-KMY "
     "dono le sakta hai?",
     "PM-KISAN haan, kyunki 1.6.2019 se landholding limit hata di gayi thi. PM-KMY nahi, "
     "kyunki usme 2 hectare tak ki limit abhi bhi lagu hai.", [(FAQR, 1), (KMY, 3)]),
    ("094", E, HARD, "A family sells part of its land in 2020 but retains 0.5 hectare. Does "
     "PM-KISAN eligibility survive?",
     "The transferor becomes ineligible only if the family has no cultivable land after the "
     "transfer. Retaining 0.5 hectare means that trigger does not apply, and officials "
     "reassess eligibility and decide whether benefits stop.", [(CLAR, 1), (CLAR, 2)]),
    ("095", E, HARD, "A farmer in Assam has no Aadhaar number. Can he receive PM-KISAN "
     "installments beyond the first?",
     "Assam was exempted from the mandatory Aadhaar requirement until 31st March 2020, with "
     "alternate prescribed documents collected for identity verification where Aadhaar was "
     "not available.", [(FAQR, 4), (FAQ, 8)]),
    ("096", E, HARD, "A household is excluded from Ujjwala on an income criterion and also owns "
     "a motorized four wheeler. How many exclusion criteria apply?",
     "Two: earning more than Rs. 10,000 per month in Section A, and owning a motorized 3 or 4 "
     "wheeler in Section B. Any one of them alone disqualifies the application.",
     [(UJJ, 1), (UJJ, 2)]),
    ("097", E, HARD, "Does giving an incorrect declaration carry the same consequence under "
     "PM-KISAN and PM-KMY?",
     "No. Under PM-KISAN the beneficiary is liable for recovery of the transferred benefit and "
     "other penal actions as per law. Under PM-KMY the beneficiary gets back their "
     "contributions without interest and the Government matching contribution stops.",
     [(FAQ, 3), (KMY, 3)]),
    ("098", E, HARD, "A minor child in a landholding family turned 18 in March 2019. Does the "
     "family composition change for PM-KISAN?",
     "The cut-off date for a minor child becoming major is 1.2.2019, so a child turning 18 in "
     "March 2019 was still a minor on the cut-off date and counts within the family as "
     "defined.", [(FAQA, 1), (FAQR, 4)]),
]

PROCEDURAL = [
    ("099", E, EASY, "What should a farmer do if their name is missing from the PM-KISAN "
     "beneficiary list?",
     "Approach the District Level Grievance Redressal Monitoring Committee in their district "
     "for inclusion of their name.", [(FAQ, 6)]),
    ("100", E, MED, "What identity and bank details must a farmer furnish to enroll in "
     "PM-KISAN?",
     "Name, age, gender and category, Aadhaar number, bank account number and IFSC code, with "
     "mobile number advised but not mandatory.", [(FAQR, 4), (FAQR, 5)]),
    ("101", E, MED, "Which three facilities does the Farmers Corner on the PM-KISAN portal "
     "provide?",
     "New Farmer Registration, Edit Aadhaar details, and Beneficiary Status.", [(FAQR, 5)]),
    ("102", E, MED, "How can a farmer check the status of their PM-KISAN installments?",
     "Through the Beneficiary Status link in the Farmers Corner, quoting their Aadhaar number, "
     "bank account number or registered mobile number.", [(FAQR, 5)]),
    ("103", E, HARD, "What happens after a farmer submits the New Farmer Registration form?",
     "The form is forwarded by an automated process to the State Nodal Officer, who verifies "
     "the details and uploads the verified data to the PM-KISAN portal, after which it is "
     "processed for payment.", [(FAQR, 5)]),
    ("104", E, MED, "Which alternate documents can be used for PM-KISAN identification where "
     "Aadhaar is unavailable?",
     "Aadhaar enrolment number, driving licence, voter ID card, NREGA job card, or other "
     "identification documents issued by Central, State or UT Governments or their "
     "authorities.", [(FAQR, 5)]),
    ("105", E, MED, "Is providing bank account details compulsory for PM-KISAN?",
     "Yes. Beneficiaries must provide bank account details with Aadhaar number, and no benefit "
     "can be given if bank account details have not been provided.", [(FAQ, 7)]),
    ("106", E, MED, "Can States submit PM-KISAN beneficiary lists in batches?",
     "Yes. States and UTs can provide lists of eligible beneficiaries in batches or phases as "
     "they are finalised, and benefits are released on a regular basis against the approved "
     "list.", [(FAQ, 8)]),
    ("107", E, MED, "Who is responsible for identifying eligible PM-KISAN beneficiaries?",
     "The State or UT Government, using the prevailing land ownership system and land records.",
     [(FAQR, 5)]),
    ("108", E, MED, "How is a PM-KISAN beneficiary notified that benefit has been sanctioned?",
     "Through a system generated SMS from the State or UT, in addition to the beneficiary "
     "lists displayed at Panchayats.", [(FAQ, 6)]),
    ("109", E, MED, "What information does a PM-KMY subscriber provide at registration?",
     "The farmer and spouse name, the farmer and spouse date of birth, and the bank account "
     "number.", [(KMY, 4)]),
    ("110", E, HARD, "Who should a PM-KMY grievance or dispute be referred to?",
     "The Joint Secretary (Farmers Welfare), Department of Agriculture, Cooperation and "
     "Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Krishi Bhavan, New Delhi "
     "110001.", [(KMY, 10)]),
    ("111", E, MED, "What must an Ujjwala applicant declare before receiving a connection?",
     "That the applicant and their family members, as per family composition, do not possess "
     "or meet any of the listed exclusion criteria.", [(UJJ, 1)]),
    ("112", E, MED, "What can an Oil Marketing Company do if an Ujjwala declaration is found "
     "false?",
     "Withdraw the gas supply, terminate the connection, seize the equipment, recover the "
     "subsidy transferred to the bank account, and take legal action against fraudulent "
     "claims.", [(UJJ, 2)]),
    ("113", E, MED, "What does the Ujjwala distributor certify on the deprivation declaration?",
     "That they received and verified photocopies of the documents submitted by the applicant "
     "against their originals, confirmed by signature and seal.", [(UJJ, 2)]),
    ("114", E, HARD, "The Ujjwala declaration is signed by an applicant who cannot read it. "
     "What does the form require?",
     "The applicant declares that the contents were read out and explained by the distributor "
     "or their authorised person, and that the applicant understood them.", [(UJJ, 2)]),
    ("115", E, MED, "Which proof of address code does the Ujjwala KYC form give to a bank "
     "passbook or statement?",
     "POA03.", [("ujjwala-kyc-en", 1)]),
    ("116", E, HARD, "What must States put in place for PM-KISAN land record mutation, and by "
     "when?",
     "An administrative mechanism based on their existing land revenue administrative regime, "
     "with clear responsibility entrusted to officials for mutation of land records, to be in "
     "place before 31st March and intimated to the Central Government.",
     [(CLAR, 1), (CLAR, 2)]),
    ("117", E, HARD, "What safeguard applies to editing the PM-KISAN database?",
     "Editing is allowed through valid user names and passwords, and such transactions and "
     "edits must be archived with transaction history maintained for data security and future "
     "scrutiny or enquiry.", [(CLAR, 2)]),
    ("118", E, MED, "How does a farmer correct their name on the PM-KISAN portal?",
     "Through the Edit Aadhaar details link in the Farmers Corner, where the farmer edits the "
     "name as per the Aadhaar card and it is updated after authentication.", [(FAQR, 5)]),
    ("119", HG, MED, "Agar PM-KISAN ki list mein naam nahi hai to kya karna chahiye?",
     "Apne district ki District Level Grievance Redressal Monitoring Committee se sampark "
     "karna chahiye taaki naam list mein shamil ho sake.", [(FAQ, 6)]),
    ("120", HG, MED, "PM-KISAN portal par apna payment status kaise dekhein?",
     "Farmers Corner ke Beneficiary Status link se, Aadhaar number, bank account number ya "
     "registered mobile number daal kar.", [(FAQR, 5)]),
    ("121", E, MED, "Under PM-KISAN, how is the benefit delivered to the beneficiary?",
     "The income support benefit is credited directly into the bank accounts of "
     "beneficiaries.", [(FAQ, 7)]),
    ("122", E, HARD, "What must happen when inheritance makes a new family freshly eligible "
     "for PM-KISAN?",
     "All details of the freshly eligible families have to be incorporated in the database so "
     "that benefits accrue from the date the inheritance became operational.", [(CLAR, 2)]),
    ("123", E, HARD, "When a deceased PM-KISAN landowner family remains eligible, what has to "
     "be submitted?",
     "The details of the new beneficiary along with other details and fresh self-declarations "
     "must be provided to the concerned authorities for inclusion or modification, so benefit "
     "to the survivors continues.", [(CLAR, 2)]),
]

# SPEC: refusal correctness cannot be measured without these, and a system that
# never refuses scores perfectly on a set that contains none. Each note records
# what was searched for and not found, so the absence is evidence rather than an
# assumption.
UNANSWERABLE = [
    ("124", E, MED, "What is the PM-KISAN installment amount for the financial year 2027-28?",
     "The corpus does not state this.",
     "Corpus documents describe Rs. 2000 per installment as at their publication dates in "
     "2019 and 2020. No document in the corpus refers to 2027-28."),
    ("125", E, MED, "How many PM-KISAN beneficiaries are there in Bihar?",
     "The corpus does not state this.",
     "The corpus holds guidelines, FAQs and clarifications. It contains no state-wise "
     "beneficiary counts for any state."),
    ("126", E, MED, "What is the toll free helpline number for PM-KISAN?",
     "The corpus does not state this.",
     "The FAQs point to the PM-KISAN web portal and the District Level Grievance Redressal "
     "Monitoring Committee, and give no helpline number."),
    ("127", E, HARD, "What penalty in rupees is levied on an official who wrongly certifies a "
     "PM-KISAN beneficiary?",
     "The corpus does not state this.",
     "The clarification assigns responsibility to officials for mutation of land records but "
     "specifies no monetary penalty for them. Penal action is described only for "
     "beneficiaries giving incorrect declarations."),
    ("128", E, MED, "What is the annual budget allocation for PM-KMY?",
     "The corpus does not state this.",
     "PM-KMY documents in the corpus cover eligibility, contributions, benefits and exit. No "
     "budget figure appears."),
    ("129", E, MED, "Which bank administers the PM-KMY pension fund?",
     "The corpus does not state this.",
     "The PM-KMY FAQ names the grievance authority and describes contributions, but does not "
     "name a fund administering bank in the pages held here."),
    ("130", E, HARD, "What is the PM-KMY contribution for someone entering at age 45?",
     "The corpus does not state this, and the question does not arise.",
     "The contribution table runs from entry age 18 to 40 only, because the entry age group "
     "is 18 to 40. There is no row for 45."),
    ("131", E, MED, "How many LPG connections were released under Ujjwala in 2025-26?",
     "The corpus does not state this.",
     "The Ujjwala documents held are a deprivation declaration, a KYC form and an insurance "
     "document. None reports connection counts."),
    ("132", E, MED, "What is the current price of an LPG cylinder under Ujjwala?",
     "The corpus does not state this.",
     "No pricing information appears in the Ujjwala documents in this corpus."),
    ("133", E, HARD, "What is the subsidy amount transferred to an Ujjwala beneficiary bank "
     "account?",
     "The corpus does not state this.",
     "The declaration refers to recovery of the subsidy amount transferred if a declaration is "
     "false, without stating what that amount is."),
    ("134", E, MED, "Which minister launched the Ujjwala scheme, and on what date?",
     "The corpus does not state this.",
     "The Ujjwala documents held are forms and an insurance document, and carry no launch "
     "history."),
    ("135", E, MED, "How long does PM-KISAN registration take to be approved?",
     "The corpus does not state this.",
     "The FAQ describes the registration and verification flow through the State Nodal "
     "Officer but gives no processing time."),
    ("136", E, HARD, "Can a PM-KISAN beneficiary appeal a rejection, and to whom?",
     "The corpus does not state an appeal route against rejection.",
     "The FAQs give a route for names missing from the beneficiary list, through the District "
     "Level Grievance Redressal Monitoring Committee, but describe no appeal against an "
     "explicit rejection."),
    ("137", E, MED, "What interest rate applies to PM-KMY contributions before age 60?",
     "The corpus does not state this.",
     "The FAQ says contributions are returned without interest on an incorrect declaration, "
     "which implies no interest in that case, but states no interest rate for the fund."),
    ("138", E, HARD, "Is a PM-KISAN beneficiary automatically enrolled in PM-KMY?",
     "The corpus does not state this.",
     "Both schemes are described separately in this corpus and neither document held here "
     "describes automatic enrollment from one into the other."),
    ("139", E, MED, "What is the PMAY-U 2.0 income ceiling for the Middle Income Group?",
     "The corpus does not state this in the pages held.",
     "The PMAY-U documents in this corpus name EWS, LIG and MIG categories but the specific "
     "income ceilings for MIG do not appear in the extracted pages."),
    ("140", E, MED, "How many houses were sanctioned under PMAY-U 2.0 specifically?",
     "The corpus does not state this.",
     "The Angikaar document reports more than 1.22 crore houses sanctioned under PMAY-U since "
     "2015, not a figure specific to PMAY-U 2.0."),
    ("141", E, HARD, "What happens to a PM-KISAN beneficiary who migrates permanently abroad?",
     "The corpus does not state this.",
     "Eligibility is described in terms of landholding, family composition and the exclusion "
     "list. Residence or migration abroad is not addressed."),
    ("142", E, MED, "Can a PM-KISAN beneficiary nominate someone to receive the benefit?",
     "The corpus does not state this.",
     "Payment is described as going to the family member with the highest landholding, with "
     "no nomination facility described."),
    ("143", E, HARD, "Does PM-KISAN benefit continue if the beneficiary land is acquired by the "
     "government?",
     "The corpus does not state this.",
     "Transfer of ownership is addressed for sale, gift, partition and succession. Compulsory "
     "acquisition by the government is not among the cases described."),
    ("144", E, MED, "What is the grievance redressal timeline for PM-KISAN complaints?",
     "The corpus does not state this.",
     "The committee is named as the route for grievances, with no timeline given for "
     "resolution."),
    ("145", HG, MED, "PM-KISAN ka helpline number kya hai?",
     "Corpus mein yeh nahi diya gaya hai.",
     "FAQs portal aur District Level Grievance Redressal Monitoring Committee ka zikr karte "
     "hain, lekin koi helpline number nahi dete."),
    ("146", HG, HARD, "PM-KMY mein 45 saal ki umar par kitna contribution dena hoga?",
     "Yeh sawaal uthta hi nahi, aur corpus mein iska jawab nahi hai.",
     "Contribution table sirf 18 se 40 tak hai, kyunki entry age group hi 18 se 40 hai. 45 ke "
     "liye koi row nahi hai."),
    ("147", E, MED, "How many farmers were removed from PM-KISAN after eligibility "
     "reassessment?",
     "The corpus does not state this.",
     "The clarification describes the reassessment mechanism and portal updates but reports "
     "no counts of removals."),
    ("148", E, HARD, "What is the penalty for an Oil Marketing Company that wrongly approves an "
     "Ujjwala connection?",
     "The corpus does not state this.",
     "The declaration describes consequences for the applicant and certification duties for "
     "the distributor, but no penalty on the company."),
    ("149", E, MED, "Which states have the highest PM-KMY enrollment?",
     "The corpus does not state this.",
     "The PM-KMY documents describe the scheme design and give no enrollment figures by "
     "state."),
    ("150", E, HARD, "Can a farmer hold both PM-KMY and a State government pension scheme?",
     "The corpus does not state this.",
     "The exclusions name NPS, ESIC, EPFO and PM-SYM specifically. State government pension "
     "schemes are not addressed either way."),
]


def build() -> list[Question]:
    """Assemble the set.

    Ids are stable and never reused, so a question that is rewritten during
    adjudication keeps its identity and a question that is replaced gets a new
    one. The draft annotator is recorded on every row, which is what makes the
    M4.3 audit a comparison between two named annotators rather than a review of
    anonymous text.
    """
    out: list[Question] = []
    for group, kind in (
        (FACTUAL, "factual"),
        (EXCLUSION, "exclusion"),
        (TABLE_LOOKUP, "table_lookup"),
        (MULTIHOP, "eligibility_multihop"),
        (PROCEDURAL, "procedural"),
    ):
        for suffix, lang, difficulty, question, answer, pages in group:
            qid = f"dz-{suffix}"
            lang, question, answer = TRANSLATED.get(qid, (lang, question, answer))
            out.append(
                Question(
                    id=qid,
                    question=question,
                    gold_answer=answer,
                    gold_pages=tuple(pages),
                    difficulty_tag=difficulty,
                    language_tag=lang,
                    question_type=kind,
                    annotator_ids=(DRAFT_ANNOTATOR,),
                )
            )
    for suffix, lang, difficulty, question, answer, note in UNANSWERABLE:
        qid = f"dz-{suffix}"
        lang, question, answer = TRANSLATED.get(qid, (lang, question, answer))
        out.append(
            Question(
                id=qid,
                question=question,
                gold_answer=answer,
                gold_pages=(),
                difficulty_tag=difficulty,
                language_tag=lang,
                question_type="unanswerable",
                annotator_ids=(DRAFT_ANNOTATOR,),
                note=note,
            )
        )
    return out


def corpus_pages() -> set[tuple[str, int]]:
    """Every page the corpus can actually cite, so a citation cannot be invented."""
    con = sqlite3.connect(CHUNKS)
    try:
        return {
            (doc, page)
            for doc, page in con.execute(
                "select distinct document_id, page from chunks where run_id=?", (RUN_ID,)
            )
        }
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the eval set.")
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()

    questions = build()
    problems = structural_errors(questions)
    problems += coverage_errors(questions, corpus_pages())
    if problems:
        for problem in problems:
            print(f"  {problem}")
        print(f"\n{len(problems)} problem(s); nothing written.")
        return 1

    digest = content_hash(questions)
    counts: dict[str, int] = {}
    langs: dict[str, int] = {}
    for q in questions:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
        langs[q.language_tag] = langs.get(q.language_tag, 0) + 1

    print(f"{len(questions)} questions, hash {digest[:16]}")
    print("  by type:     " + json.dumps(counts, sort_keys=True))
    print("  by language: " + json.dumps(langs, sort_keys=True))

    if args.check:
        return 0

    dump(questions)
    Path("eval/questions.sha256").write_text(digest + "\n", encoding="utf-8", newline="\n")
    print("\nwrote eval/questions.jsonl and eval/questions.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
