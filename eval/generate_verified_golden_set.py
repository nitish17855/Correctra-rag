import os
import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunker.recursive_character_chunker import recursive_splitters

cache_file = PROJECT_ROOT / "data" / "aapl_clean_text.txt"
with open(cache_file, "r", encoding="utf-8") as f:
    clean_text = f.read()

docs = recursive_splitters(clean_text, chunk_size=1000, chunk_overlap=100)
for idx, d in enumerate(docs):
    d.metadata["chunk_id"] = f"chunk_{idx:03d}"

print(f"Loaded {len(docs)} chunks.")

def find_cids(terms: list[str]) -> list[str]:
    cids = []
    for d in docs:
        txt = re.sub(r'\s+', ' ', d.page_content.lower())
        if all(re.sub(r'\s+', ' ', t.lower()) in txt for t in terms):
            cids.append(d.metadata["chunk_id"])
    return cids

# 75 Verified Golden Q&As strictly aligned with Apple Inc. Form 10-K (FY2025)
entries = [
    # ==========================================
    # 1. FACTUAL QUESTIONS (20 items)
    # ==========================================
    {
        "id": "gold_001",
        "category": "factual",
        "query": "What is the exact fiscal year end date for Apple's 2025 Form 10-K filing?",
        "answer": "September 27, 2025",
        "terms": ["September 27", "2025"],
        "difficulty": "easy",
        "source": "Cover Page"
    },
    {
        "id": "gold_002",
        "category": "factual",
        "query": "Where are Apple's principal executive offices located?",
        "answer": "One Apple Park Way, Cupertino, California 95014",
        "terms": ["One Apple Park Way", "Cupertino"],
        "difficulty": "easy",
        "source": "Cover Page"
    },
    {
        "id": "gold_003",
        "category": "factual",
        "query": "Under which ticker symbol is Apple's common stock registered on Nasdaq?",
        "answer": "AAPL",
        "terms": ["AAPL", "Nasdaq"],
        "difficulty": "easy",
        "source": "Cover Page"
    },
    {
        "id": "gold_004",
        "category": "factual",
        "query": "What is Apple's SEC Commission File Number?",
        "answer": "001-36743",
        "terms": ["001-36743"],
        "difficulty": "easy",
        "source": "Cover Page"
    },
    {
        "id": "gold_005",
        "category": "factual",
        "query": "What state is Apple Inc. incorporated in?",
        "answer": "California",
        "terms": ["California", "incorporation"],
        "difficulty": "easy",
        "source": "Cover Page"
    },
    {
        "id": "gold_006",
        "category": "factual",
        "query": "What is Apple's IRS Employer Identification Number?",
        "answer": "94-2404110",
        "terms": ["94-2404110"],
        "difficulty": "easy",
        "source": "Cover Page"
    },
    {
        "id": "gold_007",
        "category": "factual",
        "query": "Which independent registered public accounting firm audited Apple's financial statements?",
        "answer": "Ernst & Young LLP",
        "terms": ["Ernst & Young LLP"],
        "difficulty": "easy",
        "source": "Item 8 Auditor Report"
    },
    {
        "id": "gold_008",
        "category": "factual",
        "query": "What operating system powers Apple's Mac personal computers?",
        "answer": "macOS",
        "terms": ["macOS", "Mac"],
        "difficulty": "easy",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_009",
        "category": "factual",
        "query": "What operating system powers Apple's iPad tablets?",
        "answer": "iPadOS",
        "terms": ["iPadOS", "iPad"],
        "difficulty": "easy",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_010",
        "category": "factual",
        "query": "What operating system powers the Apple Watch?",
        "answer": "watchOS",
        "terms": ["watchOS", "Apple Watch"],
        "difficulty": "easy",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_011",
        "category": "factual",
        "query": "What operating system powers Apple Vision Pro?",
        "answer": "visionOS",
        "terms": ["visionOS", "Apple Vision Pro"],
        "difficulty": "easy",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_012",
        "category": "factual",
        "query": "What operating system powers Apple TV 4K?",
        "answer": "tvOS",
        "terms": ["tvOS", "Apple TV"],
        "difficulty": "easy",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_013",
        "category": "factual",
        "query": "What is the name of Apple's digital application marketplace for iOS and macOS?",
        "answer": "App Store",
        "terms": ["App Store", "digital content"],
        "difficulty": "easy",
        "source": "Item 1 Services"
    },
    {
        "id": "gold_014",
        "category": "factual",
        "query": "What is Apple's cashless payment service that enables mobile transactions?",
        "answer": "Apple Pay",
        "terms": ["Apple Pay", "cashless payment"],
        "difficulty": "easy",
        "source": "Item 1 Services"
    },
    {
        "id": "gold_015",
        "category": "factual",
        "query": "What is Apple's music subscription streaming service called?",
        "answer": "Apple Music",
        "terms": ["Apple Music"],
        "difficulty": "easy",
        "source": "Item 1 Services"
    },
    {
        "id": "gold_016",
        "category": "factual",
        "query": "What is Apple's curated gaming subscription service called?",
        "answer": "Apple Arcade",
        "terms": ["Apple Arcade"],
        "difficulty": "easy",
        "source": "Item 1 Services"
    },
    {
        "id": "gold_017",
        "category": "factual",
        "query": "What is Apple's subscription news and magazine aggregation service called?",
        "answer": "Apple News+",
        "terms": ["Apple News+"],
        "difficulty": "easy",
        "source": "Item 1 Services"
    },
    {
        "id": "gold_018",
        "category": "factual",
        "query": "Who is Apple's Chief Executive Officer (CEO) signing the 10-K certifications?",
        "answer": "Timothy D. Cook",
        "terms": ["Timothy D. Cook", "Chief Executive Officer"],
        "difficulty": "easy",
        "source": "Signatures & Certifications"
    },
    {
        "id": "gold_019",
        "category": "factual",
        "query": "Who is Apple's Chief Financial Officer signing the 10-K certifications?",
        "answer": "Kevan Parekh",
        "terms": ["Kevan Parekh", "Chief Financial Officer"],
        "difficulty": "easy",
        "source": "Signatures & Certifications"
    },
    {
        "id": "gold_020",
        "category": "factual",
        "query": "What European regulation designated under the acronym DMA imposes gatekeeper rules on Apple?",
        "answer": "Digital Markets Act (DMA)",
        "terms": ["Digital Markets Act", "DMA"],
        "difficulty": "medium",
        "source": "Item 1A Risk Factors"
    },

    # ==========================================
    # 2. NUMERICAL QUESTIONS (25 items)
    # ==========================================
    {
        "id": "gold_021",
        "category": "numerical",
        "query": "What was Apple's total net sales in fiscal year 2025?",
        "answer": "$416,161 million",
        "terms": ["416,161", "total net sales"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_022",
        "category": "numerical",
        "query": "What was Apple's total net sales in fiscal year 2024?",
        "answer": "$391,035 million",
        "terms": ["391,035", "total net sales"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_023",
        "category": "numerical",
        "query": "What was Apple's total net sales in fiscal year 2023?",
        "answer": "$383,285 million",
        "terms": ["383,285", "total net sales"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_024",
        "category": "numerical",
        "query": "What were iPhone net sales in fiscal year 2025?",
        "answer": "$209,586 million",
        "terms": ["209,586", "iPhone"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_025",
        "category": "numerical",
        "query": "What were iPhone net sales in fiscal year 2024?",
        "answer": "$201,183 million",
        "terms": ["201,183", "iPhone"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_026",
        "category": "numerical",
        "query": "What were iPhone net sales in fiscal year 2023?",
        "answer": "$200,583 million",
        "terms": ["200,583", "iPhone"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_027",
        "category": "numerical",
        "query": "What were Apple's Services net sales in fiscal year 2025?",
        "answer": "$109,158 million",
        "terms": ["109,158", "Services"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_028",
        "category": "numerical",
        "query": "What were Apple's Services net sales in fiscal year 2024?",
        "answer": "$96,169 million",
        "terms": ["96,169", "Services"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_029",
        "category": "numerical",
        "query": "What were Mac net sales in fiscal year 2025?",
        "answer": "$33,708 million",
        "terms": ["33,708", "Mac"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_030",
        "category": "numerical",
        "query": "What were Mac net sales in fiscal year 2024?",
        "answer": "$29,984 million",
        "terms": ["29,984", "Mac"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_031",
        "category": "numerical",
        "query": "What were iPad net sales in fiscal year 2025?",
        "answer": "$28,023 million",
        "terms": ["28,023", "iPad"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_032",
        "category": "numerical",
        "query": "What were iPad net sales in fiscal year 2024?",
        "answer": "$26,694 million",
        "terms": ["26,694", "iPad"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_033",
        "category": "numerical",
        "query": "What were Wearables, Home and Accessories net sales in fiscal year 2025?",
        "answer": "$35,686 million",
        "terms": ["35,686", "Wearables"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_034",
        "category": "numerical",
        "query": "What were Wearables, Home and Accessories net sales in fiscal year 2024?",
        "answer": "$37,005 million",
        "terms": ["37,005", "Wearables"],
        "difficulty": "easy",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_035",
        "category": "numerical",
        "query": "What was Apple's total gross margin in fiscal year 2025?",
        "answer": "$195,201 million",
        "terms": ["195,201", "gross margin"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_036",
        "category": "numerical",
        "query": "What was Apple's total gross margin in fiscal year 2024?",
        "answer": "$180,683 million",
        "terms": ["180,683", "gross margin"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_037",
        "category": "numerical",
        "query": "What was Apple's research and development (R&D) expense in fiscal year 2025?",
        "answer": "$34,550 million",
        "terms": ["34,550", "research and development"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_038",
        "category": "numerical",
        "query": "What was Apple's research and development (R&D) expense in fiscal year 2024?",
        "answer": "$31,370 million",
        "terms": ["31,370", "research and development"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_039",
        "category": "numerical",
        "query": "What was Apple's selling, general and administrative (SG&A) expense in fiscal year 2025?",
        "answer": "$27,601 million",
        "terms": ["27,601", "selling, general and administrative"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_040",
        "category": "numerical",
        "query": "What was Apple's total operating expenses in fiscal year 2025?",
        "answer": "$62,151 million",
        "terms": ["62,151", "total operating expenses"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_041",
        "category": "numerical",
        "query": "What was Apple's operating income in fiscal year 2025?",
        "answer": "$133,050 million",
        "terms": ["133,050", "operating income"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_042",
        "category": "numerical",
        "query": "What was Apple's consolidated net income in fiscal year 2025?",
        "answer": "$112,010 million",
        "terms": ["112,010", "net income"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_043",
        "category": "numerical",
        "query": "What was Apple's basic earnings per share (EPS) in fiscal year 2025?",
        "answer": "$7.49",
        "terms": ["7.49", "basic"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_044",
        "category": "numerical",
        "query": "What was Apple's diluted earnings per share (EPS) in fiscal year 2025?",
        "answer": "$7.46",
        "terms": ["7.46", "diluted"],
        "difficulty": "easy",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_045",
        "category": "numerical",
        "query": "What was Apple's cash and cash equivalents balance as of September 27, 2025?",
        "answer": "$35,934 million",
        "terms": ["35,934", "cash and cash equivalents"],
        "difficulty": "easy",
        "source": "Consolidated Balance Sheets"
    },

    # ==========================================
    # 3. MULTI-HOP QUESTIONS (15 items)
    # ==========================================
    {
        "id": "gold_046",
        "category": "multi-hop",
        "query": "How did total net sales change between fiscal year 2024 and 2025 in dollar terms and percentage?",
        "answer": "Total net sales increased from $391,035 million in 2024 to $416,161 million in 2025, an increase of $25,126 million or 6%.",
        "terms": ["416,161", "391,035"],
        "difficulty": "hard",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_047",
        "category": "multi-hop",
        "query": "How did iPhone net sales compare between fiscal year 2024 and 2025?",
        "answer": "iPhone net sales increased from $201,183 million in 2024 to $209,586 million in 2025, an increase of $8,403 million (or 4%).",
        "terms": ["209,586", "201,183"],
        "difficulty": "hard",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_048",
        "category": "multi-hop",
        "query": "How did Services net sales grow compared to Mac net sales growth between 2024 and 2025?",
        "answer": "Services net sales grew by 14% (from $96,169M to $109,158M), while Mac net sales grew by 12% (from $29,984M to $33,708M).",
        "terms": ["109,158", "33,708"],
        "difficulty": "hard",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_049",
        "category": "multi-hop",
        "query": "How did R&D expense change between fiscal 2024 and 2025?",
        "answer": "R&D expense increased from $31,370 million in 2024 to $34,550 million in 2025, an increase of $3,180 million.",
        "terms": ["34,550", "31,370"],
        "difficulty": "hard",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_050",
        "category": "multi-hop",
        "query": "Rank Apple's five product and services categories by revenue in fiscal 2025 from largest to smallest.",
        "answer": "1. iPhone ($209,586M), 2. Services ($109,158M), 3. Wearables, Home and Accessories ($35,686M), 4. Mac ($33,708M), 5. iPad ($28,023M).",
        "terms": ["209,586", "109,158", "35,686"],
        "difficulty": "hard",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_051",
        "category": "multi-hop",
        "query": "How did operating income change between fiscal year 2024 and 2025?",
        "answer": "Operating income increased from $123,216 million in 2024 to $133,050 million in 2025, an increase of $9,834 million.",
        "terms": ["133,050", "123,216"],
        "difficulty": "hard",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_052",
        "category": "multi-hop",
        "query": "How did net income and diluted EPS change between fiscal 2024 and 2025?",
        "answer": "Net income grew from $93,736 million to $112,010 million, driving diluted EPS from $6.08 to $7.46.",
        "terms": ["112,010", "7.46"],
        "difficulty": "hard",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_053",
        "category": "multi-hop",
        "query": "How did total gross margin change between fiscal year 2024 and 2025?",
        "answer": "Gross margin increased from $180,683 million in 2024 to $195,201 million in 2025, an increase of $14,518 million.",
        "terms": ["195,201", "180,683"],
        "difficulty": "hard",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_054",
        "category": "multi-hop",
        "query": "How did total operating expenses compare between fiscal 2024 and 2025?",
        "answer": "Total operating expenses rose from $57,467 million in 2024 to $62,151 million in 2025, an increase of $4,684 million.",
        "terms": ["62,151", "57,467"],
        "difficulty": "hard",
        "source": "Consolidated Statements of Operations"
    },
    {
        "id": "gold_055",
        "category": "multi-hop",
        "query": "How did Products gross margin compare with Services gross margin in fiscal 2025 in dollar terms?",
        "answer": "In 2025, Products gross margin was $112,887 million, while Services gross margin was $82,314 million.",
        "terms": ["112,887", "82,314"],
        "difficulty": "hard",
        "source": "Item 7 MD&A Gross Margin"
    },
    {
        "id": "gold_056",
        "category": "multi-hop",
        "query": "Which product category experienced a year-over-year sales decline between fiscal 2024 and 2025?",
        "answer": "Wearables, Home and Accessories was the only category to decline, falling from $37,005 million to $35,686 million (-4%).",
        "terms": ["35,686", "37,005"],
        "difficulty": "medium",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_057",
        "category": "multi-hop",
        "query": "How did Mac sales compare to iPad sales in fiscal year 2025?",
        "answer": "Mac sales reached $33,708 million, exceeding iPad sales of $28,023 million by $5,685 million.",
        "terms": ["33,708", "28,023"],
        "difficulty": "medium",
        "source": "Segment Net Sales"
    },
    {
        "id": "gold_058",
        "category": "multi-hop",
        "query": "What was the year-over-year change in commercial paper balance between 2024 and 2025?",
        "answer": "Commercial paper decreased from $9,967 million as of September 28, 2024 to $7,979 million as of September 27, 2025.",
        "terms": ["7,979", "9,967"],
        "difficulty": "medium",
        "source": "Consolidated Balance Sheets"
    },
    {
        "id": "gold_059",
        "category": "multi-hop",
        "query": "How did cash and cash equivalents change between September 28, 2024 and September 27, 2025?",
        "answer": "Cash and cash equivalents increased from $29,943 million in 2024 to $35,934 million in 2025.",
        "terms": ["35,934", "29,943"],
        "difficulty": "medium",
        "source": "Consolidated Balance Sheets"
    },
    {
        "id": "gold_060",
        "category": "multi-hop",
        "query": "How did total current assets change between fiscal 2024 and 2025?",
        "answer": "Total current assets decreased from $152,987 million in 2024 to $147,957 million in 2025.",
        "terms": ["147,957", "152,987"],
        "difficulty": "medium",
        "source": "Consolidated Balance Sheets"
    },

    # ==========================================
    # 4. SUMMARIZATION QUESTIONS (15 items)
    # ==========================================
    {
        "id": "gold_061",
        "category": "summarization",
        "query": "Summarize the principal risks Apple faces regarding outsourced manufacturing in China.",
        "answer": "Apple relies substantially on outsourced manufacturing and assembly partners in China. Any political tensions, trade restrictions, or operational disruptions could severely constrain production and shipments.",
        "terms": ["outsourced", "China"],
        "difficulty": "medium",
        "source": "Item 1A Risk Factors"
    },
    {
        "id": "gold_062",
        "category": "summarization",
        "query": "What regulatory exposure does Apple disclose regarding the European Union Digital Markets Act (DMA)?",
        "answer": "The European Commission designated Apple under the DMA, requiring Apple to support alternative app distribution, alternative fee structures, and alternative payment processing, creating compliance risks and potential substantial fines.",
        "terms": ["Digital Markets Act", "DMA"],
        "difficulty": "hard",
        "source": "Item 1A Risk Factors"
    },
    {
        "id": "gold_063",
        "category": "summarization",
        "query": "Summarize Apple's accounting policies regarding management estimates and GAAP conformity.",
        "answer": "Apple prepares its consolidated financial statements in conformity with GAAP, requiring the use of management estimates and assumptions that affect reported amounts.",
        "terms": ["Summary of Significant Accounting Policies", "GAAP"],
        "difficulty": "hard",
        "source": "Note 1 Summary of Accounting Policies"
    },
    {
        "id": "gold_064",
        "category": "summarization",
        "query": "Describe Apple's capital return program strategy regarding dividends and share repurchases.",
        "answer": "Apple returns capital to shareholders through regular quarterly dividends and open-market common stock repurchases under authorized board programs.",
        "terms": ["share repurchase program", "dividends"],
        "difficulty": "medium",
        "source": "Liquidity & Capital Resources"
    },
    {
        "id": "gold_065",
        "category": "summarization",
        "query": "What risks does Apple disclose regarding single-source component suppliers and custom silicon foundries?",
        "answer": "Apple obtains several critical components, including custom silicon and displays, from single or limited-source foundries, leaving Apple vulnerable to supplier shortages and capacity constraints.",
        "terms": ["single-source", "components"],
        "difficulty": "medium",
        "source": "Item 1A Risk Factors"
    },
    {
        "id": "gold_066",
        "category": "summarization",
        "query": "Summarize Apple's commercial paper and short-term debt position.",
        "answer": "Apple maintains active commercial paper and term debt programs for working capital, capital return, and general corporate purposes.",
        "terms": ["commercial paper", "term debt"],
        "difficulty": "medium",
        "source": "Consolidated Balance Sheets"
    },
    {
        "id": "gold_067",
        "category": "summarization",
        "query": "What antitrust allegations has Apple faced from civil lawsuits in the U.S.?",
        "answer": "Apple is subject to civil antitrust lawsuits alleging monopolization or attempted monopolization in smartphone markets regarding App Store rules and developer restrictions.",
        "terms": ["antitrust", "monopolization"],
        "difficulty": "hard",
        "source": "Item 1A Risk Factors"
    },
    {
        "id": "gold_068",
        "category": "summarization",
        "query": "Summarize how Apple uses derivative instruments to manage risk.",
        "answer": "Apple enters into derivative financial instruments recorded at fair value to mitigate risks from foreign currency and interest rate fluctuations.",
        "terms": ["derivative instruments", "fair value"],
        "difficulty": "hard",
        "source": "Note on Financial Instruments"
    },
    {
        "id": "gold_069",
        "category": "summarization",
        "query": "Summarize Apple's primary geographic reportable segments.",
        "answer": "Apple manages its business primarily on a geographic basis, reporting across Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.",
        "terms": ["reportable segments", "Americas", "Europe"],
        "difficulty": "medium",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_070",
        "category": "summarization",
        "query": "Summarize Apple's properties and leased corporate facilities.",
        "answer": "Apple owns and leases facilities for corporate functions, R&D, data centers, and retail stores across the U.S. and internationally.",
        "terms": ["leased facilities", "retail"],
        "difficulty": "medium",
        "source": "Item 2 Properties"
    },
    {
        "id": "gold_071",
        "category": "summarization",
        "query": "What cybersecurity and operational resilience risks does Apple emphasize?",
        "answer": "Apple is targeted by sophisticated cyberattacks and operational threats that could compromise customer data or disrupt network services.",
        "terms": ["cybersecurity", "Item 1A"],
        "difficulty": "medium",
        "source": "Item 1A Risk Factors"
    },
    {
        "id": "gold_072",
        "category": "summarization",
        "query": "Describe Apple's disclosures regarding critical accounting estimates.",
        "answer": "Critical accounting estimates include complex management judgments regarding revenue recognition, valuation of inventory, warranties, and income taxes.",
        "terms": ["critical accounting estimates", "accounting policies"],
        "difficulty": "hard",
        "source": "Item 7 Critical Accounting Estimates"
    },
    {
        "id": "gold_073",
        "category": "summarization",
        "query": "What role does Apple's active installed base play in driving Services growth?",
        "answer": "Apple's large and growing active installed base of devices drives ongoing user engagement and recurring revenue across Services offerings.",
        "terms": ["installed base", "Services"],
        "difficulty": "medium",
        "source": "Item 1 Business Overview"
    },
    {
        "id": "gold_074",
        "category": "summarization",
        "query": "What risks does Apple disclose regarding tariffs and international trade disputes?",
        "answer": "Tariffs announced on imports to the U.S. from China and other nations could increase product costs, strain supply chains, and trigger reciprocal international trade barriers.",
        "terms": ["tariffs", "U.S."],
        "difficulty": "medium",
        "source": "Item 1A Risk Factors"
    },
    {
        "id": "gold_075",
        "category": "summarization",
        "query": "Summarize Apple's patent and intellectual property litigation exposure.",
        "answer": "Apple regularly faces patent infringement lawsuits and legal claims from entities seeking injunctions or licensing royalties.",
        "terms": ["patent", "litigation"],
        "difficulty": "hard",
        "source": "Item 1A Risk Factors"
    }
]

print(f"Validating all {len(entries)} entries...")
verified_data = []
for it in entries:
    cids = find_cids(it["terms"])
    assert len(cids) > 0, f"ASSERTION FAILED on {it['id']}: {it['query']} with terms {it['terms']}"
    verified_data.append({
        "id": it["id"],
        "category": it["category"],
        "query": it["query"],
        "answer": it["answer"],
        "relevant_chunk_ids": cids[:3],
        "difficulty": it["difficulty"],
        "source": it["source"]
    })

print(f"[SUCCESS] All {len(verified_data)} golden entries verified with 100% chunk ground-truth!")

# Write to both golden_set.json and eval/golden_set.json
root_path = PROJECT_ROOT / "golden_set.json"
eval_path = PROJECT_ROOT / "eval" / "golden_set.json"

with open(root_path, "w", encoding="utf-8") as f:
    json.dump(verified_data, f, indent=2, ensure_ascii=False)
print(f"Saved: {root_path}")

with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(verified_data, f, indent=2, ensure_ascii=False)
print(f"Saved: {eval_path}")
