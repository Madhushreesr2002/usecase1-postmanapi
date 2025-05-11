from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import json
import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel

app = FastAPI()

# Initialize Vertex AI
vertexai.init(project="unique-epigram-458904-p8", location="us-central1")

def extract_text_from_pdf(pdf_file: bytes):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    full_text = []

    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            img = img.convert("L")
            text = pytesseract.image_to_string(img)
        full_text.append(text.strip())

    return "\n".join(full_text)

def extract_json_from_response(response_text):
    match = re.search(r"({.*})", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()

# Your prompt template (shortened here for brevity)
PROMPT_TEMPLATE = """
Extract structured details from this insurance policy document.
Ensure the output follows the exact JSON structure below with accurate extracted values.


Ensure that:
start_date and end_date are always in the **DD-MMM-YYYY** format (e.g., 04-Feb-2025).
dob fields for the proposer, insured members, nominees, and appointees **must** also follow the **DD-MMM-YYYY** format.
**Insured members' DOB and nominee DOB must follow the same DD-MMM-YYYY format as the start date and end date.**
-***For every document there must be a start_date and end_date, look for policy section and give the output.***


**Date Format:**  
   - **Always use `DD-MMM-YYYY` format** (e.g., `13-Feb-2002`).  
   - This applies to **DOB, policy start date, policy end date, and all date fields**.  
   - If the extracted date is in `DD/MM/YYYY` or `YYYY-MM-DD`, **convert it before returning**.


####  Correct Outputs:**
   - `"dob": "13-Feb-2002"`  
   - `"policy_start_date": "01-Apr-2024"`  
   - `"policy_end_date": "01-Apr-2025"`    


####  Incorrect Outputs:**  
   - `"dob": "XX-XX-2002"`  
   - `"policy_start_date": "XX-XX-2024"`  
   - `"policy_end_date": "XX-XX-2025"`    


####  Incorrect Outputs:**
   - `"13/02/2002"`  
   - `"2002-02-13"`  
   - `"2002/02/13"`  


####  Correct Outputs:**
   - `"13-Feb-2002"`  
   - `"02-Apr-2024"`  
**If a date is found in a different format, always convert it to `DD-MMM-YYYY` before returning.**  
**Do not generate `XX-XX-YYYY` placeholders. If the date is completely missing, return `null` instead.**  


### Important Note:
- `policy_type` refers to **the category of the policy** such as:
   - `"Comprehensive Health Policy"`, `"Individual Health Policy"`, `"Group Health Policy"`, `"Personal Accident Policy"`, etc.
   - These should be generic policy categories — not specific names of plans or variants.
   - If the policy type is not clearly mentioned, return `"policy_type": "Unknown"`.
   - *** if it has Accidental injury section and  Accidental Injury Benefits , name policy_type = "Personal Accident Policy" ***


- `plan_name` refers to **the specific variant or branded name** of the policy such as:
   - `"Silver-WW Exc US/Canada"`, `"Optima Restore"`, `"Gold Plus Worldwide"`.


### policy_type must be the generic category of the insurance policy. Choose from standard categories such as:


"Term Insurance Policy"


"Personal Accident Policy"


"Whole Life Insurance Policy"


"Endowment Policy"


"Unit Linked Insurance Policy" (ULIP)


"Group Health Insurance Policy"


"Critical Illness Insurance Policy"


"Health Insurance Policy"


"Top-up Health Insurance Policy"


"Travel Insurance Policy"


"Motor Insurance Policy"


"Home Insurance Policy"


"Fire Insurance Policy"


Do not include brand names or qualifiers like "Individual", "Super", "Platinum", etc. unless they are part of a standard category (e.g. Individual vs Group Health is allowed).


**Examples:**
-  Correct:
  ```json
  "policy_type": "Personal Accident Policy",
  "plan_name": "Silver-WW Exc US/Canada"


***Maintain the standard **Indian pricing format** for monetary values, e.g., ₹1,00,000, ₹26,25,000.  
- **Always return the actual rupee symbol (₹) and not Unicode (`\u20b9`)**.  
- Ensure **comma separators are correctly placed** in the **Indian numbering format** (lakh & crore system).  
- **Never return plain numbers** like `1000000`; instead, format as `₹10,00,000`.  
- **If commas are missing, correct them before returning.**  
- If the rupee symbol is missing or incorrect (Unicode `\u20b9`), **replace it with ₹ in the final output.**
- **For INR, use ₹ and comma separators (e.g., ₹1,50,000)**
- **For USD, use $ and comma separators (e.g., $3,000)**
***


####  Incorrect Outputs:
- `"base": "USD 50,000.00"` (Do not use USD)


####  Correct Outputs:
- `"base": "$ 50,000.00"` (Instead of USD use symbol $)


####  Incorrect Outputs:
- `"base": "\u20b910,00,000"`  (Unicode issue)
- `"base": "1000000"` (Missing ₹ symbol and commas)
- `"base": "5000000"` (Missing ₹ symbol and commas)


####  Correct Outputs:
- `"base": "₹10,00,000"`
- `"base": "₹50,00,000"`


###  For the "addons_opted" section, format all coverage amounts based on the currency:


- If the amount is in Indian Rupees (INR), prefix it with the rupee symbol (₹) and use comma separators (e.g., ₹1,00,000).
- If the amount is in US Dollars (USD), prefix it with the dollar symbol ($) and use comma separators (e.g., $2,500).
- Do not return plain numbers without currency formatting.


### Extracting Additional Coverage Details


- Identify **all additional coverage benefits** in the policy.  
- Coverage benefits are typically listed under sections like "Coverage Benefits," "Additional Benefits," or "Rider Benefits."  
- **Strictly avoid returning `null`** unless the policy explicitly states "No additional coverage available."
- **If multiple coverage options exist, list them as an array of bullet points.**
Display the **add-on coverage amount** in numeric format, ensuring consistency.  


Return only the JSON object in the following format:


{{
  "policy_number": "",
  "coverage": {{ "base": "", "permanent_total_disablement": "", "permanent_partial_disablement": "" }},
  "proposal_number": "",
  "start_date": "",
  "end_date": "",
  "insurer_name": "",
  "policy_type": "",
  "product_name": "",
  "plan_name": "",
  "payment_mode": "",
  "payment_term": "",
  "total_premium":"",
  "proposer_details": {{ "name": "", "dob": "", "email": "", "phone_number": "", "address": "", "pincode": "", "id_proofs": [ {{ "type": "", "doc_no": "" }} ], "bank_details": {{ "account_no": "", "ifsc_code": "", "bank_name": "" }} }},
  "insured_members": [
    {{ "name": "", "dob": "", "gender": "", "relationship_with_proposer": "", "peds_list": [], "id_proofs": [ {{ "type": "", "doc_no": "" }} ] }}
  ],
  "addons_opted": [ {{ "addon_name": "", "addon_coverage_amount": "" }} ],
  "nominee_details": [ {{ "name": "", "relationship_with_proposer": "", "dob": "", "percentage_of_claim": "", "phone": "", "email": "", "id_proofs": [ {{ "type": "", "doc_no": "" }} ], "appointee_details": {{ "name": "", "dob": "", "address": "", "pincode": "", "phone": "", "email": "", "relationship_with_nominee": "", "id_proofs": [ {{ "type": "", "doc_no": "" }} ] }} }} ],
  "other_details": {{ "additional_coverage": "", "policy_terms": "", "salient_policy_terms": [] }}
}}


Now extract from the following document:


{document_text}
"""

@app.post("/extract-policy")
async def extract_policy(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "Only PDF files are accepted."}, status_code=400)

    try:
        pdf_bytes = await file.read()
        document_text = extract_text_from_pdf(pdf_bytes)
        prompt = PROMPT_TEMPLATE.format(document_text=document_text)

        model = GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content([prompt])
        response_text = getattr(response, "text", "")
        json_text = extract_json_from_response(response_text)

        try:
            extracted = json.loads(json_text)
            return JSONResponse(content=extracted)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "Gemini returned invalid JSON", "raw": json_text}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
