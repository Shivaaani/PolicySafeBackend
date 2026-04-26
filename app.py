import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from pypdf import PdfReader
import io

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert insurance analyst specialising in Indian insurance policies.
Your job is to read policy text and identify gaps, risks and hidden clauses that could harm the policyholder.

Respond ONLY with valid JSON in this exact format, no preamble, no markdown fences:
{
  "score": <number 0-100>,
  "score_label": "<one of: Critically Under-covered / Needs Attention / Reasonably Covered / Well Covered>",
  "score_color": "<one of: bad / warn / good>",
  "insurer": "<insurer name if found, else null>",
  "policy_type": "<health / term / motor / home / other>",
  "sum_insured": "<sum insured amount if found, else null>",
  "findings": [
    {
      "type": "<critical | warning | ok>",
      "title": "<short title under 8 words>",
      "description": "<2-3 sentence plain English explanation: what the clause says, what it means for the user, what they should do about it>"
    }
  ],
  "summary": "<one sentence overall assessment, specific to this policy>"
}

Focus on: room rent sub-limits, co-payment clauses, waiting periods for pre-existing diseases,
disease-specific sub-limits, exclusions, restoration benefits, no-claim bonus, sum insured
adequacy, missing cover types, employer cover risks, deductibles, network hospital restrictions.
Be specific - use the actual numbers and percentages found in the document.
Maximum 6 findings. Score 0-100 where 0 = dangerously under-covered, 100 = comprehensively covered."""


def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    max_pages = min(len(reader.pages), 15)
    text_parts = []
    for i in range(max_pages):
        page_text = reader.pages[i].extract_text()
        if page_text and page_text.strip():
            text_parts.append(f"--- Page {i+1} ---\n{page_text}")
    full_text = "\n\n".join(text_parts)
    if len(full_text) > 12000:
        full_text = full_text[:12000] + "\n\n[Document truncated - first 15 pages analysed]"
    return full_text


@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        policy_text = ""
        policy_type = request.form.get("policy_type", "health")

        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            file_bytes = file.read()
            if len(file_bytes) > 10 * 1024 * 1024:
                return jsonify({"error": "File too large. Please upload a PDF under 10MB."}), 400
            policy_text = extract_text_from_pdf(file_bytes)
            if not policy_text.strip():
                return jsonify({"error": "Could not extract text from this PDF. It may be a scanned image. Please paste the policy text manually instead."}), 400

        elif "text" in request.form:
            policy_text = request.form.get("text", "").strip()

        else:
            return jsonify({"error": "Please upload a PDF or paste policy text."}), 400

        if len(policy_text.strip()) < 50:
            return jsonify({"error": "Not enough text to analyse. Please provide more policy details."}), 400

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Policy type: {policy_type}\n\nPolicy text:\n{policy_text}\n\nAnalyse this policy and identify all gaps, risks, and hidden clauses. Be specific about rupee amounts and percentages mentioned in the document."
            }]
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse analysis result. Please try again."}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Analysis service error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
