import os
import json
import requests as http_requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from pypdf import PdfReader
import io

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
BREVO_LIST_ID = int(os.environ.get("BREVO_LIST_ID", "2"))
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "hello@policysafe.in")
BREVO_SENDER_NAME = os.environ.get("BREVO_SENDER_NAME", "PolicySafe")

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

CONFIRMATION_EMAIL_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#faf9f6;font-family:'DM Sans',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#faf9f6;padding:40px 20px;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;border:1px solid rgba(26,26,24,0.10);overflow:hidden;">
        <!-- Header -->
        <tr>
          <td style="background:#1d6b4a;padding:28px 40px;">
            <p style="margin:0;font-size:22px;font-weight:600;color:#ffffff;letter-spacing:-0.3px;">
              Policy<span style="opacity:0.75">Safe</span>
            </p>
          </td>
        </tr>
        <!-- Body -->
        <tr>
          <td style="padding:40px;">
            <h1 style="margin:0 0 16px;font-size:26px;font-weight:500;color:#1a1a18;line-height:1.3;">
              You're on the list.
            </h1>
            <p style="margin:0 0 20px;font-size:15px;color:#6b6b65;line-height:1.7;">
              Thanks for joining the PolicySafe waitlist. We're onboarding users in Bengaluru first and you'll be among the first to get access.
            </p>
            <p style="margin:0 0 32px;font-size:15px;color:#6b6b65;line-height:1.7;">
              While you wait — if you have an insurance policy document handy, you can already try our free gap analyser. Upload your PDF and find out what your policy actually covers.
            </p>
            <table cellpadding="0" cellspacing="0">
              <tr>
                <td style="background:#1d6b4a;border-radius:8px;">
                  <a href="https://policysafe.in" style="display:inline-block;padding:13px 28px;font-size:15px;font-weight:500;color:#ffffff;text-decoration:none;">
                    Try the free analyser
                  </a>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <!-- What to expect -->
        <tr>
          <td style="padding:0 40px 32px;">
            <p style="margin:0 0 16px;font-size:11px;font-weight:500;letter-spacing:1.5px;text-transform:uppercase;color:#a8a89f;">
              What to expect
            </p>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="padding:12px 0;border-top:1px solid rgba(26,26,24,0.08);">
                  <p style="margin:0;font-size:14px;font-weight:500;color:#1a1a18;">Insurance Health Score</p>
                  <p style="margin:4px 0 0;font-size:13px;color:#6b6b65;">A single score across all your policies — health, life, motor — with specific gaps identified.</p>
                </td>
              </tr>
              <tr>
                <td style="padding:12px 0;border-top:1px solid rgba(26,26,24,0.08);">
                  <p style="margin:0;font-size:14px;font-weight:500;color:#1a1a18;">Life-event alerts</p>
                  <p style="margin:4px 0 0;font-size:13px;color:#6b6b65;">Job change, marriage, baby, home loan — we'll tell you what changes in your coverage needs.</p>
                </td>
              </tr>
              <tr>
                <td style="padding:12px 0;border-top:1px solid rgba(26,26,24,0.08);">
                  <p style="margin:0;font-size:14px;font-weight:500;color:#1a1a18;">Zero spam calls</p>
                  <p style="margin:4px 0 0;font-size:13px;color:#6b6b65;">We will never share your details with insurers or agents. Ever.</p>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <!-- Footer -->
        <tr>
          <td style="padding:20px 40px;border-top:1px solid rgba(26,26,24,0.08);background:#f3f1eb;">
            <p style="margin:0;font-size:12px;color:#a8a89f;line-height:1.6;">
              You're receiving this because you joined the PolicySafe waitlist.<br>
              Built in Bengaluru. Not an insurance company or broker.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""


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
            model="claude-haiku-4-5-20251001",
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


@app.route("/waitlist", methods=["POST"])
def waitlist():
    try:
        data = request.get_json()
        email = (data.get("email") or "").strip().lower()

        if not email or "@" not in email or "." not in email:
            return jsonify({"error": "Please enter a valid email address."}), 400

        if not BREVO_API_KEY:
            return jsonify({"error": "Waitlist service not configured."}), 500

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": BREVO_API_KEY
        }

        # Add contact to Brevo list
        contact_payload = {
            "email": email,
            "listIds": [BREVO_LIST_ID],
            "updateEnabled": True
        }

        contact_resp = http_requests.post(
            "https://api.brevo.com/v3/contacts",
            headers=headers,
            json=contact_payload,
            timeout=10
        )

        # 201 = created, 204 = already exists (updated), both are fine
        if contact_resp.status_code not in (201, 204):
            resp_json = contact_resp.json()
            # Brevo returns 400 with "Contact already exist" — treat as success
            if contact_resp.status_code == 400 and "already exist" in str(resp_json).lower():
                pass  # Already on list, still send confirmation
            else:
                return jsonify({"error": "Could not add to waitlist. Please try again."}), 500

        # Send confirmation email via Brevo transactional
        email_payload = {
            "sender": {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
            "to": [{"email": email}],
            "subject": "You're on the PolicySafe waitlist",
            "htmlContent": CONFIRMATION_EMAIL_HTML
        }

        email_resp = http_requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers=headers,
            json=email_payload,
            timeout=10
        )

        if email_resp.status_code not in (200, 201):
            # Contact was saved but email failed — still return success to user
            # Log the issue silently
            print(f"Confirmation email failed for {email}: {email_resp.text}")

        return jsonify({"success": True, "message": "You're on the list!"})

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
