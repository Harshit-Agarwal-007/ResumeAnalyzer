from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure_config import FORM_RECOGNIZER_ENDPOINT, FORM_RECOGNIZER_KEY
import io

def extract_text_from_bytes(file_bytes: bytes):
    """
    Extracts text from a file (PDF, DOCX, etc.) using Azure Document Intelligence (Form Recognizer).
    """
    try:
        client = DocumentAnalysisClient(
            endpoint=FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
        )

        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=io.BytesIO(file_bytes)
        )
        result = poller.result()

        text_content = []
        for page in result.pages:
            for line in page.lines:
                text_content.append(line.content)
        return "\n".join(text_content)
    except Exception as e:
        return f"❌ Error extracting text: {e}"
