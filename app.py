from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import json
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

api_key = "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"  # Hugging face API

def pdf_page_to_text(pdf_path, page_number, quality=300):
    try:
        # Convert PDF's first page into an image
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=quality)

        if images:
            # Get the image from the PDF
            image = images[0]

            # Use Tesseract to do OCR on the image
            text = pytesseract.image_to_string(image)
            return text
        else:
            print(f"Error: Unable to extract page {page_number} from the PDF.")
            return None

    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def get_llm_response(question, answer):
    template = "Question: {question}\n{answer}"
    prompt = PromptTemplate(template=template, input_variables=["question", "answer"])
    llm_chain = LLMChain(
        prompt=prompt,
        llm=HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.1, "max_length": 4000},
            huggingfacehub_api_token=api_key,
        ),
    )
    response = llm_chain.run(question=question, answer=answer)
    return response

if __name__ == "__main__":
    # Extract text from PDF image
    input_pdf_path = r"C:\Users\punam.chaudhari\Downloads\22083E3W3042933732_Import1.PDF"
    extracted_text = pdf_page_to_text(input_pdf_path, page_number=1, quality=300)

    # Truncate the text to fit within the token limit
    truncated_text = extracted_text[:900]  # Adjust the length as needed
    #print(truncated_text)
    # Continue with LangChain processing
    question1 = "What is bank name?"
    response1 = get_llm_response(truncated_text, question1)

    question2 = "What is bank Address?"
    response2 = get_llm_response(truncated_text, question2)

    response_dict = {"Bank Name": response1, "Bank Address": response2}
    json_response = json.dumps(response_dict, indent=2)
    print(json_response)
