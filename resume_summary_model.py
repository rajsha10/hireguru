from langchain.llms import HuggingFaceHub
import os

def get_model():
    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
        model_kwargs={"temperature": 0.5}, 
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
    )
    return model

def create_prompt(text):
    prompt = f"""You are a professional HR assistant. Your task is to extract the following information from the resume text provided:
        1. Name
        2. Email
        3. Phone number
        4. Education
        5. Work experience
        6. Skills
        7. Speaking Languages Known
        8. Certifications

        Resume text: {text}
        Answer:"""
    return prompt

def extract_answer(full_response, prompt):
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
        return answer

    if full_response.startswith(prompt):
        answer = full_response[len(prompt):].strip()
        return answer

    return full_response

def get_response(text):
    prompt = create_prompt(text)
    model = get_model()
    full_response = model.invoke(prompt)
    answer = extract_answer(full_response, prompt)
    return answer
