# HireGuru - AI-Powered Recruitment Assistant

HireGuru is an intelligent recruitment platform that combines resume analysis and AI-driven interviewing capabilities to streamline the hiring process.

## Features

- **Resume Analysis**: Automatically extracts key information from resumes including:
  - Personal Information
  - Education
  - Work Experience
  - Skills
  - Languages
  - Certifications

- **AI Interviewer**: Conducts interactive job interviews with:
  - Multiple interview stages
  - Dynamic question generation
  - Voice-enabled interaction
  - Comprehensive interview summaries

## Setup

1. Clone the repository
```bash
https://github.com/rajsha10/hireguru.git
cd hireguru
```
2. Install dependencies:
```bash
pip install langchain PyPDF2 faiss-cpu python-dotenv pyttsx3
```

3. Configure environment variables:
   - Create a `.env` file
   - Add your HuggingFace API token:
```
HUGGINGFACEHUB_API_TOKEN = "your-token-here"
```

## Usage

### AI Interviewer
1. Run the interviewer:
```bash
python AI_Interviewer.py
```

2. Interview Commands:
- `/restart`: Start a new interview session
- `/exit`: End the interview

## Interview Stages

1. Introduction
2. General Questions
3. Technical Questions
4. Experience Questions
5. Behavioral Questions
6. Closing

## Models Used

- Resume Analysis: Mistral-7B-Instruct-v0.3
- Interviewer: Mixtral-8x7B-Instruct-v0.1
- Embeddings: sentence-transformers/all-MiniLM-L6-v2

## Additional Notes

A prototype has been deployed for this application which is attached with this GitHub repo, the above steps are given for running the AI Interviewer which is not yet integrated in the website but it will added later.
