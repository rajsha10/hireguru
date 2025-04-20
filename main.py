import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load questions from file
def load_questions(file_path: str):
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 6:
                    question_text = parts[0]
                    options = parts[1:5]
                    correct_answer = parts[5].strip().upper()
                    questions.append({
                        'question': question_text,
                        'options': options,
                        'answer': correct_answer
                    })
        return questions
    except Exception as e:
        print(f"Error loading questions: {e}")
        return []

# Load questions at startup
questions_data = load_questions('questionsdata.csv')

class QuizRequest(BaseModel):
    num_questions: int = 4

class AnswerSubmission(BaseModel):
    question_id: int
    selected_answer: str

class QuizSubmission(BaseModel):
    answers: List[AnswerSubmission]

@app.get("/")
def read_root():
    return {"message": "Quiz API is running"}

@app.get("/questions/count")
def get_questions_count():
    return {"count": len(questions_data)}

@app.post("/quiz")
def create_quiz(request: QuizRequest):
    if not questions_data:
        raise HTTPException(status_code=500, detail="Questions data not loaded")
    
    num_questions = min(request.num_questions, len(questions_data))
    selected_questions = random.sample(questions_data, num_questions)
    
    # Format questions for the frontend
    quiz_questions = []
    for i, q in enumerate(selected_questions):
        quiz_questions.append({
            "id": i,
            "question": q["question"],
            "options": [
                {"key": "A", "text": q["options"][0]},
                {"key": "B", "text": q["options"][1]},
                {"key": "C", "text": q["options"][2]},
                {"key": "D", "text": q["options"][3]}
            ],
            # Don't send the answer to the client
        })
    
    return {
        "quiz_id": random.randint(1000, 9999),  # Simple random ID
        "questions": quiz_questions,
        "correct_answers": [q["answer"] for q in selected_questions]  # Store for validation
    }

@app.post("/submit")
def submit_quiz(submission: QuizSubmission):
    if not questions_data:
        raise HTTPException(status_code=500, detail="Questions data not loaded")
    
    # In a real app, you would validate against stored correct answers
    # For this example, we'll return a mock score
    total_questions = len(submission.answers)
    correct_answers = random.randint(0, total_questions)  # Mock score
    
    return {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "score_percentage": round((correct_answers / total_questions) * 100, 2) if total_questions > 0 else 0
    }

# For a real implementation, add an endpoint to validate answers against stored correct answers

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
