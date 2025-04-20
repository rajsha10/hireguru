import random
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid

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
questions_data = load_questions('./questionsdata.csv')

# Store active quizzes
active_quizzes = {}

class QuizRequest(BaseModel):
    num_questions: int = 4

class AnswerSubmission(BaseModel):
    question_id: int
    selected_answer: str

class QuizSubmission(BaseModel):
    quiz_id: str
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
    
    # Generate a unique quiz ID
    quiz_id = str(uuid.uuid4())
    
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
            ]
        })
    
    # Store the quiz data for validation later
    active_quizzes[quiz_id] = {
        "questions": selected_questions,
        "submitted": False
    }
    
    return {
        "quiz_id": quiz_id,
        "questions": quiz_questions
    }

@app.post("/submit")
def submit_quiz(submission: QuizSubmission):
    quiz_id = submission.quiz_id
    
    if quiz_id not in active_quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    if active_quizzes[quiz_id]["submitted"]:
        raise HTTPException(status_code=400, detail="Quiz already submitted")
    
    quiz_data = active_quizzes[quiz_id]
    questions = quiz_data["questions"]
    
    if len(submission.answers) != len(questions):
        raise HTTPException(status_code=400, detail="Number of answers doesn't match number of questions")
    
    # Calculate score
    correct_count = 0
    results = []
    
    for answer in submission.answers:
        question_id = answer.question_id
        if question_id < 0 or question_id >= len(questions):
            raise HTTPException(status_code=400, detail=f"Invalid question ID: {question_id}")
        
        correct_answer = questions[question_id]["answer"]
        is_correct = answer.selected_answer.upper() == correct_answer
        
        if is_correct:
            correct_count += 1
            
        results.append({
            "question_id": question_id,
            "is_correct": is_correct,
            "correct_answer": correct_answer
        })
    
    # Mark quiz as submitted
    active_quizzes[quiz_id]["submitted"] = True
    
    return {
        "total_questions": len(questions),
        "correct_answers": correct_count,
        "score_percentage": round((correct_count / len(questions)) * 100, 2),
        "results": results
    }

@app.get("/quiz/{quiz_id}")
def get_quiz(quiz_id: str):
    if quiz_id not in active_quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz_data = active_quizzes[quiz_id]
    questions = quiz_data["questions"]
    
    # Format questions for the frontend
    quiz_questions = []
    for i, q in enumerate(questions):
        quiz_questions.append({
            "id": i,
            "question": q["question"],
            "options": [
                {"key": "A", "text": q["options"][0]},
                {"key": "B", "text": q["options"][1]},
                {"key": "C", "text": q["options"][2]},
                {"key": "D", "text": q["options"][3]}
            ]
        })
    
    return {
        "quiz_id": quiz_id,
        "questions": quiz_questions,
        "submitted": quiz_data["submitted"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)