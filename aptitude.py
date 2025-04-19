import random

def load_questions(file_path):
    questions = []
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

def run_quiz(questions, num_questions=20):
    selected_questions = random.sample(questions, min(num_questions, len(questions)))
    score = 0
    for i, q in enumerate(selected_questions, 1):
        print(f"\nQ{i}: {q['question']}")
        print(f"A. {q['options'][0]}")
        print(f"B. {q['options'][1]}")
        print(f"C. {q['options'][2]}")
        print(f"D. {q['options'][3]}")
        user_ans = input("Your answer (A/B/C/D): ").strip().upper()
        if user_ans == q['answer']:
            print("✅ Correct!")
            score += 1
        else:
            print(f"❌ Incorrect. The correct answer was {q['answer']}")
    print(f"\n🎯 Quiz complete! Your score: {score}/{num_questions}")

# Usage example
# Replace 'questions.txt' with the path to your dataset file
if __name__ == "__main__":
    all_questions = load_questions('questionsdata.csv')
    run_quiz(all_questions)
