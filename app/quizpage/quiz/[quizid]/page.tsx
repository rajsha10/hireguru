"use client"

import { useState, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"

interface Option {
  key: string
  text: string
}

interface Question {
  id: number
  question: string
  options: Option[]
}

interface QuizData {
  quiz_id: string
  questions: Question[]
}

export default function QuizPage() {
  const params = useParams()
  const router = useRouter()
  const quizId = params.quizid as string

  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [quizData, setQuizData] = useState<QuizData | null>(null)
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [answers, setAnswers] = useState<Record<number, string>>({})

  useEffect(() => {

    console.log("Params:", params);
  console.log("Quiz ID:", quizId);
    const fetchQuiz = async () => {
        if (!quizId) {
            console.error("Quiz ID is undefined");
            setLoading(false);
            return;
          }
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/quiz/${quizId}`)

        if (!response.ok) {
          throw new Error("Failed to fetch quiz")
        }

        const data = await response.json()
        console.log("datadata",data)
        setQuizData(data)
        setLoading(false)
      } catch (error) {
        console.error("Error fetching quiz:", error)
        setLoading(false)
      }
    }

    fetchQuiz()
  }, [quizId])

  const handleAnswerSelect = (value: string) => {
    if (!quizData) return

    setAnswers({
      ...answers,
      [quizData.questions[currentQuestionIndex].id]: value,
    })
  }

  const goToNextQuestion = () => {
    if (!quizData) return

    if (currentQuestionIndex < quizData.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1)
    }
  }

  const goToPreviousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1)
    }
  }

  const submitQuiz = async () => {
    if (!quizData) return
    
    setSubmitting(true)
    
    try {
      const answersArray = Object.entries(answers).map(([questionId, selectedAnswer]) => ({
        question_id: Number.parseInt(questionId),
        selected_answer: selectedAnswer,
      }))
      
      console.log("Submitting data:", {
        quiz_id: quizId,
        answers: answersArray,
      })
  
      // Make sure the URL matches exactly what's in your FastAPI app
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/submit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          quiz_id: quizId,
          answers: answersArray,
        }),
      })
  
      if (!response.ok) {
        throw new Error(`Failed to submit quiz: ${response.status} ${response.statusText}`)
      }
  
      const result = await response.json()
      router.push(
        `/quizpage/result/${quizId}?score=${result.correct_answers}&total=${result.total_questions}&percentage=${result.score_percentage}`,
      )
    } catch (error) {
      console.error("Error submitting quiz:", error)
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
        <Card className="w-full max-w-2xl">
          <CardHeader>
            <Skeleton className="h-8 w-3/4 mb-2" />
            <Skeleton className="h-4 w-1/2" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-24 w-full mb-4" />
            <div className="space-y-3">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          </CardContent>
          <CardFooter>
            <Skeleton className="h-10 w-full" />
          </CardFooter>
        </Card>
      </div>
    )
  }

  console.log(quizData);

  if (!quizData) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-xl text-center text-red-500">Quiz Not Found</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-center">Sorry, we couldn't find the quiz you're looking for.</p>
          </CardContent>
          <CardFooter className="flex justify-center">
            <Button onClick={() => router.push("/")}>Return to Home</Button>
          </CardFooter>
        </Card>
      </div>
    )
  }

  const currentQuestion = quizData.questions[currentQuestionIndex]
  const progress = (Object.keys(answers).length / quizData.questions.length) * 100
  const isLastQuestion = currentQuestionIndex === quizData.questions.length - 1
  const canSubmit = Object.keys(answers).length === quizData.questions.length

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <div className="flex justify-between items-center mb-2">
            <CardTitle className="text-xl">
              Question {currentQuestionIndex + 1} of {quizData.questions.length}
            </CardTitle>
            <span className="text-sm text-gray-500">
              {Object.keys(answers).length} of {quizData.questions.length} answered
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </CardHeader>
        <CardContent>
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4">{currentQuestion.question}</h3>
            <RadioGroup
              value={answers[currentQuestion.id] || ""}
              onValueChange={handleAnswerSelect}
              className="space-y-3"
            >
              {currentQuestion.options.map((option) => (
                <div key={option.key} className="flex items-center space-x-2 border p-3 rounded-md hover:bg-gray-50">
                  <RadioGroupItem value={option.key} id={`option-${option.key}`} />
                  <Label htmlFor={`option-${option.key}`} className="flex-1 cursor-pointer">
                    <span className="font-medium mr-2">{option.key}.</span>
                    {option.text}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={goToPreviousQuestion} disabled={currentQuestionIndex === 0}>
            Previous
          </Button>
          <div className="flex space-x-2">
            {isLastQuestion ? (
              <Button
                onClick={submitQuiz}
                disabled={!canSubmit || submitting}
                className="bg-green-600 hover:bg-green-700"
              >
                {submitting ? "Submitting..." : "Submit Quiz"}
              </Button>
            ) : (
              <Button onClick={goToNextQuestion}>Next</Button>
            )}
          </div>
        </CardFooter>
      </Card>

      {canSubmit && !isLastQuestion && (
        <div className="mt-4">
          <Button onClick={submitQuiz} disabled={submitting} className="bg-green-600 hover:bg-green-700">
            {submitting ? "Submitting..." : "Submit Quiz"}
          </Button>
        </div>
      )}
    </div>
  )
}
