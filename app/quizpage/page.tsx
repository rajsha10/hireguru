"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export default function QuizMainPage() {
  const [loading, setLoading] = useState(false)

  const startQuiz = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/quiz`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ num_questions: 4 }),
      })

      if (!response.ok) {
        throw new Error("Failed to start quiz")
      }

      const data = await response.json()
      console.log(data);
      window.location.href = `/quizpage/quiz/${data.quiz_id}`
    } catch (error) {
      console.error("Error starting quiz:", error)
      setLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">Quiz Application</CardTitle>
          <CardDescription>Test your knowledge with our multiple-choice quiz</CardDescription>
        </CardHeader>
        <CardContent className="text-center">
          <p className="mb-4">
            This quiz contains 20 random questions. You'll have to select the correct answer for each question.
          </p>
          <p className="mb-4">Your final score will be displayed at the end of the quiz.</p>
        </CardContent>
        <CardFooter className="flex justify-center">
          <Button onClick={startQuiz} disabled={loading} className="w-full">
            {loading ? "Loading..." : "Start Quiz"}
          </Button>
        </CardFooter>
      </Card>
    </main>
  )
}
