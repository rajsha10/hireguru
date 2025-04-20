"use client"

import { useEffect, useState } from "react"
import { useParams, useSearchParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Trophy, Award } from "lucide-react"
import confetti from "canvas-confetti"

export default function ResultsPage() {
  const params = useParams()
  const searchParams = useSearchParams()
  const router = useRouter()

  const quizId = params.quizid as String
  const score = Number.parseInt(searchParams.get("score") || "0")
  const total = Number.parseInt(searchParams.get("total") || "0")
  const percentage = Number.parseFloat(searchParams.get("percentage") || "0")

  const [showConfetti, setShowConfetti] = useState(false)

  useEffect(() => {
    if (percentage >= 70) {
      setShowConfetti(true)
      const duration = 3 * 1000
      const animationEnd = Date.now() + duration

      const randomInRange = (min: number, max: number) => {
        return Math.random() * (max - min) + min
      }

      const interval = setInterval(() => {
        const timeLeft = animationEnd - Date.now()

        if (timeLeft <= 0) {
          return clearInterval(interval)
        }

        const particleCount = 50 * (timeLeft / duration)

        confetti({
          particleCount,
          spread: randomInRange(50, 100),
          origin: { y: 0.6 },
        })
      }, 250)

      return () => clearInterval(interval)
    }
  }, [percentage])

  const getGrade = () => {
    if (percentage >= 90) return { text: "Excellent!", color: "text-green-600" }
    if (percentage >= 80) return { text: "Great Job!", color: "text-green-500" }
    if (percentage >= 70) return { text: "Good Work!", color: "text-green-400" }
    if (percentage >= 60) return { text: "Not Bad!", color: "text-yellow-500" }
    if (percentage >= 50) return { text: "You Passed!", color: "text-yellow-600" }
    return { text: "Try Again!", color: "text-red-500" }
  }

  const grade = getGrade()

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">Quiz Results</CardTitle>
        </CardHeader>
        <CardContent className="text-center">
          <div className="mb-6">
            {percentage >= 70 ? (
              <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
            ) : (
              <Award className="h-16 w-16 mx-auto text-blue-500" />
            )}
          </div>

          <h2 className={`text-3xl font-bold mb-2 ${grade.color}`}>{grade.text}</h2>

          <div className="text-2xl font-bold mb-4">
            {score} / {total} correct
          </div>

          <div className="mb-2 text-sm text-gray-500">Your score: {percentage}%</div>

          <Progress value={percentage} className="h-3 mb-6" />

          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <div className="text-3xl font-bold text-green-600">{score}</div>
              <div className="text-sm text-gray-500">Correct</div>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <div className="text-3xl font-bold text-red-600">{total - score}</div>
              <div className="text-sm text-gray-500">Incorrect</div>
            </div>
          </div>

          {percentage >= 70 ? (
            <p className="text-green-600 font-medium">Congratulations! You did great on this quiz!</p>
          ) : percentage >= 50 ? (
            <p className="text-yellow-600 font-medium">You passed! Keep practicing to improve your score.</p>
          ) : (
            <p className="text-red-600 font-medium">Don't worry! Practice more and try again.</p>
          )}
        </CardContent>
        <CardFooter className="flex flex-col space-y-2">
          <Button onClick={() => router.push("/")} className="w-full">
            Take Another Quiz
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
