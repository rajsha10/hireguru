"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { AlertCircle } from "lucide-react"

interface TakeAptitudeTestButtonProps {
  jobId: string
  applicationId: string
  hasScore: boolean
}

export default function TakeAptitudeTestButton({ jobId, applicationId, hasScore }: TakeAptitudeTestButtonProps) {
  const [open, setOpen] = useState(false)
  const [isStarting, setIsStarting] = useState(false)

  const startTest = async () => {
    setIsStarting(true)
    try {
      // In a real application, you would redirect to the test page
      // or load the test interface
      window.location.href = `/candidate-dashboard/aptitude-test/${applicationId}`
    } catch (error) {
      console.error("Failed to start test:", error)
      setIsStarting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant={hasScore ? "outline" : "default"} className={hasScore ? "gap-2" : ""}>
          {hasScore ? (
            <>
              <AlertCircle className="h-4 w-4" />
              View Test Results
            </>
          ) : (
            "Take Aptitude Test"
          )}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{hasScore ? "View Your Test Results" : "Take Aptitude Test"}</DialogTitle>
          <DialogDescription>
            {hasScore
              ? "You have already completed the aptitude test for this job application. You can view your results or retake the test."
              : "You are about to start the aptitude test for this job application. The test will take approximately 30 minutes to complete."}
          </DialogDescription>
        </DialogHeader>

        {!hasScore && (
          <div className="space-y-4 py-4">
            <div className="rounded-md bg-amber-50 p-4 border border-amber-200">
              <div className="flex">
                <div className="flex-shrink-0">
                  <AlertCircle className="h-5 w-5 text-amber-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-amber-800">Important Information</h3>
                  <div className="mt-2 text-sm text-amber-700">
                    <ul className="list-disc space-y-1 pl-5">
                      <li>Once started, the test cannot be paused.</li>
                      <li>Ensure you have a stable internet connection.</li>
                      <li>You will have 30 minutes to complete all questions.</li>
                      <li>Your results will be saved automatically.</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          {hasScore ? (
            <>
              <Button variant="outline" onClick={() => setOpen(false)}>
                Close
              </Button>
              <Button onClick={startTest}>View Results</Button>
            </>
          ) : (
            <>
              <Button variant="outline" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button onClick={startTest} disabled={isStarting}>
                {isStarting ? "Starting..." : "Start Test"}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
