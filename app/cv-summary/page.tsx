"use client"

import type React from "react"
import { useState } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useAuth } from "@/contexts/auth-context"

export default function CvSummary() {
  const { user } = useAuth()
  const [file, setFile] = useState<File | null>(null)
  const [summary, setSummary] = useState<string>("")
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>("")
  const [apiUrl, setApiUrl] = useState<string>("http://localhost:8000/upload") // external summarizer

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0])
    }
  }

  const handleApiUrlChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setApiUrl(event.target.value)
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!file) {
      setError("Please select a file")
      return
    }

    if (!user) {
      setError("User not authenticated")
      return
    }

    setLoading(true)
    setError("")

    try {
      // STEP 1: Send file to external summarization API
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to summarize the CV")
      }

      const data = await response.json()
      const generatedSummary = data.summary

      // STEP 2: Send summary + user role to your own API
      const payload = {
        name: file.name.split(".")[0],
        role: user.role,
        summary: generatedSummary,
      }

      const saveRes = await fetch("/api/upload", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })

      if (!saveRes.ok) {
        const errorData = await saveRes.json()
        throw new Error(errorData.error || "Failed to save summary")
      }

      setSummary(generatedSummary)
    } catch (err) {
      console.error("Upload error:", err)
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container py-8">
      <Card>
        <CardHeader>
          <CardTitle>CV Summary Generator</CardTitle>
          <CardDescription>Upload your CV to generate a summary.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="cv-file" className="text-sm font-medium">
                Upload CV
              </label>
              <input
                type="file"
                id="cv-file"
                onChange={handleFileChange}
                className="flex h-10 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
              />
              <p className="text-xs text-gray-500">Upload your CV in PDF format.</p>
            </div>

            <div className="space-y-2">
              <label htmlFor="api-url" className="text-sm font-medium">
                Summarizer API URL
              </label>
              <div className="flex space-x-2">
                <input
                  id="api-url"
                  type="text"
                  value={apiUrl}
                  onChange={handleApiUrlChange}
                  className="flex-1 p-2 border rounded-md"
                  placeholder="e.g. http://localhost:8000/upload"
                />
                <Button
                  variant="outline"
                  onClick={() => setApiUrl("http://localhost:8000/upload")}
                  type="button"
                  size="sm"
                >
                  Local
                </Button>
              </div>
            </div>

            <Button disabled={loading} type="submit">
              {loading ? "Generating..." : "Generate Summary"}
            </Button>
            {loading && <Progress value={100} />}
            {error && <p className="text-red-500">{error}</p>}
          </form>

          {summary && (
            <div className="space-y-2">
              <h2 className="text-lg font-semibold">Summary:</h2>
              <Textarea readOnly value={summary} className="min-h-[100px]" />
            </div>
          )}
        </CardContent>
        <CardFooter />
      </Card>
    </div>
  )
}
