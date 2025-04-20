"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/components/ui/use-toast"
import { ArrowLeft, Calendar, MapPin, Briefcase, Building, User } from "lucide-react"

type Job = {
  _id: string
  title: string
  description: string
  requirements: string
  type: string
  department: string
  location: string
  status: "active" | "draft" | "closed"
  createdAt: string
  postedByName: string
  postedByDesignation: string
  applicants?: number
}

type JobDetailsPageProps = {
  params: {
    id: string
  }
}

export default function JobDetailsPage({ params }: JobDetailsPageProps) {
  const jobId = params.id
  const router = useRouter()
  const { toast } = useToast()
  const [job, setJob] = useState<Job | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchJobDetails = async () => {
      try {
        setIsLoading(true)
        console.log(`Fetching job details for ID: ${jobId}`)
        const response = await fetch(`/api/jobs/${jobId}`)
        
        console.log('API Response status:', response.status)
        
        if (!response.ok) {
          throw new Error(`Failed to fetch job details: ${response.status}`)
        }
  
        const data = await response.json()
        console.log('Job data received:', data)
        setJob(data)
      } catch (err) {
        console.error('Error fetching job details:', err)
        setError(`Failed to load job details: ${err.message}`)
        toast({
          title: "Error",
          description: `Failed to load job details: ${err.message}`,
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }
  
    if (jobId) {
      console.log('JobID found, triggering fetch:', jobId)
      fetchJobDetails()
    } else {
      console.log('No JobID found in params')
      setIsLoading(false)
      setError("No job ID provided")
    }
  }, [jobId, toast])

  const handleStatusChange = async (newStatus: "active" | "closed") => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: newStatus }),
      })

      if (!response.ok) {
        throw new Error('Failed to update job status')
      }

      setJob(prev => prev ? { ...prev, status: newStatus } : prev)

      toast({
        title: "Status Updated",
        description: `Job has been ${newStatus === 'active' ? 'reopened' : 'closed'}.`,
      })
    } catch (err) {
      console.error('Error updating job status:', err)
      toast({
        title: "Error",
        description: "Failed to update job status. Please try again.",
        variant: "destructive",
      })
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    })
  }

  if (isLoading) {
    return (
      <div className="container py-10">
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
        </div>
      </div>
    )
  }

  if (error || !job) {
    return (
      <div className="container py-10">
        <div className="text-center py-8 text-destructive">
          {error || "Job not found."}
          <div className="mt-4">
            <Button variant="outline" onClick={() => router.push("/hr-dashboard/jobs")}>
              Back to Jobs
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="container py-10">
      <div className="flex flex-col gap-8 max-w-4xl mx-auto">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => router.push("/hr-dashboard/jobs")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">{job.title}</h1>
          <Badge 
            variant={
              job.status === "active" ? "default" : 
              job.status === "draft" ? "outline" : "secondary"
            }
            className="ml-2"
          >
            {job.status === "active" ? "Active" : 
             job.status === "draft" ? "Draft" : "Closed"}
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Job Description</CardTitle>
            </CardHeader>
            <CardContent className="prose max-w-none">
              <p className="whitespace-pre-wrap">{job.description}</p>
              <h3 className="text-lg font-semibold mt-6">Requirements</h3>
              <p className="whitespace-pre-wrap">{job.requirements}</p>
            </CardContent>
          </Card>

          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Job Details</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <div className="flex items-center gap-2">
                  <Briefcase className="h-4 w-4 text-muted-foreground" />
                  <span>{job.type}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Building className="h-4 w-4 text-muted-foreground" />
                  <span>{job.department}</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span>{job.location}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span>Posted on {formatDate(job.createdAt)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <span>Posted by {job.postedByName}, {job.postedByDesignation}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Actions</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <Button asChild>
                  <Link href={`/hr-dashboard/jobs/${job._id}/edit`}>Edit Job</Link>
                </Button>
                <Button asChild variant="outline">
                  <Link href={`/hr-dashboard/jobs/${job._id}/applicants`}>View Applicants ({job.applicants || 0})</Link>
                </Button>
                <Button 
                  variant={job.status === "active" ? "destructive" : "default"}
                  onClick={() => handleStatusChange(job.status === "active" ? "closed" : "active")}
                >
                  {job.status === "active" ? "Close Job" : "Reopen Job"}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
