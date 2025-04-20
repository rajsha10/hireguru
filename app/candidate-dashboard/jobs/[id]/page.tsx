"use client";

import { notFound } from "next/navigation"
import { Briefcase, MapPin, Calendar, Building, Clock, Users } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import connectToDatabase from "@/lib/mongodb"
import Job from "@/models/job"
import Application from "@/models/application"
import TakeAptitudeTestButton from "@/components/take-apptitude-test-button"

interface JobPageProps {
  params: {
    id: string
  }
}

async function getJobDetails(id: string) {
  await connectToDatabase()
  const job = await Job.findById(id)

  if (!job) {
    return null
  }

  return JSON.parse(JSON.stringify(job))
}

async function getApplicationStatus(jobId: string, candidateId: string) {
  await connectToDatabase()
  const application = await Application.findOne({ jobId, candidateId })

  if (!application) {
    return null
  }

  return JSON.parse(JSON.stringify(application))
}

export default async function JobDetailsPage({ params }: JobPageProps) {
  const job = await getJobDetails(params.id)

  if (!job) {
    notFound()
  }

  // In a real application, you would get the candidateId from the session
  // This is a placeholder - replace with actual authentication logic
  const candidateId = "placeholder-candidate-id"

  const application = await getApplicationStatus(params.id, candidateId)

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="grid gap-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold">{job.title}</h1>
            <div className="flex flex-wrap items-center gap-2 mt-2 text-muted-foreground">
              <Building className="h-4 w-4" />
              <span>{job.companyName}</span>
              <span className="mx-2">•</span>
              <MapPin className="h-4 w-4" />
              <span>{job.location}</span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
            {application ? (
              <>
                <Badge variant="outline" className="justify-center">
                  Applied on {new Date(application.appliedDate).toLocaleDateString()}
                </Badge>
                <TakeAptitudeTestButton
                  jobId={params.id}
                  applicationId={application._id}
                  hasScore={!!application.interviewScore}
                />
              </>
            ) : (
              <Button className="w-full sm:w-auto">Apply Now</Button>
            )}
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Job Details</CardTitle>
            <CardDescription>
              Posted by {job.postedByName}, {job.postedByDesignation}
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-2">
                <Briefcase className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Job Type</p>
                  <p className="font-medium">{job.type}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Posted Date</p>
                  <p className="font-medium">{new Date(job.createdAt).toLocaleDateString()}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Users className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm text-muted-foreground">Applicants</p>
                  <p className="font-medium">{job.applicants || 0}</p>
                </div>
              </div>
            </div>

            <Separator />

            <div>
              <h3 className="font-semibold text-lg mb-2">Description</h3>
              <div className="prose max-w-none">
                <p className="whitespace-pre-line">{job.description}</p>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-2">Requirements</h3>
              <div className="prose max-w-none">
                <p className="whitespace-pre-line">{job.requirements}</p>
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex flex-col items-start gap-4">
            <div className="text-sm text-muted-foreground">
              <Clock className="h-4 w-4 inline mr-1" />
              Last updated: {new Date(job.updatedAt).toLocaleDateString()}
            </div>
          </CardFooter>
        </Card>

        {application && application.status !== "applied" && (
          <Card>
            <CardHeader>
              <CardTitle>Application Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <div className="flex justify-between">
                  <span className="font-medium">Status:</span>
                  <Badge
                    variant={
                      application.status === "accepted"
                        ? "success"
                        : application.status === "rejected"
                          ? "destructive"
                          : "secondary"
                    }
                  >
                    {application.status.charAt(0).toUpperCase() + application.status.slice(1)}
                  </Badge>
                </div>

                {application.interviewDate && (
                  <div className="flex justify-between">
                    <span className="font-medium">Interview Date:</span>
                    <span>{new Date(application.interviewDate).toLocaleDateString()}</span>
                  </div>
                )}

                {application.interviewTime && (
                  <div className="flex justify-between">
                    <span className="font-medium">Interview Time:</span>
                    <span>{application.interviewTime}</span>
                  </div>
                )}

                {application.interviewScore !== undefined && (
                  <div className="flex justify-between">
                    <span className="font-medium">Aptitude Test Score:</span>
                    <span>{application.interviewScore}/100</span>
                  </div>
                )}

                {application.feedback && (
                  <div className="mt-4">
                    <span className="font-medium">Feedback:</span>
                    <p className="mt-1 p-3 bg-muted rounded-md">{application.feedback}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
