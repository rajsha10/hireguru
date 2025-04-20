"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { useAuth } from "@/contexts/auth-context"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { CalendarDays, Clock, Filter, Plus, Search, Users, Briefcase } from 'lucide-react'
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/use-toast"

// Types for our data
interface Job {
  _id: string
  title: string
  department: string
  status: string
  createdAt: string
  applicants?: number
  location: string
  type: string
}

interface Application {
  _id: string
  jobId: string
  candidateId: {
    _id: string
    name: string
    email: string
  }
  status: string
  appliedDate: string
  interviewDate?: string
  interviewTime?: string
  job?: {
    title: string
  }
}

export default function HRDashboard() {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const [jobs, setJobs] = useState<Job[]>([])
  const [applications, setApplications] = useState<Application[]>([])
  const [isPageLoading, setIsPageLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [stats, setStats] = useState({
    totalJobs: 0,
    activeJobs: 0,
    totalApplicants: 0,
    scheduledInterviews: 0
  })

  // Fetch jobs from the API
  const fetchJobs = async () => {
    try {
      const response = await fetch('/api/jobs')
      if (!response.ok) throw new Error('Failed to fetch jobs')
      const data = await response.json()
      setJobs(data)
      
      // Calculate stats
      const activeJobsCount = data.filter(job => job.status === 'active').length
      const totalApplicantsCount = data.reduce((acc, job) => acc + (job.applicants || 0), 0)
      
      setStats(prev => ({
        ...prev,
        totalJobs: data.length,
        activeJobs: activeJobsCount,
        totalApplicants: totalApplicantsCount
      }))
      
      return data
    } catch (error) {
      console.error('Error fetching jobs:', error)
      toast({
        title: "Error",
        description: "Failed to fetch jobs. Please try again.",
        variant: "destructive"
      })
      return []
    }
  }

  // Fetch applications from the API
  const fetchApplications = async () => {
    try {
      const response = await fetch('/api/applications')
      if (!response.ok) throw new Error('Failed to fetch applications')
      const data = await response.json()
      setApplications(data)
      
      // Calculate scheduled interviews
      const scheduledCount = data.filter(app => 
        app.status === 'interview' && app.interviewDate
      ).length
      
      setStats(prev => ({
        ...prev,
        scheduledInterviews: scheduledCount
      }))
      
      return data
    } catch (error) {
      console.error('Error fetching applications:', error)
      toast({
        title: "Error",
        description: "Failed to fetch applications. Please try again.",
        variant: "destructive"
      })
      return []
    }
  }

  // Load data on component mount
  useEffect(() => {
    if (!isLoading && user && user.role === "hr") {
      const loadData = async () => {
        setIsPageLoading(true)
        await Promise.all([fetchJobs(), fetchApplications()])
        setIsPageLoading(false)
      }
      
      loadData()
    }
  }, [isLoading, user])

  // Redirect if not authenticated or not an HR user
  useEffect(() => {
    if (!isLoading && (!user || user.role !== "hr")) {
      router.push("/login")
    }
  }, [user, isLoading, router])

  // Filter jobs based on search term
  const filteredJobs = jobs.filter(job => 
    job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    job.department.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Filter applications based on search term
  const filteredApplications = applications.filter(app => 
    app.candidateId.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (app.job?.title && app.job.title.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  // If still checking auth or user is not HR, show loading
  if (isLoading || !user || user.role !== "hr") {
    return (
      <div className="container py-10">
        <div className="space-y-6">
          <Skeleton className="h-12 w-[250px]" />
          <div className="grid gap-6 md:grid-cols-3">
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
          </div>
          <Skeleton className="h-[400px] rounded-xl" />
        </div>
      </div>
    )
  }

  return (
    <div className="container py-10">
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold tracking-tight">Welcome back, {user.name}</h1>
          <p className="text-muted-foreground">Manage your jobs, applications, and interviews all in one place.</p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Total Jobs</CardTitle>
              <CardDescription>Across all departments</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.totalJobs}</div>
              <div className="text-xs text-muted-foreground">
                {stats.activeJobs} active jobs
              </div>
            </CardContent>
            <CardFooter className="pt-2">
              <Button asChild variant="ghost" size="sm" className="w-full justify-start px-2">
                <Link href="/hr-dashboard/jobs">
                  <span>View all jobs</span>
                </Link>
              </Button>
            </CardFooter>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Upcoming Interviews</CardTitle>
              <CardDescription>Scheduled interviews</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.scheduledInterviews}</div>
              <div className="text-xs text-muted-foreground">
                {applications.filter(app => app.status === 'interview').length} candidates in interview stage
              </div>
            </CardContent>
            <CardFooter className="pt-2">
              <Button asChild variant="ghost" size="sm" className="w-full justify-start px-2">
                <Link href="/hr-dashboard/interviews">
                  <span>View all interviews</span>
                </Link>
              </Button>
            </CardFooter>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Total Applicants</CardTitle>
              <CardDescription>In your pipeline</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.totalApplicants}</div>
              <div className="text-xs text-muted-foreground">
                {applications.filter(app => app.status === 'accepted').length} hired candidates
              </div>
            </CardContent>
            <CardFooter className="pt-2">
              <Button asChild variant="ghost" size="sm" className="w-full justify-start px-2">
                <Link href="/hr-dashboard/candidates">
                  <span>View all candidates</span>
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>

        <Tabs defaultValue="jobs" className="w-full">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <TabsList>
              <TabsTrigger value="jobs">Jobs</TabsTrigger>
              <TabsTrigger value="applications">Applications</TabsTrigger>
            </TabsList>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  type="search"
                  placeholder="Search..."
                  className="w-full rounded-md pl-8 md:w-[200px] lg:w-[300px]"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              <Button variant="outline" size="icon">
                <Filter className="h-4 w-4" />
                <span className="sr-only">Filter</span>
              </Button>
              <Button asChild>
                <Link href="/hr-dashboard/jobs/create">
                  <Plus className="mr-2 h-4 w-4" />
                  <span className="hidden sm:inline">Create Job</span>
                  <span className="sm:hidden">New</span>
                </Link>
              </Button>
            </div>
          </div>

          <TabsContent value="jobs" className="space-y-4">
            <Card>
              <CardContent className="p-0">
                <div className="rounded-md border">
                  <div className="grid grid-cols-1 divide-y">
                    {isPageLoading
                      ? Array(4)
                          .fill(0)
                          .map((_, i) => (
                            <div key={i} className="p-4">
                              <div className="space-y-2">
                                <Skeleton className="h-4 w-[250px]" />
                                <Skeleton className="h-4 w-[200px]" />
                              </div>
                            </div>
                          ))
                      : filteredJobs.length > 0 ? (
                          filteredJobs.map((job) => (
                            <div key={job._id} className="p-4">
                              <div className="flex items-center gap-4">
                                <div className="flex-1 space-y-1">
                                  <div className="flex items-center gap-2">
                                    <p className="text-sm font-medium leading-none">{job.title}</p>
                                    <Badge
                                      variant={job.status === "active" ? "default" : "secondary"}
                                      className="ml-auto"
                                    >
                                      {job.status === "active" ? "Active" : "Closed"}
                                    </Badge>
                                  </div>
                                  <p className="text-sm text-muted-foreground">{job.department}</p>
                                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                    <div className="flex items-center gap-1">
                                      <Users className="h-3 w-3" />
                                      <span>{job.applicants || 0} Applicants</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      <Briefcase className="h-3 w-3" />
                                      <span>{job.type}</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      <CalendarDays className="h-3 w-3" />
                                      <span>{new Date(job.createdAt).toLocaleDateString()}</span>
                                    </div>
                                  </div>
                                </div>
                                <Button variant="ghost" size="sm" asChild>
                                  <Link href={`/hr-dashboard/jobs/${job._id}`}>View</Link>
                                </Button>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="p-8 text-center text-muted-foreground">
                            No jobs found. Create your first job posting.
                          </div>
                        )}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex items-center justify-between border-t p-4">
                <div className="text-xs text-muted-foreground">
                  Showing {filteredJobs.length} {filteredJobs.length === 1 ? 'job' : 'jobs'}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" disabled>
                    Previous
                  </Button>
                  <Button variant="outline" size="sm" disabled>
                    Next
                  </Button>
                </div>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="applications" className="space-y-4">
            <Card>
              <CardContent className="p-0">
                <div className="rounded-md border">
                  <div className="grid grid-cols-1 divide-y">
                    {isPageLoading
                      ? Array(4)
                          .fill(0)
                          .map((_, i) => (
                            <div key={i} className="p-4">
                              <div className="flex items-center gap-4">
                                <Skeleton className="h-10 w-10 rounded-full" />
                                <div className="space-y-2">
                                  <Skeleton className="h-4 w-[250px]" />
                                  <Skeleton className="h-4 w-[200px]" />
                                </div>
                              </div>
                            </div>
                          ))
                      : filteredApplications.length > 0 ? (
                          filteredApplications.map((application) => (
                            <div key={application._id} className="p-4">
                              <div className="flex items-center gap-4">
                                <Avatar>
                                  <AvatarImage src={`/placeholder.svg?height=40&width=40`} alt={application.candidateId.name} />
                                  <AvatarFallback>
                                    {application.candidateId.name
                                      .split(" ")
                                      .map((n) => n[0])
                                      .join("")}
                                  </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 space-y-1">
                                  <div className="flex items-center gap-2">
                                    <p className="text-sm font-medium leading-none">{application.candidateId.name}</p>
                                    <Badge
                                      variant={getStatusVariant(application.status)}
                                      className="ml-auto"
                                    >
                                      {formatStatus(application.status)}
                                    </Badge>
                                  </div>
                                  <p className="text-sm text-muted-foreground">{application.job?.title || "Job Title"}</p>
                                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                    <div className="flex items-center gap-1">
                                      <CalendarDays className="h-3 w-3" />
                                      <span>Applied: {new Date(application.appliedDate).toLocaleDateString()}</span>
                                    </div>
                                    {application.interviewDate && (
                                      <div className="flex items-center gap-1">
                                        <Clock className="h-3 w-3" />
                                        <span>Interview: {new Date(application.interviewDate).toLocaleDateString()} {application.interviewTime}</span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                                <Button variant="ghost" size="sm" asChild>
                                  <Link href={`/hr-dashboard/applications/${application._id}`}>View</Link>
                                </Button>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="p-8 text-center text-muted-foreground">
                            No applications found.
                          </div>
                        )}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex items-center justify-between border-t p-4">
                <div className="text-xs text-muted-foreground">
                  Showing {filteredApplications.length} {filteredApplications.length === 1 ? 'application' : 'applications'}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" disabled>
                    Previous
                  </Button>
                  <Button variant="outline" size="sm" disabled>
                    Next
                  </Button>
                </div>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

// Helper functions
function getStatusVariant(status: string) {
  switch (status) {
    case 'applied':
      return 'secondary'
    case 'review':
      return 'default'
    case 'interview':
      return 'default'
    case 'accepted':
      return 'success'
    case 'rejected':
      return 'destructive'
    default:
      return 'secondary'
  }
}

function formatStatus(status: string) {
  return status.charAt(0).toUpperCase() + status.slice(1)
}
