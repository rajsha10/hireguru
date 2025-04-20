"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { CalendarDays, Clock, Filter, Search, Sparkles, Briefcase, Award } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"

// Define interfaces for your data types
interface Interview {
  _id: string
  jobTitle: string
  companyName: string
  interviewDate: string
  interviewTime: string
  status: string
  interviewScore?: number
}

interface Application {
  _id: string
  jobTitle: string
  companyName: string
  appliedDate: string
  status: string
}

interface Job {
  _id: string
  title: string
  companyName: string
  department: string
  location: string
  type: string
}

interface MockInterview {
  _id: string
  status: string
  score?: number
}

// Helper function to safely parse JSON and handle errors
async function safelyParseJson(response: Response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    console.error("Failed to parse JSON response:", error);
    console.error("Response text:", text.substring(0, 500) + (text.length > 500 ? "..." : ""));
    return [];
  }
}

export default function CandidateDashboard() {
  const { user, isLoading, error } = useAuth()
  const router = useRouter()
  const [interviews, setInterviews] = useState<Interview[]>([])
  const [applications, setApplications] = useState<Application[]>([])
  const [availableJobs, setAvailableJobs] = useState<Job[]>([])
  const [mockInterviews, setMockInterviews] = useState<MockInterview[]>([])
  const [stats, setStats] = useState({
    upcomingInterviews: 0,
    activeApplications: 0,
    averageScore: 0,
    totalMockInterviews: 0,
    totalApplications: 0,
  })
  const [isPageLoading, setIsPageLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedTab, setSelectedTab] = useState("interviews")
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    if (!isLoading && user) {
      fetchUserData()
    }
  }, [user, isLoading])

  const fetchUserData = async () => {
    try {
      setIsPageLoading(true)
      setFetchError(null)

      // Fetch applications
      try {
        const applicationsRes = await fetch(`/api/candidates/${user._id}/applications`)
        if (!applicationsRes.ok) throw new Error(`Applications fetch failed with status: ${applicationsRes.status}`)
        const applicationsData = await safelyParseJson(applicationsRes)
        setApplications(Array.isArray(applicationsData) ? applicationsData : [])
      } catch (error) {
        console.error("Error fetching applications:", error)
        setApplications([])
      }

      // Fetch interviews
      try {
        const interviewsRes = await fetch(`/api/candidates/${user._id}/interviews`)
        if (!interviewsRes.ok) throw new Error(`Interviews fetch failed with status: ${interviewsRes.status}`)
        const interviewsData = await safelyParseJson(interviewsRes)
        setInterviews(Array.isArray(interviewsData) ? interviewsData : [])
      } catch (error) {
        console.error("Error fetching interviews:", error)
        setInterviews([])
      }

      // Fetch mock interviews
      try {
        const mockInterviewsRes = await fetch(`/api/candidates/${user._id}/mock-interviews`)
        if (!mockInterviewsRes.ok) throw new Error(`Mock interviews fetch failed with status: ${mockInterviewsRes.status}`)
        const mockInterviewsData = await safelyParseJson(mockInterviewsRes)
        setMockInterviews(Array.isArray(mockInterviewsData) ? mockInterviewsData : [])
      } catch (error) {
        console.error("Error fetching mock interviews:", error)
        setMockInterviews([])
      }

      // Fetch available jobs
      try {
        const jobsRes = await fetch(`/api/jobs`)
        if (!jobsRes.ok) throw new Error(`Jobs fetch failed with status: ${jobsRes.status}`)
        
        // Get response text and try to parse it
        const jobsData = await safelyParseJson(jobsRes)
        
        // Ensure we have an array
        if (Array.isArray(jobsData)) {
          console.log("Jobs data fetched successfully:", jobsData.length)
          setAvailableJobs(jobsData)
        } else {
          console.error("Jobs data is not an array:", jobsData)
          setAvailableJobs([])
        }
      } catch (error) {
        console.error("Error fetching jobs:", error)
        setAvailableJobs([])
        setFetchError("Failed to load job listings. Please try again later.")
      }

      // Calculate stats based on what data we were able to fetch
      calculateStats(
        Array.isArray(applications) ? applications : [], 
        Array.isArray(interviews) ? interviews : [], 
        Array.isArray(mockInterviews) ? mockInterviews : []
      )

      setIsPageLoading(false)
    } catch (error) {
      console.error("Error in fetchUserData:", error)
      setIsPageLoading(false)
      setFetchError("Failed to load data. Please refresh the page.")
    }
  }

  const calculateStats = (apps: Application[], ints: Interview[], mocks: MockInterview[]) => {
    const upcomingInterviews = ints.filter(
      i => i.status === "scheduled" && new Date(i.interviewDate) > new Date()
    ).length

    const activeApplications = apps.filter(
      a => a.status === "review" || a.status === "interview"
    ).length

    // Calculate average interview score
    const completedInterviews = [...ints.filter(i => i.status === "completed"), ...mocks.filter(m => m.status === "completed")]
    const totalScore = completedInterviews.reduce((acc, curr) => acc + (curr.interviewScore || curr.score || 0), 0)
    const averageScore = completedInterviews.length ? Math.round(totalScore / completedInterviews.length) : 0

    setStats({
      upcomingInterviews,
      activeApplications,
      averageScore,
      totalMockInterviews: mocks.length,
      totalApplications: apps.length,
    })
  }

  const handleApplyToJob = async (jobId: string) => {
    try {
      // Validate inputs before sending the request
      if (!jobId) {
        throw new Error('Job ID is missing');
      }
      
      if (!user || !user.id) {
        throw new Error('User is not properly authenticated');
      }
      
      // Show loading state
      setIsPageLoading(true);
      
      // Get job details
      const jobDetails = availableJobs.find(job => job._id === jobId);
      if (!jobDetails) {
        throw new Error('Job not found');
      }
      
      // Log the data we're sending to help debug
      console.log("Applying with data:", {
        jobId,
        candidateId: user.id
      });
      
      // Make the API request with explicit string conversion to ensure proper format
      const res = await fetch('/api/applications', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jobId: jobId.toString(),
          candidateId: user.id.toString()
        }),
      });
      
      // Log the raw response
      console.log("API response status:", res.status);
      
      // Get response text first for debugging
      const responseText = await res.text();
      console.log("Raw response:", responseText);
      
      // Then parse it if possible
      let data;
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch (e) {
        console.error("Error parsing response:", e);
        data = { error: "Invalid response format" };
      }
      
      // Check if we received an error about already applying
      if (!res.ok) {
        if (res.status === 400 && data?.error === "You have already applied to this job") {
          alert('You have already applied to this job.');
          
          // If there's an applicationId in the response, we can navigate to it
          if (data?.applicationId) {
            router.push(`/quizpage`);
          } else {
            // Otherwise just switch to applications tab
            setSelectedTab('applications');
          }
          return;
        }
        
        // Handle other errors
        throw new Error(data?.error || 'Failed to apply for this job');
      }
      
      // Success - add the new application to the local state
      const newApplication: Application = {
        _id: data.applicationId,
        jobTitle: jobDetails.title,
        companyName: jobDetails.companyName,
        appliedDate: new Date().toISOString(),
        status: "applied"
      };
      
      setApplications(prev => [newApplication, ...prev]);
      
      // Update stats
      setStats(prev => ({
        ...prev,
        activeApplications: prev.activeApplications + 1,
        totalApplications: prev.totalApplications + 1,
      }));
      
      // Show success message
      alert(`Successfully applied to ${jobDetails.title}`);
      
      // Navigate to the applications tab
      setSelectedTab('applications');
      
    } catch (error) {
      console.error('Error applying to job:', error);
      alert(error instanceof Error ? error.message : 'Failed to apply for this job. Please try again.');
    } finally {
      setIsPageLoading(false);
    }
  };

  // Handle authentication and routing in client
  useEffect(() => {
    // Only redirect if there's an authentication error and not loading
    if (!isLoading && error) {
      console.log('Auth error, redirecting to login:', error)
      router.push('/login')
    }
  }, [isLoading, error, router])

  const filteredJobs = availableJobs.filter(job => 
    job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    job.companyName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    job.department.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="container py-10">
        <div className="space-y-6">
          <Skeleton className="h-12 w-[250px]" />
          <div className="grid gap-6 md:grid-cols-4">
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
          </div>
          <Skeleton className="h-[400px] rounded-xl" />
        </div>
      </div>
    )
  }

  // If there's an error, show error state
  if (error) {
    return (
      <div className="container py-10">
        <Card>
          <CardHeader>
            <CardTitle>Authentication Error</CardTitle>
            <CardDescription>There was a problem accessing your account.</CardDescription>
          </CardHeader>
          <CardContent>
            <p>{error.message || "Please try logging in again."}</p>
          </CardContent>
          <CardFooter>
            <Button onClick={() => router.push('/login')}>Go to Login</Button>
          </CardFooter>
        </Card>
      </div>
    )
  }

  // If no user is loaded yet, show loading state
  if (!user) {
    return (
      <div className="container py-10">
        <div className="space-y-6">
          <Skeleton className="h-12 w-[250px]" />
          <div className="grid gap-6 md:grid-cols-4">
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
            <Skeleton className="h-[180px] rounded-xl" />
          </div>
          <Skeleton className="h-[400px] rounded-xl" />
        </div>
      </div>
    )
  }

  // Check if user has the correct role
  if (user.role !== "candidate") {
    console.log(`User role mismatch: Expected "candidate", got "${user.role}"`)
    router.push("/hr-dashboard")
    return null
  }

  return (
    <div className="container py-10">
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold tracking-tight">Welcome back, {user.name}</h1>
          <p className="text-muted-foreground">Track your interviews, applications, and practice sessions.</p>
        </div>

        {fetchError && (
          <Card className="bg-red-50 border-red-200">
            <CardContent className="pt-4">
              <p className="text-red-600">{fetchError}</p>
            </CardContent>
          </Card>
        )}

        <div className="grid gap-6 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Upcoming Interviews</CardTitle>
              <CardDescription>Next 7 days</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.upcomingInterviews}</div>
              <div className="text-xs text-muted-foreground">
                {interviews.filter((i) => i.status === "completed").length} completed interviews
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Active Applications</CardTitle>
              <CardDescription>In review or interview stage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.activeApplications}</div>
              <div className="text-xs text-muted-foreground">{stats.totalApplications} total applications</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Average Score</CardTitle>
              <CardDescription>Across all interviews</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.averageScore}%</div>
              <div className="text-xs text-muted-foreground">Based on AI evaluation</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Mock Interviews</CardTitle>
              <CardDescription>Total practice sessions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.totalMockInterviews}</div>
              <div className="text-xs text-muted-foreground">
                {mockInterviews.filter(m => m.status === "completed").length} completed sessions
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="interviews" className="w-full" value={selectedTab} onValueChange={setSelectedTab}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <TabsList>
              <TabsTrigger value="interviews">Interviews</TabsTrigger>
              <TabsTrigger value="applications">Applications</TabsTrigger>
              <TabsTrigger value="jobs">Available Jobs</TabsTrigger>
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
              <Button onClick={() => router.push('/mock-interviews/new')}>
                <Sparkles className="mr-2 h-4 w-4" />
                <span className="hidden sm:inline">Practice Interview</span>
                <span className="sm:hidden">Practice</span>
              </Button>
            </div>
          </div>

          <TabsContent value="interviews" className="space-y-4">
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
                      : interviews.length > 0 ? interviews.map((interview) => (
                        <div key={interview._id} className="p-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-1 space-y-1">
                              <div className="flex items-center gap-2">
                                <p className="text-sm font-medium leading-none">{interview.jobTitle || "Interview"}</p>
                                <Badge
                                  variant={interview.status === "completed" ? "secondary" : "default"}
                                  className="ml-auto"
                                >
                                  {interview.status === "completed" ? "Completed" : "Scheduled"}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground">{interview.companyName}</p>
                              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                <div className="flex items-center gap-1">
                                  <CalendarDays className="h-3 w-3" />
                                  <span>{new Date(interview.interviewDate).toLocaleDateString()}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  <span>{interview.interviewTime}</span>
                                </div>
                              </div>
                              {interview.interviewScore !== null && interview.interviewScore !== undefined && (
                                <div className="mt-2">
                                  <div className="flex items-center justify-between text-xs">
                                    <span>Score</span>
                                    <span>{interview.interviewScore}%</span>
                                  </div>
                                  <Progress value={interview.interviewScore} className="h-1.5 mt-1" />
                                </div>
                              )}
                            </div>
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={() => router.push(`/interviews/${interview._id}`)}
                            >
                              {interview.status === "completed" ? "Review" : "Prepare"}
                            </Button>
                          </div>
                        </div>
                      )) : (
                        <div className="p-8 text-center">
                          <p className="text-muted-foreground">No interviews scheduled yet.</p>
                          <Button 
                            variant="link" 
                            onClick={() => setSelectedTab("jobs")}
                            className="mt-2"
                          >
                            Browse available jobs
                          </Button>
                        </div>
                      )}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex items-center justify-between border-t p-4">
                <div className="text-xs text-muted-foreground">Showing {interviews.length} interviews</div>
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
                              <div className="space-y-2">
                                <Skeleton className="h-4 w-[250px]" />
                                <Skeleton className="h-4 w-[200px]" />
                              </div>
                            </div>
                          ))
                      : applications.length > 0 ? applications.map((application) => (
                        <div key={application._id} className="p-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-1 space-y-1">
                              <div className="flex items-center gap-2">
                                <p className="text-sm font-medium leading-none">{application.jobTitle}</p>
                                <Badge
                                  variant={
                                    application.status === "rejected"
                                      ? "destructive"
                                      : application.status === "interview"
                                        ? "default"
                                        : application.status === "accepted"
                                          ? "outline"
                                          : "secondary"
                                  }
                                  className="ml-auto"
                                >
                                  {application.status === "rejected"
                                    ? "Rejected"
                                    : application.status === "interview"
                                      ? "Interview"
                                      : application.status === "accepted"
                                        ? "Accepted"
                                        : "In Review"}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground">{application.companyName}</p>
                              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                <div className="flex items-center gap-1">
                                  <CalendarDays className="h-3 w-3" />
                                  <span>Applied on {new Date(application.appliedDate).toLocaleDateString()}</span>
                                </div>
                              </div>
                            </div>
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={() => router.push(`/quizpage`)}
                            >
                              View
                            </Button>
                          </div>
                        </div>
                      )) : (
                        <div className="p-8 text-center">
                          <p className="text-muted-foreground">You haven't applied to any jobs yet.</p>
                          <Button 
                            variant="link" 
                            onClick={() => setSelectedTab("jobs")}
                            className="mt-2"
                          >
                            Browse available jobs
                          </Button>
                        </div>
                      )}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex items-center justify-between border-t p-4">
                <div className="text-xs text-muted-foreground">Showing {applications.length} applications</div>
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
                      : filteredJobs.length > 0 ? filteredJobs.map((job) => (
                        <div key={job._id} className="p-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-1 space-y-1">
                              <div className="flex items-center gap-2">
                                <p className="text-sm font-medium leading-none">{job.title}</p>
                                <Badge variant="outline" className="ml-auto">
                                  {job.type}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground">{job.companyName}</p>
                              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                <div className="flex items-center gap-1">
                                  <Briefcase className="h-3 w-3" />
                                  <span>{job.department}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <Award className="h-3 w-3" />
                                  <span>{job.location}</span>
                                </div>
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <Button 
                                variant="ghost" 
                                size="sm"
                                onClick={() => router.push(`/jobs/${job._id}`)}
                              >
                                View
                              </Button>
                              <Button 
                                variant="default" 
                                size="sm"
                                onClick={() => handleApplyToJob(job._id)}
                              >
                                Apply
                              </Button>
                            </div>
                          </div>
                        </div>
                      )) : (
                        <div className="p-8 text-center">
                          <p className="text-muted-foreground">
                            {fetchError ? "Failed to load jobs. Please try again later." : "No jobs available matching your search criteria."}
                          </p>
                          {fetchError && (
                            <Button 
                              variant="link" 
                              onClick={fetchUserData}
                              className="mt-2"
                            >
                              Retry
                            </Button>
                          )}
                        </div>
                      )}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex items-center justify-between border-t p-4">
                <div className="text-xs text-muted-foreground">Showing {filteredJobs.length} jobs</div>
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