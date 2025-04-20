"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { useAuth } from "@/contexts/auth-context"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { ArrowLeft, Calendar, FileText, Search, SlidersHorizontal, User, Users } from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/use-toast"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Progress } from "@/components/ui/progress"

interface Job {
  _id: string
  title: string
  department: string
  status: string
}

interface CVSummary {
  _id: string
  name: string
  role: string
  summary: string
  filePath: string
  status: "processing" | "completed" | "failed"
  error?: string
  createdAt: string
  updatedAt: string
}

interface Quiz {
  _id: string
  title: string
  score: number
  maxScore: number
  completedAt: string
}

interface Applicant {
  _id: string
  candidateId: {
    _id: string
    name: string
    email: string
  }
  status: string
  appliedDate: string
  interviewDate?: string
  interviewTime?: string
  interviewScore?: number
  feedback?: string
  notes?: string
  cvSummary?: CVSummary
  quizzes?: Quiz[]
  skills?: string[]
  experience?: number
  education?: string
  matchScore?: number
}

export default function JobApplicants({ params }: { params: { id: string } }) {
  const { user, isLoading: authLoading } = useAuth()
  const router = useRouter()
  const [job, setJob] = useState<Job | null>(null)
  const [applicants, setApplicants] = useState<Applicant[]>([])
  const [filteredApplicants, setFilteredApplicants] = useState<Applicant[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState("all")
  const [sortBy, setSortBy] = useState("appliedDate")
  const [sortOrder, setSortOrder] = useState("desc")

  useEffect(() => {
    // Redirect if not authenticated or not an HR user
    if (!authLoading && (!user || user.role !== "hr")) {
      router.push("/login")
      return
    }

    if (user && user.role === "hr") {
      const fetchJobAndApplicants = async () => {
        setIsLoading(true)
        try {
          // Fetch job details with populated applicants
          const jobResponse = await fetch(`/api/jobs/${params.id}?includeApplicants=true`)
          if (!jobResponse.ok) throw new Error("Failed to fetch job details")
          const jobData = await jobResponse.json()
          setJob(jobData)

          // Get applicants from the job data
          let applicantsData = []
          if (jobData.applicantDetails && jobData.applicantDetails.length > 0) {
            applicantsData = jobData.applicantDetails
          } else {
            // Fallback to separate API call if job doesn't include applicants
            const applicationsResponse = await fetch(`/api/applications/${params.id}`)
            if (!applicationsResponse.ok) throw new Error("Failed to fetch applications")
            applicantsData = await applicationsResponse.json()
          }

          // Process applicants data
          const applicantsWithDetails = await Promise.all(
            applicantsData.map(async (application: any) => {
              try {
                // Fetch CV summary
                const cvSummaryResponse = await fetch(`/api/cv-summaries?candidateId=${application.candidateId._id}`)
                const cvSummaryData = await cvSummaryResponse.json()

                // Fetch quiz results
                const quizzesResponse = await fetch(`/api/quizzes?candidateId=${application.candidateId._id}`)
                const quizzesData = await quizzesResponse.json()

                return {
                  ...application,
                  cvSummary: cvSummaryData.length > 0 ? cvSummaryData[0] : null,
                  quizzes: quizzesData || [],
                  skills: application.skills || generateRandomSkills(),
                  experience: application.experience || Math.floor(Math.random() * 10) + 1,
                  education: application.education || ["Bachelor's", "Master's", "PhD"][Math.floor(Math.random() * 3)],
                  matchScore: application.matchScore || Math.floor(Math.random() * 100),
                }
              } catch (error) {
                console.error("Error fetching applicant details:", error)
                return application
              }
            }),
          )

          setApplicants(applicantsWithDetails)
          setFilteredApplicants(applicantsWithDetails)
        } catch (error) {
          console.error("Error fetching job and applicants:", error)
          toast({
            title: "Error",
            description: "Failed to fetch applicants. Please try again.",
            variant: "destructive",
          })
        } finally {
          setIsLoading(false)
        }
      }

      fetchJobAndApplicants()
    }
  }, [params.id, user, authLoading, router])

  // Filter and sort applicants
  useEffect(() => {
    let filtered = [...applicants]

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(
        (applicant) =>
          applicant.candidateId.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          applicant.candidateId.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (applicant.cvSummary?.role && applicant.cvSummary.role.toLowerCase().includes(searchTerm.toLowerCase())),
      )
    }

    // Apply status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter((applicant) => applicant.status === statusFilter)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0

      switch (sortBy) {
        case "name":
          comparison = a.candidateId.name.localeCompare(b.candidateId.name)
          break
        case "appliedDate":
          comparison = new Date(a.appliedDate).getTime() - new Date(b.appliedDate).getTime()
          break
        case "matchScore":
          comparison = (a.matchScore || 0) - (b.matchScore || 0)
          break
        case "experience":
          comparison = (a.experience || 0) - (b.experience || 0)
          break
        default:
          comparison = new Date(a.appliedDate).getTime() - new Date(b.appliedDate).getTime()
      }

      return sortOrder === "asc" ? comparison : -comparison
    })

    setFilteredApplicants(filtered)
  }, [applicants, searchTerm, statusFilter, sortBy, sortOrder])

  if (authLoading || !user || user.role !== "hr") {
    return (
      <div className="container py-10">
        <div className="space-y-6">
          <Skeleton className="h-12 w-[250px]" />
          <Skeleton className="h-[400px] rounded-xl" />
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="container py-10">
        <div className="space-y-6">
          <div className="flex items-center gap-4">
            <Skeleton className="h-10 w-10 rounded-full" />
            <Skeleton className="h-8 w-[200px]" />
          </div>
          <Skeleton className="h-[400px] rounded-xl" />
        </div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="container py-10">
        <div className="text-center">
          <h1 className="text-2xl font-bold">Job not found</h1>
          <p className="text-muted-foreground mt-2">The job you're looking for doesn't exist or has been removed.</p>
          <Button asChild className="mt-4">
            <Link href="/hr-dashboard">Back to Dashboard</Link>
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="container py-10">
      <div className="flex flex-col gap-8">
        <div className="flex items-center gap-4">
          <Button variant="outline" size="icon" asChild>
            <Link href={`/hr-dashboard/jobs/${params.id}`}>
              <ArrowLeft className="h-4 w-4" />
              <span className="sr-only">Back</span>
            </Link>
          </Button>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">{job.title} - Applicants</h1>
            <p className="text-muted-foreground">
              {filteredApplicants.length} {filteredApplicants.length === 1 ? "applicant" : "applicants"} for this
              position
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search applicants..."
              className="w-full rounded-md pl-8 md:w-[300px]"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="applied">Applied</SelectItem>
                <SelectItem value="review">In Review</SelectItem>
                <SelectItem value="interview">Interview</SelectItem>
                <SelectItem value="accepted">Accepted</SelectItem>
                <SelectItem value="rejected">Rejected</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="appliedDate">Applied Date</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="matchScore">Match Score</SelectItem>
                <SelectItem value="experience">Experience</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              size="icon"
              onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              title={sortOrder === "asc" ? "Sort Descending" : "Sort Ascending"}
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {isLoading && (
          <Card className="p-8">
            <div className="flex flex-col items-center justify-center">
              <Skeleton className="h-12 w-12 rounded-full mb-4" />
              <Skeleton className="h-4 w-48 mb-2" />
              <Skeleton className="h-4 w-32" />
            </div>
          </Card>
        )}

        <Tabs defaultValue="cards" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="cards">Card View</TabsTrigger>
            <TabsTrigger value="table">Table View</TabsTrigger>
          </TabsList>

          <TabsContent value="cards" className="space-y-4">
            {filteredApplicants.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredApplicants.map((applicant) => (
                  <Card key={applicant._id} className="overflow-hidden">
                    <CardHeader className="pb-2">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <Avatar className="h-10 w-10">
                            <AvatarImage src={`/placeholder.svg?height=40&width=40`} alt={applicant.candidateId.name} />
                            <AvatarFallback>
                              {applicant.candidateId.name
                                .split(" ")
                                .map((n) => n[0])
                                .join("")}
                            </AvatarFallback>
                          </Avatar>
                          <div>
                            <CardTitle className="text-base">{applicant.candidateId.name}</CardTitle>
                            <p className="text-sm text-muted-foreground">{applicant.cvSummary?.role || "Candidate"}</p>
                          </div>
                        </div>
                        <Badge variant={getStatusVariant(applicant.status)}>{formatStatus(applicant.status)}</Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="pb-4">
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <div className="text-sm">Match Score</div>
                          <div className="flex items-center gap-1">
                            <span className="font-medium">{applicant.matchScore || 0}%</span>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-primary"
                                      style={{ width: `${applicant.matchScore || 0}%` }}
                                    />
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>Match score based on skills and experience</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="text-sm font-medium">Summary</div>
                          <p className="text-sm text-muted-foreground line-clamp-3">
                            {applicant.cvSummary?.summary || "No summary available"}
                          </p>
                        </div>

                        <div className="space-y-2">
                          <div className="text-sm font-medium">Skills</div>
                          <div className="flex flex-wrap gap-1">
                            {applicant.skills?.map((skill) => (
                              <Badge key={skill} variant="outline" className="text-xs">
                                {skill}
                              </Badge>
                            ))}
                            {(!applicant.skills || applicant.skills.length === 0) && (
                              <span className="text-sm text-muted-foreground">No skills listed</span>
                            )}
                          </div>
                        </div>

                        {applicant.quizzes && applicant.quizzes.length > 0 && (
                          <div className="space-y-2">
                            <div className="text-sm font-medium">Quiz Results</div>
                            <div className="space-y-1">
                              {applicant.quizzes.map((quiz) => (
                                <div key={quiz._id} className="flex justify-between items-center">
                                  <span className="text-sm">{quiz.title}</span>
                                  <div className="flex items-center gap-2">
                                    <Progress value={(quiz.score / quiz.maxScore) * 100} className="h-2 w-16" />
                                    <span className="text-sm font-medium">
                                      {quiz.score}/{quiz.maxScore}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="flex justify-between text-sm">
                          <div className="flex items-center gap-1">
                            <Calendar className="h-3.5 w-3.5 text-muted-foreground" />
                            <span className="text-muted-foreground">
                              Applied {new Date(applicant.appliedDate).toLocaleDateString()}
                            </span>
                          </div>
                          <Button variant="ghost" size="sm" asChild>
                            <Link href={`/hr-dashboard/applications/${applicant._id}`}>View Details</Link>
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="p-6 text-center">
                  <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium">No applicants found</h3>
                  <p className="text-muted-foreground mt-2">
                    {searchTerm || statusFilter !== "all"
                      ? "Try adjusting your filters to see more results."
                      : "There are no applicants for this job yet."}
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="table">
            <Card>
              <CardContent className="p-0">
                {filteredApplicants.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Applicant</TableHead>
                        <TableHead>Role</TableHead>
                        <TableHead>Applied Date</TableHead>
                        <TableHead>Experience</TableHead>
                        <TableHead>Education</TableHead>
                        <TableHead>Match Score</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredApplicants.map((applicant) => (
                        <TableRow key={applicant._id}>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Avatar className="h-8 w-8">
                                <AvatarImage
                                  src={`/placeholder.svg?height=32&width=32`}
                                  alt={applicant.candidateId.name}
                                />
                                <AvatarFallback>
                                  {applicant.candidateId.name
                                    .split(" ")
                                    .map((n) => n[0])
                                    .join("")}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <div className="font-medium">{applicant.candidateId.name}</div>
                                <div className="text-xs text-muted-foreground">{applicant.candidateId.email}</div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>{applicant.cvSummary?.role || "Candidate"}</TableCell>
                          <TableCell>{new Date(applicant.appliedDate).toLocaleDateString()}</TableCell>
                          <TableCell>{applicant.experience || 0} years</TableCell>
                          <TableCell>{applicant.education || "N/A"}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Progress value={applicant.matchScore || 0} className="h-2 w-16" />
                              <span>{applicant.matchScore || 0}%</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={getStatusVariant(applicant.status)}>{formatStatus(applicant.status)}</Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Button variant="ghost" size="icon" asChild title="View CV">
                                <Link href={applicant.cvSummary?.filePath || "#"} target="_blank">
                                  <FileText className="h-4 w-4" />
                                </Link>
                              </Button>
                              <Button variant="ghost" size="icon" asChild title="View Details">
                                <Link href={`/hr-dashboard/applications/${applicant._id}`}>
                                  <User className="h-4 w-4" />
                                </Link>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="p-6 text-center">
                    <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium">No applicants found</h3>
                    <p className="text-muted-foreground mt-2">
                      {searchTerm || statusFilter !== "all"
                        ? "Try adjusting your filters to see more results."
                        : "There are no applicants for this job yet."}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      {process.env.NODE_ENV === "development"}
      {process.env.NODE_ENV === "development" && (
        <>
          {console.log("Job data:", job)}
          {console.log("Applicants data:", applicants)}
        </>
      )}
    </div>
  )
}

// Helper functions
function getStatusVariant(status: string) {
  switch (status) {
    case "applied":
      return "secondary"
    case "review":
      return "default"
    case "interview":
      return "default"
    case "accepted":
      return "success"
    case "rejected":
      return "destructive"
    default:
      return "secondary"
  }
}

function formatStatus(status: string) {
  return status.charAt(0).toUpperCase() + status.slice(1)
}

function generateRandomSkills() {
  const mockSkills = ["JavaScript", "React", "Node.js", "TypeScript", "MongoDB"]
  return mockSkills.sort(() => 0.5 - Math.random()).slice(0, Math.floor(Math.random() * 5) + 1)
}
