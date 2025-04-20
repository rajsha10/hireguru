"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { MoreHorizontal, Plus, Search, Users } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useToast } from "@/components/ui/use-toast"

// Job type
type Job = {
  _id: string
  title: string
  department: string
  location: string
  createdAt: string
  applicants?: number
  status: "active" | "closed" | "draft"
}

export function JobsList() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [departmentFilter, setDepartmentFilter] = useState<string>("all")
  const { toast } = useToast()

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/api/jobs')

        if (!response.ok) {
          throw new Error('Failed to fetch jobs')
        }

        const data: Job[] = await response.json()
        setJobs(data)
        setError(null)
      } catch (err) {
        console.error('Error fetching jobs:', err)
        setError('Failed to load jobs. Please try again later.')
        toast({
          title: "Error",
          description: "Failed to load jobs. Please try again later.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchJobs()
  }, [toast])

  const handleStatusChange = async (jobId: string, newStatus: Job["status"]) => {
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

      setJobs(jobs.map(job =>
        job._id === jobId ? { ...job, status: newStatus } : job
      ))

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

  const filteredJobs = jobs.filter((job) => {
    const matchesSearch =
      job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.department.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesStatus = statusFilter === "all" || job.status === statusFilter
    const matchesDepartment = departmentFilter === "all" || job.department === departmentFilter

    return matchesSearch && matchesStatus && matchesDepartment
  })

  const departments = ["all", ...new Set(jobs.map((job) => job.department))]

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  return (
    <div className="container py-10">
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold tracking-tight">Job Roles</h1>
          <p className="text-muted-foreground">Manage and create job roles for your organization.</p>
        </div>

        <div className="flex flex-col sm:flex-row justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search jobs..."
                className="w-full rounded-md pl-8 md:w-[300px]"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[130px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="closed">Closed</SelectItem>
                <SelectItem value="draft">Draft</SelectItem>
              </SelectContent>
            </Select>
            <Select value={departmentFilter} onValueChange={setDepartmentFilter}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Department" />
              </SelectTrigger>
              <SelectContent>
                {departments.map((dept) => (
                  <SelectItem key={dept} value={dept}>
                    {dept === "all" ? "All Departments" : dept}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button asChild>
            <Link href="/hr-dashboard/jobs/create">
              <Plus className="mr-2 h-4 w-4" />
              <span>Create Job</span>
            </Link>
          </Button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>All Job Roles</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
              </div>
            ) : error ? (
              <div className="text-center py-8 text-destructive">{error}</div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Title</TableHead>
                    <TableHead>Department</TableHead>
                    <TableHead>Location</TableHead>
                    <TableHead>Posted Date</TableHead>
                    <TableHead>Applicants</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredJobs.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                        No jobs found. Try adjusting your filters or create a new job.
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredJobs.map((job) => (
                      <TableRow key={job._id}>
                        <TableCell className="font-medium">{job.title}</TableCell>
                        <TableCell>{job.department}</TableCell>
                        <TableCell>{job.location}</TableCell>
                        <TableCell>{formatDate(job.createdAt)}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <Users className="h-3 w-3" />
                            <span>{job.applicants || 0}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge 
                            variant={
                              job.status === "active" ? "default" : 
                              job.status === "draft" ? "outline" : "secondary"
                            }
                          >
                            {job.status === "active" ? "Active" : 
                             job.status === "draft" ? "Draft" : "Closed"}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <MoreHorizontal className="h-4 w-4" />
                                <span className="sr-only">Actions</span>
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuLabel>Actions</DropdownMenuLabel>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem asChild>
                                <Link href={`/hr-dashboard/jobs/${job._id}`}>View Details</Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem asChild>
                                <Link href={`/hr-dashboard/jobs/${job._id}/edit`}>Edit Job</Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem asChild>
                                <Link href={`/hr-dashboard/jobs/${job._id}/applicants`}>View Applicants</Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem 
                                className={job.status === "active" ? "text-destructive" : "text-primary"}
                                onClick={() => handleStatusChange(job._id, job.status === "active" ? "closed" : "active")}
                              >
                                {job.status === "active" ? "Close Job" : "Reopen Job"}
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            )}
          </CardContent>
          <CardFooter className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Showing {filteredJobs.length} of {jobs.length} jobs
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
      </div>
    </div>
  )
}