"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/hooks/use-toast"

export default function ProfilePage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const { toast } = useToast()
  const [isPageLoading, setIsPageLoading] = useState(true)
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  })

  const [latestSummary, setLatestSummary] = useState<string | null>(null)

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const res = await fetch("/api/my-summary", {
          credentials: "include",
        })
        if (res.ok) {
          const data = await res.json()
          console.log(data.summary);
          setLatestSummary(data.summary)
        }
      } catch (err) {
        console.error("Failed to fetch summary", err)
      }
    }
  
    if (user) fetchSummary()
  }, [user])  

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault()

    toast({
      title: "Profile updated",
      description: "Your profile information has been updated successfully.",
    })
  }

  const handlePasswordUpdate = async (e: React.FormEvent) => {
    e.preventDefault()

    // Validate passwords match
    if (formData.newPassword !== formData.confirmPassword) {
      toast({
        title: "Passwords don't match",
        description: "Please make sure your new passwords match.",
        variant: "destructive",
      })
      return
    }

    toast({
      title: "Password updated",
      description: "Your password has been updated successfully.",
    })
  }

  // If still checking auth or user is not logged in, show loading
  if (isLoading || !user) {
    return (
      <div className="container py-10">
        <div className="space-y-6 max-w-2xl mx-auto">
          <Skeleton className="h-12 w-[250px]" />
          <Skeleton className="h-[400px] rounded-xl" />
        </div>
      </div>
    )
  }

  return (
    <div className="container py-10">
      <div className="max-w-2xl mx-auto">
        <div className="flex flex-col gap-2 mb-8">
          <h1 className="text-3xl font-bold tracking-tight">Your Profile</h1>
          <p className="text-muted-foreground">Manage your account settings and change your password</p>
        </div>

        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="password">Password</TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="space-y-4">
          {user && (
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>Candidate Details</CardTitle>
                <CardDescription>Here’s what we know about you:</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p><strong>Name:</strong> {user.name}</p>
                <p><strong>Email:</strong> {user.email}</p>
                <p><strong>Role:</strong> {user.role}</p>
                {latestSummary && (
                  <div>
                    <p className="font-medium text-black">Latest CV Summary:</p>
                    <div className="border rounded-md p-2 text-muted-foreground whitespace-pre-wrap bg-gray-50 text-sm">
                      {latestSummary}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          </TabsContent>

          <TabsContent value="password" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Change Password</CardTitle>
                <CardDescription>Update your password to keep your account secure.</CardDescription>
              </CardHeader>
              <form onSubmit={handlePasswordUpdate}>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="currentPassword">Current Password</Label>
                    <Input
                      id="currentPassword"
                      name="currentPassword"
                      type="password"
                      value={formData.currentPassword}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="newPassword">New Password</Label>
                    <Input
                      id="newPassword"
                      name="newPassword"
                      type="password"
                      value={formData.newPassword}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword">Confirm New Password</Label>
                    <Input
                      id="confirmPassword"
                      name="confirmPassword"
                      type="password"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      required
                    />
                  </div>
                </CardContent>
                <CardFooter>
                  <Button type="submit">Update Password</Button>
                </CardFooter>
              </form>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
