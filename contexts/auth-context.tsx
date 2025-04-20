"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { useRouter } from "next/navigation"
import { useToast } from "@/hooks/use-toast"

// Mock fallback for development/demo
const mockUser = {
  id: "1",
  name: "Sarah Johnson",
  email: "sarah.johnson@example.com",
  role: "hr" as const,
  avatar: "/placeholder.svg",
  company: "Acme Corp"
}

type Role = "candidate" | "hr"

interface User {
  id: string
  name: string
  email: string
  role: Role
  avatar?: string
  company?: string
}

interface SignupData {
  name: string
  email: string
  password: string
  role: Role
  company?: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (userData: SignupData) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()
  const { toast } = useToast()

  const checkAuthStatus = async () => {
    try {
      const response = await fetch("/api/auth/me", { credentials: "include" })
      if (response.ok) {
        const data = await response.json()
        if (data.user && data.user.role) {
          setUser(data.user)
        } else {
          console.error("Invalid user object from /me")
          setUser(null)
        }
      } else {
        setUser(null)
      }
    } catch (error) {
      console.error("Auth check failed:", error)
      setUser(null)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    checkAuthStatus()
  }, [])

  const login = async (email: string, password: string) => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      })

      const data = await response.json()
      if (!response.ok) throw new Error(data.error || "Login failed")

      setUser(data.user || mockUser)
      toast({
        title: "Login successful",
        description: `Welcome back, ${data.user?.name || mockUser.name}!`,
      })

      router.push(data.user?.role === "candidate" ? "/candidate-dashboard" : "/hr-dashboard")
    } catch (error: any) {
      toast({
        title: "Login failed",
        description: error.message,
        variant: "destructive",
      })
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const signup = async (userData: SignupData) => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(userData),
      })

      const data = await response.json()
      if (!response.ok) throw new Error(data.error || "Signup failed")

      setUser(data.user || mockUser)
      toast({
        title: "Account created",
        description: "Your account has been created successfully!",
      })

      router.push(userData.role === "candidate" ? "/candidate-dashboard" : "/hr-dashboard")
    } catch (error: any) {
      toast({
        title: "Signup failed",
        description: error.message,
        variant: "destructive",
      })
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const logout = async () => {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        credentials: "include",
      })
      setUser(null)
      router.push("/")
      toast({
        title: "Logged out",
        description: "You have been logged out successfully.",
      })
    } catch (error: any) {
      toast({
        title: "Logout failed",
        description: error.message,
        variant: "destructive",
      })
    }
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
