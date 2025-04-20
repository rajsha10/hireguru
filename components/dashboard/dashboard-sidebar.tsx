"use client"

import { usePathname } from "next/navigation"
import Link from "next/link"
import { useEffect, useState } from "react"
import { Briefcase, Building2, Calendar, Home, LogOut, Settings, Users } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { useAuth } from "@/contexts/auth-context"

interface HrUser {
  id: string;
  name: string;
  email: string;
  company?: string;
  role: string;
  avatar?: string;
}

export function DashboardSidebar() {
  const pathname = usePathname()
  const { isMobile } = useSidebar()
  const { user, logout } = useAuth()
  const [hrData, setHrData] = useState<HrUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Only fetch HR data if the user is authenticated and has an HR role
    const fetchHrData = async () => {
      try {
        setIsLoading(true)

        if (!user || user.role !== 'hr') {
          setIsLoading(false)
          return
        }

        const response = await fetch('/api/hr/profile')
        
        if (!response.ok) {
          throw new Error('Failed to fetch HR profile')
        }

        const data = await response.json()
        setHrData(data)
      } catch (error) {
        console.error('Error fetching HR data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchHrData()
  }, [user])

  // Get first letter of user's name for avatar fallback
  const getNameInitials = () => {
    if (hrData?.name) {
      return hrData.name.charAt(0).toUpperCase()
    }
    if (user?.name) {
      return user.name.charAt(0).toUpperCase()
    }
    return 'HR'
  }

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center gap-2 px-4 py-2">
          <Building2 className="h-6 w-6" />
          <div className="font-semibold">HR Portal</div>
          {!isMobile && <SidebarTrigger className="ml-auto" />}
        </div>
      </SidebarHeader>
      <SidebarSeparator />
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Dashboard</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={pathname === "/hr-dashboard"}>
                  <Link href="/hr-dashboard">
                    <Home className="h-4 w-4" />
                    <span>Overview</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Recruitment</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={pathname === "/hr-dashboard/jobs" || pathname.startsWith("/hr-dashboard/jobs/")}
                >
                  <Link href="/hr-dashboard/jobs">
                    <Briefcase className="h-4 w-4" />
                    <span>Job Roles</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton 
                  asChild
                  isActive={pathname === "/hr-dashboard/candidates" || pathname.startsWith("/hr-dashboard/candidates/")}
                >
                  <Link href="/hr-dashboard/candidates">
                    <Users className="h-4 w-4" />
                    <span>Candidates</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton 
                  asChild
                  isActive={pathname === "/hr-dashboard/interviews" || pathname.startsWith("/hr-dashboard/interviews/")}
                >
                  <Link href="/hr-dashboard/interviews">
                    <Calendar className="h-4 w-4" />
                    <span>Interviews</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Settings</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton 
                  asChild
                  isActive={pathname === "/hr-dashboard/settings" || pathname.startsWith("/hr-dashboard/settings/")}
                >
                  <Link href="/hr-dashboard/settings">
                    <Settings className="h-4 w-4" />
                    <span>Account Settings</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <div className="p-4">
          <div className="flex items-center gap-3 rounded-md border p-3">
            <Avatar className="h-9 w-9">
              <AvatarImage src={hrData?.avatar || user?.avatar || "/placeholder.svg"} alt="Avatar" />
              <AvatarFallback>{getNameInitials()}</AvatarFallback>
            </Avatar>
            <div className="flex-1 overflow-hidden">
              <p className="text-sm font-medium leading-none">
                {isLoading ? "Loading..." : hrData?.name || user?.name || "HR User"}
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {isLoading ? "" : hrData?.company || user?.company || "HR Manager"}
              </p>
            </div>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => logout()}>
              <LogOut className="h-4 w-4" />
              <span className="sr-only">Log out</span>
            </Button>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}