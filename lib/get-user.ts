import { NextRequest } from "next/server"
import { useAuth } from "@/contexts/auth-context"

export const getUserFromSession = async (req: NextRequest) => {
  try {
    const session = await useAuth()
    if (!session || !session.user) {
      throw new Error("No session found or user not authenticated")
    }
    return session.user
  } catch (error) {
    console.error("Error getting user from session:", error)
    return null
  }
}
