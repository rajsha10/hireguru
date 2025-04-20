// File: app/api/auth/me/route.ts
import { type NextRequest, NextResponse } from "next/server"
import connectToDatabase from "@/lib/mongodb"
import User from "@/models/user"
import { verifyJWT } from "@/lib/jwt"

export async function GET(request: NextRequest) {
  try {
    await connectToDatabase()

    // Get token from cookies
    const token = request.cookies.get("token")?.value
    if (!token) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 })
    }

    // Verify token
    const decoded = verifyJWT(token) as any
    if (!decoded) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    // Find user - Fixed the select statement
    const user = await User.findById(decoded.userId).select("-password")
    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 })
    }

    // Return user data
    return NextResponse.json({ 
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        ...(user.company && { company: user.company }),
      }
    })
  } catch (error: any) {
    console.error("Get current user error:", error)
    return NextResponse.json({ error: "Failed to get user" }, { status: 500 })
  }
}