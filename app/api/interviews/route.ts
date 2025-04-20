import { type NextRequest, NextResponse } from "next/server"
import connectToDatabase from "@/lib/mongodb"
import { getTokenFromCookies, verifyJWT } from "@/lib/jwt"

export async function POST(request: NextRequest) {
  try {
    await connectToDatabase()

    // Get token from cookies
    const token = getTokenFromCookies()
    if (!token) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 })
    }

    // Verify token
    const decoded = verifyJWT(token) as any
    if (!decoded) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    // Get request body
    const body = await request.json()

    // Here you would create an interview in your database
    // For now, we'll just return a mock response

    return NextResponse.json({
      success: true,
      interview: {
        id: Math.random().toString(36).substring(7),
        ...body,
        createdBy: decoded.userId,
        createdAt: new Date(),
      },
    })
  } catch (error: any) {
    console.error("Create interview error:", error)
    return NextResponse.json({ error: error.message || "Something went wrong" }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  try {
    await connectToDatabase()

    // Get token from cookies
    const token = getTokenFromCookies()
    if (!token) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 })
    }

    // Verify token
    const decoded = verifyJWT(token) as any
    if (!decoded) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    // Here you would fetch interviews from your database
    // For now, we'll just return mock data based on the user role

    if (decoded.role === "hr") {
      return NextResponse.json({
        interviews: [
          {
            id: "1",
            position: "Frontend Developer",
            candidate: "Alex Johnson",
            date: "2023-06-15",
            time: "10:00 AM",
            status: "completed",
          },
          {
            id: "2",
            position: "UX Designer",
            candidate: "Sam Wilson",
            date: "2023-06-16",
            time: "2:00 PM",
            status: "scheduled",
          },
        ],
      })
    } else {
      return NextResponse.json({
        interviews: [
          {
            id: "1",
            position: "Frontend Developer",
            company: "TechCorp Inc.",
            date: "2023-06-15",
            time: "10:00 AM",
            status: "completed",
            score: 85,
          },
          {
            id: "2",
            position: "UX Designer",
            company: "DesignHub",
            date: "2023-06-16",
            time: "2:00 PM",
            status: "scheduled",
            score: null,
          },
        ],
      })
    }
  } catch (error: any) {
    console.error("Get interviews error:", error)
    return NextResponse.json({ error: error.message || "Something went wrong" }, { status: 500 })
  }
}
