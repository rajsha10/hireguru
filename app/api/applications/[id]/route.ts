import { NextRequest, NextResponse } from "next/server"
import { cookies } from "next/headers"
import { verify } from "jsonwebtoken"
import connectToDatabase from "@/lib/mongodb"
import Application from "@/models/application"
import User from "@/models/user"

interface Params {
  params: { id: string }
}

export async function GET(req: NextRequest, { params }: Params) {
  try {
    // Verify authentication
    const cookieStore = cookies()
    const token = cookieStore.get("token")?.value

    if (!token) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Verify JWT token
    let decoded
    try {
      decoded = verify(token, process.env.JWT_SECRET || "your_jwt_secret")
    } catch (error) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    // Connect to MongoDB
    await connectToDatabase()

    // Find the user by ID from token
    const userId = typeof decoded === "object" && decoded !== null ? decoded.userId : null

    if (!userId) {
      return NextResponse.json({ error: "Invalid token payload" }, { status: 401 })
    }

    const user = await User.findById(userId).select("-password")

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 })
    }

    // Verify user is HR
    if (user.role !== "hr") {
      return NextResponse.json({ error: "Unauthorized, not an HR user" }, { status: 403 })
    }

    // Fetch application with populated candidate and job data
    const application = await Application.findById(params.id)
      .populate("candidateId", "name email")
      .populate("jobId", "title department location type")

    if (!application) {
      return NextResponse.json({ error: "Application not found" }, { status: 404 })
    }

    return NextResponse.json(application)
  } catch (error) {
    console.error("Error fetching application:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

export async function PUT(req: NextRequest, { params }: Params) {
  try {
    // Verify authentication (similar to GET)
    const cookieStore = cookies()
    const token = cookieStore.get("token")?.value

    if (!token) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Verify JWT token
    let decoded
    try {
      decoded = verify(token, process.env.JWT_SECRET || "your_jwt_secret")
    } catch (error) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    // Connect to MongoDB
    await connectToDatabase()

    // Find the user by ID from token
    const userId = typeof decoded === "object" && decoded !== null ? decoded.userId : null

    if (!userId) {
      return NextResponse.json({ error: "Invalid token payload" }, { status: 401 })
    }

    const user = await User.findById(userId).select("-password")

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 })
    }

    // Verify user is HR
    if (user.role !== "hr") {
      return NextResponse.json({ error: "Unauthorized, not an HR user" }, { status: 403 })
    }

    // Get updated data from request
    const updateData = await req.json()

    // Update application
    const application = await Application.findByIdAndUpdate(
      params.id,
      updateData,
      { new: true, runValidators: true }
    )

    if (!application) {
      return NextResponse.json({ error: "Application not found" }, { status: 404 })
    }

    return NextResponse.json(application)
  } catch (error) {
    console.error("Error updating application:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
