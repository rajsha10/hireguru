import { type NextRequest, NextResponse } from "next/server"
import connectToDatabase from "@/lib/mongodb"
import User from "@/models/user"
import { createJWT, setTokenCookie } from "@/lib/jwt"

export async function POST(request: NextRequest) {
  try {
    await connectToDatabase()

    const body = await request.json()
    const { name, email, password, role, company } = body

    // Check if user already exists
    const existingUser = await User.findOne({ email })
    if (existingUser) {
      return NextResponse.json({ error: "Email already exists" }, { status: 400 })
    }

    // Validate required fields
    if (!name || !email || !password || !role) {
      return NextResponse.json({ error: "Please provide all required fields" }, { status: 400 })
    }

    // Validate HR role requires company
    if (role === "hr" && !company) {
      return NextResponse.json({ error: "Company name is required for HR accounts" }, { status: 400 })
    }

    // Create user
    const user = await User.create({
      name,
      email,
      password,
      role,
      ...(company && { company }),
    })

    // Create token
    const token = createJWT({
      userId: user._id,
      name: user.name,
      role: user.role,
    })

    // Set cookie
    setTokenCookie(token)

    // Return user data (excluding password)
    return NextResponse.json({
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        ...(user.company && { company: user.company }),
      },
    })
  } catch (error: any) {
    console.error("Signup error:", error)
    return NextResponse.json({ error: error.message || "Something went wrong" }, { status: 500 })
  }
}
