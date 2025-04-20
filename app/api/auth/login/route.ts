// File: app/api/auth/login/route.ts
import { type NextRequest, NextResponse } from "next/server"
import connectToDatabase from "@/lib/mongodb"
import User from "@/models/user"
import { createJWT } from "@/lib/jwt"

export async function POST(request: NextRequest) {
  try {
    await connectToDatabase()

    const body = await request.json()
    const { email, password } = body

    // Validate required fields
    if (!email || !password) {
      return NextResponse.json({ error: "Please provide email and password" }, { status: 400 })
    }

    // Find user
    const user = await User.findOne({ email })
    if (!user) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 })
    }

    // Compare password
    const isPasswordValid = await user.comparePassword(password)
    if (!isPasswordValid) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 })
    }

    // Create token
    const token = createJWT({
      userId: user._id,
      name: user.name,
      role: user.role,
    })

    // Create the response
    const response = NextResponse.json({
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        ...(user.company && { company: user.company }),
      },
    })

    // Set the cookie on the response object
    response.cookies.set('token', token, {
      httpOnly: true,
      secure: false, // this line is OK
      sameSite: 'lax',
      path: '/',
      maxAge: 24 * 60 * 60,
    })    

    return response
  } catch (error: any) {
    console.error("Login error:", error)
    return NextResponse.json({ error: error.message || "Something went wrong" }, { status: 500 })
  }
}