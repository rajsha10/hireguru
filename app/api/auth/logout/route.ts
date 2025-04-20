import { NextResponse } from "next/server"
import { clearTokenCookie } from "@/lib/jwt"

export async function POST() {
  try {
    // Clear the token cookie
    clearTokenCookie()

    return NextResponse.json({ message: "Logged out successfully" })
  } catch (error: any) {
    console.error("Logout error:", error)
    return NextResponse.json({ error: error.message || "Something went wrong" }, { status: 500 })
  }
}
