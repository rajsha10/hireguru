// app/api/my-summary/route.ts
import { NextRequest, NextResponse } from "next/server"
import connectToDatabase from "@/lib/mongodb"
import CVSummary from "@/models/cv-summary"
import { getUserFromSession } from "@/lib/get-user";

export async function GET(req: NextRequest) {
  try {
    const user = await getUserFromSession(req)
    if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })

    await connectToDatabase()
    const latestSummary = await CVSummary.findOne({ role: user.role })
      .sort({ createdAt: -1 })  
      .limit(1)

    if (!latestSummary) {
      return NextResponse.json({ summary: null }, { status: 200 })
    }

    return NextResponse.json({ summary: latestSummary.summary }, { status: 200 })
  } catch (err) {
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 })
  }
}
