import { NextRequest, NextResponse } from "next/server"
import connectToDatabase from "@/lib/mongodb"
import CVSummary from "@/models/cv-summary"

export const config = {
  api: {
    bodyParser: false,
  },
}

export async function POST(req: NextRequest) {
    try {
      const body = await req.json()
      const { name, role, summary } = body
  
      if (!name || !role || !summary) {
        return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
      }
  
      await connectToDatabase()
  
      const newEntry = await CVSummary.create({
        name,
        role,
        summary,
        filePath: "N/A",
        status: "completed",
      })
  
      return NextResponse.json({ summary: newEntry.summary }, { status: 200 })
    } catch (error: any) {
      return NextResponse.json({ error: error.message || "Internal Server Error" }, { status: 500 })
    }
  }
  
