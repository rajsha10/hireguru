// File: /app/api/jobs/[id]/route.ts

import { NextRequest, NextResponse } from "next/server";
import connectToDatabase from "@/lib/mongodb";
import Job from "@/models/job";

interface Params {
  params: { id: string };
}

// GET a single job by ID
export async function GET(req: NextRequest, { params }: Params) {
  try {
    await connectToDatabase();
    const job = await Job.findById(params.id);

    if (!job) {
      return NextResponse.json({ message: "Job not found" }, { status: 404 });
    }

    return NextResponse.json(job);
  } catch (error: any) {
    console.error("Error fetching job:", error);
    return NextResponse.json(
      { message: error.message || "Failed to fetch job" },
      { status: 500 }
    );
  }
}

// UPDATE a job by ID
export async function PUT(req: NextRequest, { params }: Params) {
  try {
    await connectToDatabase();
    const data = await req.json();

    const job = await Job.findByIdAndUpdate(params.id, data, {
      new: true,
      runValidators: true,
    });

    if (!job) {
      return NextResponse.json({ message: "Job not found" }, { status: 404 });
    }

    return NextResponse.json(job);
  } catch (error: any) {
    console.error("Error updating job:", error);
    return NextResponse.json(
      { message: error.message || "Failed to update job" },
      { status: 500 }
    );
  }
}

// DELETE a job by ID
export async function DELETE(req: NextRequest, { params }: Params) {
  try {
    await connectToDatabase();
    const job = await Job.findByIdAndDelete(params.id);

    if (!job) {
      return NextResponse.json({ message: "Job not found" }, { status: 404 });
    }

    return NextResponse.json({ message: "Job deleted successfully" });
  } catch (error: any) {
    console.error("Error deleting job:", error);
    return NextResponse.json(
      { message: error.message || "Failed to delete job" },
      { status: 500 }
    );
  }
}
