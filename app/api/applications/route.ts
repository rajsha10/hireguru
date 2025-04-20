// File: /app/api/applications/route.js
import { NextResponse } from "next/server";
import connectToDatabase from "@/lib/mongodb";
import Application from "@/models/application";
import Job from "@/models/job";
import User from "@/models/user";

export async function POST(request) {
  try {
    await connectToDatabase();
    
    const body = await request.json();
    const { jobId, candidateId } = body;
    
    // Validate inputs
    if (!jobId || !candidateId) {
      return NextResponse.json({ error: "Job ID and Candidate ID are required" }, { status: 400 });
    }
    
    // Check if job exists
    const job = await Job.findById(jobId);
    if (!job) {
      return NextResponse.json({ error: "Job not found" }, { status: 404 });
    }
    
    // Check if job is active
    if (job.status !== "active") {
      return NextResponse.json({ error: "This job is not currently accepting applications" }, { status: 400 });
    }
    
    // Check if candidate exists
    const candidate = await User.findById(candidateId);
    if (!candidate || candidate.role !== "candidate") {
      return NextResponse.json({ error: "Candidate not found" }, { status: 404 });
    }
    
    // Check if application already exists
    const existingApplication = await Application.findOne({ jobId, candidateId });
    if (existingApplication) {
      return NextResponse.json({ 
        error: "You have already applied to this job",
        applicationId: existingApplication._id 
      }, { status: 400 });
    }
    
    // Create new application
    const application = new Application({
      jobId,
      candidateId,
      status: "applied",
      appliedDate: new Date(),
    });
    
    await application.save();
    
    // Increment applicants count for the job
    job.applicants = (job.applicants || 0) + 1;
    await job.save();
    
    return NextResponse.json({ 
      message: "Application submitted successfully",
      applicationId: application._id,
      jobTitle: job.title,
      company: job.department, // or whatever field contains company name
      appliedDate: application.appliedDate
    }, { status: 201 });
    
  } catch (error) {
    console.error("Error submitting application:", error);
    return NextResponse.json(
      { error: "Internal server error" }, 
      { status: 500 }
    );
  }
}