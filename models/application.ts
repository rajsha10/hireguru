import mongoose from "mongoose";

export interface IApplication extends mongoose.Document {
  jobId: mongoose.Types.ObjectId;
  candidateId: mongoose.Types.ObjectId;
  status: "applied" | "review" | "interview" | "rejected" | "accepted";
  appliedDate: Date;
  interviewDate?: Date;
  interviewTime?: string;
  interviewScore?: number;
  feedback?: string;
  notes?: string;
}

const ApplicationSchema = new mongoose.Schema<IApplication>(
  {
    jobId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Job",
      required: [true, "Job ID is required"],
    },
    candidateId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: [true, "Candidate ID is required"],
    },
    status: {
      type: String,
      enum: ["applied", "review", "interview", "rejected", "accepted"],
      default: "applied",
    },
    appliedDate: {
      type: Date,
      default: Date.now,
    },
    interviewDate: {
      type: Date,
    },
    interviewTime: {
      type: String,
    },
    interviewScore: {
      type: Number,
      min: 0,
      max: 100,
    },
    feedback: {
      type: String,
    },
    notes: {
      type: String,
    },
  },
  { timestamps: true }
);

ApplicationSchema.index({ candidateId: 1, status: 1 });
ApplicationSchema.index({ jobId: 1, status: 1 });

const Application = mongoose.models.Application || mongoose.model<IApplication>("Application", ApplicationSchema);

export default Application;