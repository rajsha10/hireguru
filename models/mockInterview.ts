import mongoose from "mongoose";

export interface IMockInterview extends mongoose.Document {
  candidateId: mongoose.Types.ObjectId;
  jobTitle: string;
  category: string;
  date: Date;
  duration: number; // in minutes
  score: number;
  feedback: string;
  questions: Array<{
    question: string;
    answer?: string;
    feedback?: string;
    score?: number;
  }>;
  status: "scheduled" | "completed" | "cancelled";
}

const MockInterviewSchema = new mongoose.Schema<IMockInterview>(
  {
    candidateId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: [true, "Candidate ID is required"],
    },
    jobTitle: {
      type: String,
      required: [true, "Job title is required"],
    },
    category: {
      type: String,
      required: [true, "Interview category is required"],
      enum: ["Technical", "Behavioral", "Case Study", "General"],
    },
    date: {
      type: Date,
      required: [true, "Interview date is required"],
    },
    duration: {
      type: Number,
      required: [true, "Duration is required"],
      default: 30,
    },
    score: {
      type: Number,
      min: 0,
      max: 100,
    },
    feedback: {
      type: String,
    },
    questions: [
      {
        question: String,
        answer: String,
        feedback: String,
        score: Number,
      },
    ],
    status: {
      type: String,
      enum: ["scheduled", "completed", "cancelled"],
      default: "scheduled",
    },
  },
  { timestamps: true }
);

MockInterviewSchema.index({ candidateId: 1, status: 1 });
MockInterviewSchema.index({ date: 1 });

const MockInterview = mongoose.models.MockInterview || mongoose.model<IMockInterview>("MockInterview", MockInterviewSchema);

export default MockInterview;