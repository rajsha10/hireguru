import mongoose from "mongoose"

export interface ICVSummary extends mongoose.Document {
  name: string
  role: string
  summary: string
  filePath: string
  status: "processing" | "completed" | "failed"
  error?: string
  createdAt: Date
  updatedAt: Date
}

const CVSummarySchema = new mongoose.Schema<ICVSummary>(
  {
    name: {
      type: String,
      required: [true, "Please provide a contestant name"],
      maxlength: [100, "Name cannot be more than 100 characters"],
    },
    role: {
      type: String,
      required: [true, "Please provide a role"],
      maxlength: [100, "Role cannot be more than 100 characters"],
    },
    summary: {
      type: String,
      default: "",
    },
    filePath: {
      type: String,
      required: [true, "File path is required"],
    },
    status: {
      type: String,
      enum: ["processing", "completed", "failed"],
      default: "processing",
    },
    error: {
      type: String,
    },
  },
  { timestamps: true },
)

// Prevent mongoose from creating a new model if it already exists
export default mongoose.models.CVSummary || mongoose.model<ICVSummary>("CVSummary", CVSummarySchema)
