import mongoose from "mongoose";

const JobSchema = new mongoose.Schema({
  title: { 
    type: String, 
    required: [true, "Job title is required"],
    minlength: [3, "Job title must be at least 3 characters"]
  },
  department: { 
    type: String, 
    required: [true, "Department is required"],
    enum: ["Engineering", "Design", "Product", "Marketing", "Sales", "HR", "Finance"]
  },
  location: { 
    type: String, 
    required: [true, "Location is required"] 
  },
  type: { 
    type: String, 
    required: [true, "Job type is required"],
    enum: ["Full-time", "Part-time", "Contract", "Internship", "Temporary"]
  },
  description: { 
    type: String, 
    required: [true, "Description is required"],
    minlength: [50, "Description must be at least 50 characters"]
  },
  requirements: { 
    type: String, 
    required: [true, "Requirements are required"],
    minlength: [50, "Requirements must be at least 50 characters"]
  },
  companyName: { 
    type: String, 
    required: [true, "Company name is required"] 
  },
  postedByName: { 
    type: String, 
    required: [true, "Posted by name is required"] 
  },
  postedByDesignation: { 
    type: String, 
    required: [true, "Posted by designation is required"] 
  },
  createdAt: { 
    type: Date, 
    default: Date.now 
  },
  updatedAt: { 
    type: Date, 
    default: Date.now 
  },
  // Additional fields that might be useful
  status: { 
    type: String, 
    enum: ["active", "closed", "draft"], 
    default: "active" 
  },
  applicants: {
    type: Number,
    default: 0
  }
});

JobSchema.pre('save', function (next) {
    this.updatedAt = new Date();
    next();
  });  

// Check if the model exists before compiling it
const Job = mongoose.models.Job || mongoose.model('Job', JobSchema);

export default Job;