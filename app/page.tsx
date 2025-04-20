"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, CheckCircle2, Sparkles } from "lucide-react"
import Link from "next/link"
import { useAuth } from "@/contexts/auth-context"

export default function Home() {
  const { user } = useAuth()

  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)]">
      {/* Hero Section */}
      <section className="flex-1 flex flex-col items-center justify-center text-center px-4 py-16 md:py-24">
        <div className="container max-w-4xl relative">
          <div className="absolute top-0 left-0 w-72 h-72 bg-primary/20 rounded-full filter blur-3xl opacity-30 animate-blob"></div>
          <div className="absolute bottom-0 right-0 w-72 h-72 bg-secondary/20 rounded-full filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
          <div className="absolute bottom-0 left-0 w-72 h-72 bg-accent/20 rounded-full filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
          <div className="relative">
            <h1 className="text-4xl md:text-6xl font-bold tracking-tighter mb-6">
              AI-Powered Interview Platform for{" "}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary via-accent to-secondary">
                Smarter Hiring
              </span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              InterviewIQ helps candidates prepare and companies hire better with AI-simulated interviews, real-time
              feedback, and comprehensive analytics.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {user ? (
                <Link href={user.role === "candidate" ? "/candidate-dashboard" : "/hr-dashboard"}>
                  <Button size="lg" className="glow-button">
                    Go to Dashboard
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              ) : (
                <>
                  <Link href="/signup">
                    <Button size="lg" className="glow-button">
                      Get Started
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                  <Link href="/login">
                    <Button size="lg" variant="outline">
                      Log In
                    </Button>
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 md:py-24 bg-muted/50">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Choose InterviewIQ?</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our platform offers unique features designed to transform the interview process for both candidates and
              recruiters.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-background p-6 rounded-xl border border-border/50">
              <div className="bg-primary/10 p-3 rounded-full w-fit mb-4">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2">AI-Powered Interviews</h3>
              <p className="text-muted-foreground">
                Practice with our AI interviewer that adapts questions based on your responses and provides real-time
                feedback.
              </p>
            </div>

            <div className="bg-background p-6 rounded-xl border border-border/50">
              <div className="bg-secondary/10 p-3 rounded-full w-fit mb-4">
                <CheckCircle2 className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="text-xl font-bold mb-2">Comprehensive Analytics</h3>
              <p className="text-muted-foreground">
                Get detailed insights on your performance, including strengths, areas for improvement, and personalized
                tips.
              </p>
            </div>

            <div className="bg-background p-6 rounded-xl border border-border/50">
              <div className="bg-accent/10 p-3 rounded-full w-fit mb-4">
                <CheckCircle2 className="h-6 w-6 text-accent" />
              </div>
              <h3 className="text-xl font-bold mb-2">Industry-Specific Questions</h3>
              <p className="text-muted-foreground">
                Access a vast library of interview questions tailored to your industry, role, and experience level.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 md:py-24">
        <div className="container max-w-4xl text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Transform Your Interview Experience?</h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join thousands of candidates and companies who are already using InterviewIQ to improve their hiring
            process.
          </p>
          {user ? (
            <Link href={user.role === "candidate" ? "/candidate-dashboard" : "/hr-dashboard"}>
              <Button size="lg" className="glow-button">
                Go to Dashboard
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          ) : (
            <Link href="/signup">
              <Button size="lg" className="glow-button">
                Get Started Today
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          )}
        </div>
      </section>
    </div>
  )
}
